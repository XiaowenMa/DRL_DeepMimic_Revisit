import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
from mocap.mocap import MocapDM
import random
import mujoco
from PIL import Image
import os
from scipy.spatial.transform import Rotation as R
import pyquaternion
from mocap.mocap_util import DOF_DEF, END_EFFECTORS, JOINT_WEIGHT, MUJOCO_PYBULLET_MAP, BODY_JOINTS, JOINT_RANGE
import math
import json
from utils.pd_control import PDController
from utils.transformations import quaternion_slerp
import torch

MOCAP_PATH = "mocap_data/walk_long.txt"
XML_PATH = "./test_humanoid.xml"
RENDER_FOLDER = "./mujoco_render"



class MyEnv(MujocoEnv):
    """TODO: observation space init"""
    def __init__(self, xml_path):
        self.width = 960 # size of renderer
        self.height = 640 # size of renderer
        self.metadata["render_modes"] = [
            "human",
            "rgb_array",
            "depth_array",
        ]
        super().__init__(XML_PATH,frame_skip = 5,observation_space = None)

        # for simplicity only use qpos and qvel for now
        # print(self.data.cinert.shape)
        self.obs_dim = self.data.qpos.shape[0] + self.data.qvel.shape[0]+self.data.cinert.flatten().shape[0]+1
        # print(self.obs_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )
        self.render_mode = "rgb_array"
        self.render_path = RENDER_FOLDER
        self.curr_frame_ind = 0

        self.mocap = MocapDM()
        self.load_mocap(MOCAP_PATH)
        # print("mocap data shape: ", self.mocap.data_config[0].shape)

        # random initialization of ref state
        self.reference_state_init()
        self.idx_curr = 0
        self.idx_tmp_count = 0
        self.step_len=1 # what's this

        self.curr_rollout_step = 0
        self.max_step = 64
        self.ref_frame_offset = 0

        self.load_endeffector()
        self.load_com_info()
        self.timestep = 0.002
        self.last_torque = np.zeros_like( self.data.qvel.shape)
        self.pd_controller = PDController()
    def load_com_info(self):
        self.com = np.loadtxt("mocap/com.txt", delimiter=",")

    def apply_pd_control(self,target_pos):
        if (self.curr_rollout_step%16)==0:
            curr_qpos = self.data.qpos[7:] # ignore root
            curr_qvel = self.data.qvel[6:] # ignore root

            target_qpos = target_pos
            target_qvel = (target_qpos-curr_qpos)/self.dt
            
            self.last_torque = self.pd_controller.pd_control(target_qpos,curr_qpos,target_qvel, curr_qvel)
        return self.last_torque

    def load_endeffector(self):
        self.end_effector = {}
        with open('mocap/end_effector.json','r') as f:
            self.end_effector = json.load(f)
        for key in self.end_effector.keys():
          self.end_effector[key] = np.array(self.end_effector[key])
        # print(len(self.end_effector['left_ankle']))

    # def set_renderer(self):
    #     from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

    #     self.mujoco_renderer = MujocoRenderer(
    #         self.model,
    #         self.data,
    #         width = self.width,
    #         height = self.height,
    #         camera_id=self.camera_id,
    #         camera_name=self.camera_name
    #     )
    """Try to load mocap data from mocap path"""
    def load_mocap(self,mocap_path):
        self.mocap.load_mocap(mocap_path)
        self.mocap_dt = self.mocap.dt
        self.mocap_data_len = len(self.mocap.data)
    
    def set_state(self, qpos, qvel):
        # print(qpos.shape,self.model.nq,qvel.shape,self.model.nv)
        super().set_state(qpos, qvel)

    """TODO: Change this if want to include more dim in observation"""
    def _get_obs(self):

        # cinert = self.data.cinert.flatten().copy()
        position = self.data.qpos.flatten().copy()
        velocity = self.data.qvel.flatten().copy()
        cinert = self.data.cinert.flatten().copy()
        # position = self.data.qpos.flatten().copy()[7:] # ignore root joint
        # velocity = self.data.qvel.flatten().copy()[6:] # ignore root joint
        phase = np.array([((self.curr_rollout_step//16+self.idx_init)%39)/38])
        return np.concatenate((position, velocity,cinert,phase))
    
    """Random initialization"""
    def reference_state_init(self):
        self.idx_init = random.randint(0, self.mocap_data_len//4)
        # self.idx_init = 0
        self.idx_curr = self.idx_init
        # self.idx_tmp_count = 0
        self.curr_rollout_step = 0
        self.ref_frame_offset = self.idx_init
        # print("ref state initialization succeeded.")

    """TODO: reward functions."""
    def step(self, action):
        self.step_len = 1
        step_times = 16 # to match 30fps mocap
        action=np.clip(action,JOINT_RANGE[:,0],JOINT_RANGE[:,1])
        torque = self.apply_pd_control(action)
        # print(torque)
        self.do_simulation(torque, step_times)
        observation = self._get_obs()
        
        truncated = (self.curr_rollout_step//16)>=self.max_step

        # reward_alive = 1.0
        reward_alive = 0.
        pos_reward = 0.
        vel_reward = 0.
        end_effector_reward = 0.

        # reward = self.calc_config_reward()
        # TODO: modify reward_alive
        pos_reward = self.calc_pos_reward()
        vel_reward = self.calc_vel_reward()
        end_effector_reward = self.calc_end_effector_reward()
        if self.curr_rollout_step!=0 and self.curr_rollout_step%16==0:
            # print("curr simul step:" ,self.curr_rollout_step)

            self.idx_curr += 1 # ind of curr_frame
            self.idx_curr = self.idx_curr % self.mocap_data_len
        self.curr_rollout_step+=step_times

        # reward = reward_alive
        info = dict()
        done = self.is_done() 
        if done:
            # reward_alive = -1 # magic number
            reward = 0
        else:
            # reward = 0.75*pos_reward+0.1*vel_reward+0.15*end_effector_reward+0.1*self.calc_com_reward()
            reward = 0.65*pos_reward+0.1*(self.calc_root_vel_reward()+vel_reward)+0.15*end_effector_reward+0.1*self.calc_com_reward()

        done = done | truncated

        return observation, reward, done, truncated, info

    def calc_end_effector_reward(self):
      total_diff = 0
      last_frame_ind = self.idx_curr
      next_frame_ind = (last_frame_ind+1)%self.mocap_data_len
      t = (self.curr_rollout_step%16)/16

      for joint in END_EFFECTORS:
          joint_indx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, joint)
          cur_xpos = self.data.xpos[joint_indx]
          target_xpos = (1-t)*self.end_effector[joint][last_frame_ind]+t*self.end_effector[joint][next_frame_ind]
          curr_diff = sum((cur_xpos-target_xpos)**2)
          total_diff+=curr_diff

      return math.exp(-40*total_diff)
    
    def calc_pos_reward(self, interpolate = True):
        assert len(self.mocap.data) != 0
        pos_diff = 0

        # root pos_diff(quat)
        curr_root_pos = pyquaternion.Quaternion(self.data.qpos[3:7]) # wxyz
        target_root_pos = pyquaternion.Quaternion(self.mocap.data_config[self.idx_curr][3:7]) # mujoco returns quat in wxyz
        root_weight = JOINT_WEIGHT['root']

        root_diff = (curr_root_pos*target_root_pos.conjugate).angle**2
        pos_diff+=root_diff*root_weight

        curr_pos_offset = 7
        for curr_joint in BODY_JOINTS:
            dof = DOF_DEF[curr_joint]
            scalar = JOINT_WEIGHT[curr_joint]
            if dof==1:

                curr_pos_diff = self.new_calc_pos_errs_interpolation(dof,curr_joint)
                pos_diff += curr_pos_diff*scalar

            if dof==3:

                curr_pos_diff = self.new_calc_pos_errs_interpolation(dof,curr_joint)
                # print(curr_pos_diff,pos_diff)
                pos_diff += curr_pos_diff*scalar

            curr_pos_offset+=dof

        pos_reward = math.exp(-2*pos_diff)

        return pos_reward


    def new_calc_pos_errs_interpolation(self, dof, joint_name):
        def slerp(q1, q2, t):
            q1 = list(q1[1:])+[q1[0]]
            q2 = list(q2[1:])+[q2[0]]
            q_slerp = list(quaternion_slerp(q1,q2,t))
            return [q_slerp[3]]+list(q_slerp[:3])
        last_frame_ind = self.idx_curr
        next_frame_ind = (last_frame_ind+1)%self.mocap_data_len
        t = (self.curr_rollout_step%16)/16
        # ref_ind = MUJOCO_PYBULLET_MAP[joint_name][1][0]
        mjc_ind = MUJOCO_PYBULLET_MAP[joint_name][0][0]
        ref_ind = mjc_ind
        if dof == 1:
            ref = (1-t)*self.mocap.data_config[last_frame_ind][ref_ind:ref_ind+dof]+t*self.mocap.data_config[next_frame_ind][ref_ind:ref_ind+dof]
            curr = self.data.qpos[mjc_ind]
            diff = ref-curr
            return diff[0]**2
        
        if dof==3:
            last_target = self.mocap.data_config[last_frame_ind][ref_ind:ref_ind+dof]
            quat1 = self.euler2quat(last_target[0],last_target[1],last_target[2],scalar_first=True)
            next_target = self.mocap.data_config[next_frame_ind][ref_ind:ref_ind+dof]
            quat2 = self.euler2quat(next_target[0],next_target[1],next_target[2],scalar_first=True)

            q_inpl = slerp(quat1,quat2,t)
            
            curr_pos = self.data.qpos[mjc_ind:mjc_ind+dof]
            q_curr = self.euler2quat(curr_pos[0],curr_pos[1],curr_pos[2],True)
            q_inpl = pyquaternion.Quaternion(q_inpl[3],q_inpl[0],q_inpl[1],q_inpl[2]).normalised
            q_curr = pyquaternion.Quaternion(q_curr[3],-q_curr[0],-q_curr[1],-q_curr[2]).normalised
            return (q_inpl*q_curr).angle**2
        return 0
    
    def calc_root_vel_reward(self):
        
        target_root_vel = self.mocap.data_vel[self.idx_curr][:3]
        curr_root_vel = self.data.qvel[:3]
        vel_diff = sum((curr_root_vel-target_root_vel)**2)
        vel_reward = math.exp(-1*vel_diff)
        # print(f"root vel: {vel_reward}")
        return vel_reward

    def calc_vel_reward(self):
        assert len(self.mocap.data) != 0

        vel_diff = 0
        curr_vel_offset = 6
        last_frame_ind = self.idx_curr
        next_frame_ind = (last_frame_ind+1)%self.mocap_data_len
        t = (self.curr_rollout_step%16)/16
        for curr_joint in BODY_JOINTS:
            dof = DOF_DEF[curr_joint]
            scalar = JOINT_WEIGHT[curr_joint]
            if dof==1:
                target = (1-t)*self.mocap.data_vel[last_frame_ind][curr_vel_offset:curr_vel_offset+dof]+t*self.mocap.data_vel[next_frame_ind][curr_vel_offset:curr_vel_offset+dof]
                curr_vel_diff = (self.data.qvel[curr_vel_offset:curr_vel_offset+dof]-target)[0]**2
                vel_diff += curr_vel_diff*scalar

            if dof==3:

                target = (1-t)*self.mocap.data_vel[last_frame_ind][curr_vel_offset:curr_vel_offset+dof]+t*self.mocap.data_vel[next_frame_ind][curr_vel_offset:curr_vel_offset+dof]
                curr_vel_diff = sum((self.data.qvel[curr_vel_offset:curr_vel_offset+dof]-target)**2)
                vel_diff += curr_vel_diff*scalar

            curr_vel_offset+=dof

        vel_reward = math.exp(-.1*vel_diff)

        return vel_reward

    def calc_vel_errs(self,curr_vel, target_vel):
        # both in quat
        return sum((curr_vel-target_vel)**2)

    def calc_com_reward(self):

        last_frame_ind = self.idx_curr
        next_frame_ind = (last_frame_ind+1)%self.mocap_data_len
        t = (self.curr_rollout_step%16)/16

        body_mass = self.model.body_mass.reshape(-1,1)
        total_mass = np.sum(body_mass)
        com = np.sum(body_mass*self.data.xpos)/total_mass
        target_com = self.com[last_frame_ind*3:last_frame_ind*3+3]*(1-t)+t*self.com[next_frame_ind*3:next_frame_ind*3+3]
        return math.exp(-10*sum((com-target_com)**2))
    
        # # match root pos
        # last_frame_ind = self.idx_curr
        # target_root_pos = self.mocap.data_config[last_frame_ind][:2]
        # # print(self.mocap.data_config[last_frame_ind][0])
        # curr_root_pos = self.data.xpos[1][:2]
        # return math.exp(-10*sum((curr_root_pos-target_root_pos)**2))


    def euler2quat(self,x,y,z,scalar_first = True):
        r = R.from_euler('xyz', [x, y, z], degrees=False)
    
        # Convert to quaternion
        quaternion = r.as_quat()  # returns [x, y, z, w] (vector part first) # need to check if this is consistent with 
        return quaternion
    '''
    alive reward: Check z position of weighted CoM across all body elements.
    TODO: Other source?
    '''
    def is_done(self):
        mass = np.expand_dims(self.model.body_mass, 1)
        xpos = self.data.xpos #CoM of all bodies in global coordinate
        z_com = (np.sum(mass * xpos, 0) / np.sum(mass))[2]
        done = bool((z_com < 0.7) or (z_com > 2.0))
        return done
    
    def reset_model(self):
        self.reference_state_init()
        qpos = self.mocap.data_config[self.idx_init]
        qvel = self.mocap.data_vel[self.idx_init]
        # qvel = self.init_qvel
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        self.idx_tmp_count = -self.step_len
        self.prev_pos = self.data.xpos[1]
        return observation

    def reset_model_init(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()
    
    def save_render_image(self,image,ind):
        outpath = os.path.join(RENDER_FOLDER,f"{ind}.png")
        im = Image.fromarray(image)
        im.save(outpath)

    
if __name__=="__main__":
    testEnv = MyEnv("")
    assert testEnv is not None, "No env created"
    print("Successfully created env.")
    

        


    
    