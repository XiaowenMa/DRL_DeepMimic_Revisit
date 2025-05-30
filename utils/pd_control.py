import numpy as np
import torch

from mocap.mocap_util import DOF_DEF, BODY_JOINTS, PARAMS_KP_KD
# PARAMS_KP_KD = {"chest": [1000, 100], "neck": [100, 10], "right_shoulder": [400, 40], "right_elbow": [300, 30], 
#         "left_shoulder": [400, 40], "left_elbow": [300, 30], "right_hip": [500, 50], "right_knee": [500, 50], 
#         "right_ankle": [400, 40], "left_hip": [500, 50], "left_knee": [500, 50], "left_ankle": [400, 40]}


class PDController:
    def __init__(self):
        self.kp = torch.zeros(28)
        self.kd = torch.zeros(28)
        self.get_pd_arrays()

    def get_pd_arrays(self):
        curr_offset = 0
        for joint in BODY_JOINTS:
            if DOF_DEF[joint] == 1:
                self.kp[curr_offset],self.kd[curr_offset] = PARAMS_KP_KD[joint]
                curr_offset+=1
            elif DOF_DEF[joint]==3:
                self.kp[curr_offset:curr_offset+3],self.kd[curr_offset:curr_offset+3] = PARAMS_KP_KD[joint]
                curr_offset+=3
        # self.kp = torch.Tensor(self.kp)
        # self.kd = torch.Tensor(self.kd)


    def pd_control(self,target_qpos, curr_qpos, target_qvel,curr_qvel):
        # print(type(target_qpos))
        torque = self.kp * (target_qpos - curr_qpos) - self.kd * (target_qvel-curr_qvel)

        return torque