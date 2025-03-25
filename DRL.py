import torch
from PPOAgent import PPOAgent
from torch.optim import Adam
import gymnasium as gym
from CustomEnv import MyEnv
from TrajData import TrajData
# from ReplayBuffer import ReplayBuffer

class DRL:
    def __init__(self):

        self.n_envs = 16 # for testing
        self.n_steps = 500
        self.n_obs = 35+34+160 # model.qvel+model.qpos+cinert
        self.n_actions = 28 # 28 actuators, each action modeled as a gaussian


        self.envs = gym.vector.SyncVectorEnv([lambda: MyEnv("test.xml") for _ in range(self.n_envs)])

        # self.replay_buffer = ReplayBuffer()
        self.traj_data = TrajData(self.n_steps,self.n_envs,self.n_obs,self.n_actions, 0) # placeholder for ind, maybe no longer in use
  
        self.agent = PPOAgent(self.n_obs, n_actions=self.n_actions, a_lambda=.95, gamma = .99)  
        self.optimizer = Adam(self.agent.parameters(), lr=1e-4)
        # self.writer = SummaryWriter(log_dir=f'runs/{self.agent.name}')


    def rollout(self, i):

        obs, _ = self.envs.reset() # obs, reset_info
        obs = torch.Tensor(obs)

        for t in range(self.n_steps):
            with torch.no_grad() if self.agent.name == 'PPO' else torch.enable_grad():
                actions, probs = self.agent.get_action(obs)
            log_probs = probs.log_prob(actions).sum(-1)
            next_obs, rewards, done, truncated, infos = self.envs.step(actions.numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            self.traj_data.store(t, obs, actions, rewards, log_probs, done)
            # self.replay_buffer.store(obs,actions,rewards,next_obs,done,traj_ind,t)
            obs = torch.Tensor(next_obs)
            

        self.traj_data.calc_returns()
        # self.traj_datas.append(traj_data)

        # self.writer.add_scalar("Reward", self.traj_data.rewards.mean(), i)
        # self.writer.flush()
        print("avg reward: ", self.traj_data.rewards.mean())


    def update(self):

        # A primary benefit of PPO is that it can train for
        # many epochs on 1 rollout without going unstable
        epochs = 10 if self.agent.name == 'PPO' else 1

        for _ in range(epochs):

            loss = self.agent.get_loss(self.traj_data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.traj_data.detach()
if __name__=="__main__":
    drl = DRL()
    drl.rollout(0)
    # print(drl.traj_data.states.shape)
    drl.update()

