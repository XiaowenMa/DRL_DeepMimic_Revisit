import torch
from PPOAgent import PPOAgent
from torch.optim import Adam
import gymnasium as gym
from CustomEnv import MyEnv
from TrajData import TrajData
from ReplayBuffer import ReplayBuffer
import numpy as np
from tqdm import tqdm

N = 2048
OBS_DIM = 35+34+160+1
ACT_DIM = 28

class DRL:
    def __init__(self):

        self.n_envs = 32 
        self.n_steps = 64
        self.n_obs = OBS_DIM # model.qvel+model.qpos+cinert
        self.n_actions = ACT_DIM # 28 actuators, each action modeled as a gaussian

        self.envs = gym.vector.SyncVectorEnv([lambda: MyEnv("test_humanoid.xml") for _ in range(self.n_envs)])

        # self.replay_buffer = ReplayBuffer()
        self.traj_data = TrajData(self.n_steps,self.n_envs,self.n_obs,self.n_actions, 0) # placeholder for ind, maybe no longer in use
        # self.replay_buffer = ReplayBuffer(buffer_size=2048,batch_size=256)
  
        self.agent = PPOAgent(self.n_obs, n_actions=self.n_actions, a_lambda=.95, gamma = .99)  
        # self.optimizer = Adam(self.agent.parameters(), lr=1e-4)
        self.actor_optimizer = Adam(list(self.agent.policy.parameters())+list(self.agent.policy_out.parameters()), lr=5e-5)
        self.critic_optimizer = Adam(list(self.agent.value.parameters())+list(self.agent.value_out.parameters()),lr=1e-4)

        self.value_losses = []
        self.policy_losses = []
        self.rewards_history = []

    
    def calc_gaes(self,gamma = 0.99, a_lambda=0.95):

        T = self.traj_data.n_steps

        self.advantages = torch.zeros_like(self.traj_data.rewards)
        self.values = torch.zeros_like(self.traj_data.rewards)

        self.values[-1] = self.agent.value_forward(self.traj_data.states[-1]).flatten()
        # calc advantages
        gae = torch.zeros_like(self.values[-1])
        for t in range(T-2,-1,-1):

          value = self.agent.value_forward(self.traj_data.states[t]).flatten()
          self.values[t] = value

          next_value = self.values[t+1]*(self.traj_data.not_dones[t])
          delta = self.traj_data.rewards[t]+gamma*next_value-value
          # print(traj_data.not_dones[t])
          gae = delta + gamma*a_lambda*self.traj_data.not_dones[t]*gae
          self.advantages[t] = gae

        self.advantages = self.advantages * self.traj_data.not_dones
        # print(self.traj_data.not_dones.sum(dim=0))

    def get_avg_loss(self):
        return self.avg_policy_loss,self.avg_value_loss
    
    def get_avg_reward(self):
        return self.avg_reward
    
    def get_avg_sim_steps(self):
        return self.avg_sim_steps
    
    def rollout(self, i):
        obs, _ = self.envs.reset() # obs, reset_info
        obs = torch.Tensor(obs)
        buffer_ets = [-1]*self.n_envs

        for t in range(self.n_steps):

            with torch.no_grad() if self.agent.name == 'PPO' else torch.enable_grad():
                actions, probs = self.agent.get_action(obs)
            # print(actions.shape)
            log_probs = probs.log_prob(actions).sum(-1)

            next_obs, rewards, done, truncated, infos = self.envs.step(actions.numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            self.traj_data.store(t, obs, actions, rewards, log_probs, done)
            # self.traj_data.store(t, obs_norm, actions, rewards, log_probs, done)
            for i in range(self.n_envs):
                if done[i] and buffer_ets[i]==-1:
                    buffer_ets[i]=t

            for i in range(self.n_envs):
                tt = buffer_ets[i]
                if tt!=-1:
                    self.traj_data.not_dones[tt:,i] = 0
            
            obs = torch.Tensor(next_obs)

        self.traj_data.calc_returns()
        self.calc_gaes()

        self.avg_reward = (self.traj_data.rewards*self.traj_data.not_dones).mean()
        self.avg_steps = np.mean((np.array(buffer_ets)+64)%64)

    # def rollout(self, i):

    #     obs, _ = self.envs.reset() # obs, reset_info
    #     obs = torch.Tensor(obs)

    #     for t in range(self.n_steps):
    #         with torch.no_grad() if self.agent.name == 'PPO' else torch.enable_grad():
    #             actions, probs = self.agent.get_action(obs)
    #         log_probs = probs.log_prob(actions).sum(-1)
    #         next_obs, rewards, done, truncated, infos = self.envs.step(actions.numpy())
    #         done = done | truncated  # episode doesnt truncate till t = 500, so never
    #         self.traj_data.store(t, obs, actions, rewards, log_probs, done)
    #         # self.replay_buffer.store(obs,actions,rewards,next_obs,done,traj_ind,t)
    #         obs = torch.Tensor(next_obs)
            

    #     self.traj_data.calc_returns()
    #     # self.traj_datas.append(traj_data)

    #     # self.writer.add_scalar("Reward", self.traj_data.rewards.mean(), i)
    #     # self.writer.flush()
    #     print("avg reward: ", self.traj_data.rewards.mean())



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

    def update3(self):
        epochs = 10 
        epoch_policy_loss = []
        epoch_value_loss = []

        T, N = self.traj_data.states.shape[:2]

        for e in range(epochs):
            # print(f"epoch {e}")
            self.calc_gaes()
            flat_states = self.traj_data.states.reshape(T * N, -1)
            flat_actions = self.traj_data.actions.reshape(T * N, -1)
            flat_log_probs = self.traj_data.log_probs.reshape(T * N)
            flat_rewards = self.traj_data.rewards.reshape(T*N)
            flat_returns = self.traj_data.returns.reshape(T * N)
            flat_advantages = self.advantages.reshape(T * N)
            flat_values = self.values.reshape(T*N)

            flat_not_dones = self.traj_data.not_dones.reshape(T*N)

            valid_indices = torch.where(flat_not_dones)[0]


            valid_adv = flat_advantages[valid_indices]
            # normalize advantages
            flat_advantages[valid_indices] = (flat_advantages[valid_indices]-torch.mean(valid_adv))/(torch.std(valid_adv)+1e-8)
            flat_advantages = torch.clamp(flat_advantages,-4,4)

            # print("Adv range:", flat_advantages.min().item(), flat_advantages.max().item())
            # print("Return range:", flat_returns.min().item(), flat_returns.max().item())

            batch_size = 256
            indices = torch.randperm(T * N)
            sub_vloss = []
            sub_ploss= []

            for i in range(0, T * N, batch_size):
                batch_idx = indices[i:i+batch_size]
                
                batch_states = flat_states[batch_idx]
                batch_actions = flat_actions[batch_idx]
                batch_rewards = flat_rewards[batch_idx]
                batch_returns = flat_returns[batch_idx]
                batch_advantages = flat_advantages[batch_idx]
                batch_log_probs = flat_log_probs[batch_idx]
                batch_values = flat_values[batch_idx]
                batch_not_dones = flat_not_dones[batch_idx]
                # print(batch_returns.shape,batch_values.shape)

                policy_loss,value_loss = self.agent.get_loss3(batch_states,batch_actions,batch_rewards,batch_returns,batch_advantages,batch_values,batch_log_probs,batch_not_dones)
                epoch_policy_loss.append(policy_loss.item())
                epoch_value_loss.append(value_loss.item())
                sub_ploss.append(policy_loss)
                sub_vloss.append(value_loss)
            value_loss = torch.stack(sub_vloss).mean()
            policy_loss = torch.stack(sub_ploss).mean()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            value_loss.backward(retain_graph = True)
            policy_loss.backward()
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            # print(value_loss.item(),policy_loss.item())
            
    
        self.avg_policy_loss = np.mean(epoch_policy_loss)
        self.avg_value_loss = np.mean(epoch_value_loss)
        self.traj_data.detach()

    def save_training_data(self):
        with open("reward.txt",'w') as f:
            f.write(" ".join(map(str, self.rewards_history)))
        with open("value_loss.txt","w") as f:
            f.write(" ".join(map(str, self.value_losses)))
        with open("policy_loss.txt","w") as f:
            f.write(" ".join(map(str, self.policy_losses)))


if __name__=="__main__":
    drl = DRL()
    for i in range(1):
        drl.rollout(i)
        drl.update3()



