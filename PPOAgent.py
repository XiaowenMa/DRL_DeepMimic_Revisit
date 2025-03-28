import torch
from torch import nn
from torch.optim import Adam
from copy import deepcopy
# from torch.distributions import categorical
from torch.distributions import Normal
from TrajData import TrajData
from ReplayBuffer import ReplayBuffer

class PPOAgent(nn.Module):
    def __init__(self, n_obs, n_actions, a_lambda, gamma=.99, epochs=10): # for this model, ? actuator
        super().__init__()
        self.name = 'PPO'
        self.epochs = epochs

        torch.manual_seed(0)  # needed before network init for fair comparison

        # todo: student code here
        # self.policy = None  # replace
        self.policy = nn.Sequential(
            nn.Linear(n_obs,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,n_actions)
        )

        # self.value = None  # replace
        self.value = nn.Sequential(
            nn.Linear(n_obs,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )
        # TODO: directly use Adam for now, the paper use soft update
        # self.policy_old = deepcopy(self.policy)
        # self.value_old = deepcopy(self.value)

        self.a_lambda = a_lambda
        self.gamma = gamma

        # self.advantages = None

        # self.init_weight()

        # end student code
    def init_weight(self):
        for m in self.policy.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.value.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # def compute_advantages(self, traj_data:TrajData):
    #     T = traj_data.n_steps
    #     # print(traj_data.not_dones.shape,traj_data.not_dones[-1])

    #     advantages = torch.zeros_like(traj_data.rewards)
    #     values = torch.zeros_like(traj_data.rewards)
    #     values[-1] = self.value(traj_data.states[-1]).flatten()
    #     # calc advantages
    #     gae = torch.zeros_like(values[-1])
    #     for t in range(T-2,-1,-1):
    #       value = self.value(traj_data.states[t]).flatten()
    #       values[t] = value

    #       next_value = values[t+1]*(traj_data.not_dones[t])
    #       delta = traj_data.rewards[t]+self.gamma*next_value-value
    #       # print(traj_data.not_dones[t])
    #       gae = delta + self.gamma*self.a_lambda*traj_data.not_dones[t]*gae
    #       advantages[t] = gae

    #     self.advantages = advantages

    
    def calcualte_gae(self, replay_buffer:ReplayBuffer, gamma = .99, a_lambda = .95):
        T = replay_buffer.buffer_size
        # print(traj_data.not_dones.shape,traj_data.not_dones[-1])

        advantages = torch.zeros_like(replay_buffer.rewards)
        values = torch.zeros_like(replay_buffer.rewards)
        values[-1] = self.value(replay_buffer.states[-1]).flatten()
        # calc advantages
        gae = torch.zeros_like(values[-1])
        for t in range(T-2,-1,-1):
          value = self.value(replay_buffer.states[t]).flatten()
          values[t] = value

          next_value = values[t+1]*(replay_buffer.not_dones[t])
          delta = replay_buffer.rewards[t]+gamma*next_value-value
          gae = delta + gamma*a_lambda*replay_buffer.not_dones[t]*gae
          advantages[t] = gae
          return values, advantages

        
    def get_value_loss(self, replay_buffer, batch_indices):
        values,advantages = self.calcualte_gae(replay_buffer)
        self.advantages = advantages
        value_loss = torch.mean((replay_buffer.returns[batch_indices]-values[batch_indices])**2)
        return value_loss
    
    def get_policy_loss(self, replay_buffer,batch_indices, epsilon=.2):
        #   print(self.advantages)
          batch_states = replay_buffer.states[batch_indices]
        #   print(batch_states.shape)
          _,probs = self.get_action(batch_states)
          p = probs.log_prob(replay_buffer.actions[batch_indices]).sum(-1)
        #   print(self.advantages[batch_indices].shape)
          ratio = torch.exp(p-replay_buffer.log_probs[batch_indices].squeeze(-1))
          print("adv: ", torch.max(self.advantages[batch_indices]),torch.min(self.advantages[batch_indices]))
        #   print(ratio.shape)
          policy_loss=torch.mean(torch.min(ratio*self.advantages[batch_indices].detach().squeeze(-1),self.clip(ratio,epsilon)*self.advantages[batch_indices].detach().squeeze(-1)))
          return policy_loss


    """
        Version 1: Use entire traj as in HW1, in the original paper, in each update step, the authors sampled from memory(traj?)
    """
    def get_loss(self, traj_data:TrajData, epsilon=.2):
        policy_loss = []
        value_loss = []
        T = traj_data.n_steps
        # print(traj_data.not_dones.shape,traj_data.not_dones[-1])

        advantages = torch.zeros_like(traj_data.rewards)
        values = torch.zeros_like(traj_data.rewards)
        values[-1] = self.value(traj_data.states[-1]).flatten()
        # calc advantages
        gae = torch.zeros_like(values[-1])
        for t in range(T-2,-1,-1):
          value = self.value(traj_data.states[t]).flatten()
          values[t] = value

          next_value = values[t+1]*(traj_data.not_dones[t])
          delta = traj_data.rewards[t]+self.gamma*next_value-value
          # print(traj_data.not_dones[t])
          gae = delta + self.gamma*self.a_lambda*traj_data.not_dones[t]*gae
          advantages[t] = gae


        for t in range(T):
        #   print(torch.mean(traj_data.returns[t]),torch.mean(values[t]))
          value_loss.append((traj_data.returns[t]-values[t])**2) # need to calc value_loss before t_prime loop

          _,probs = self.get_action(traj_data.states[t])
          p = probs.log_prob(traj_data.actions[t]).sum(-1)
          ratio = torch.exp(p-traj_data.log_probs[t])
        #   print("adv: ", advantages[t].shape)
          policy_loss.append(torch.min(ratio*advantages[t],self.clip(ratio,epsilon)*advantages[t]))

        policy_loss = -torch.stack(policy_loss).mean()
        value_loss = torch.stack(value_loss).mean()

        loss = policy_loss+value_loss  # replace

        print(policy_loss.item(),value_loss.item())
        return loss
    

    def clip(self,ratio, epsilon):
        return torch.clamp(ratio, 1-epsilon,1+epsilon)

    def get_action(self, obs):
        """TODO: update the clamp part, clamp the log_std_dev instead of std_dev"""
        # mean, log_std_dev = self.policy(obs).chunk(2, dim=-1)
        mean= self.policy(obs)
        std_dev = torch.ones_like(mean)

        # print("loc min:", mean.min().item(), "loc max:", mean.max().item())
        # print("NaNs in loc:", torch.isnan(mean).sum().item())
        # print("Infs in loc:", torch.isinf(mean).sum().item())
        # mean = torch.tanh(mean)
        # mean = -100+200*(mean+1)/2 # suppose torque -100 - 100
        # print(max(mean),min(mean))

        # std_dev = log_std_dev.exp().clamp(.2, 2)
        dist = Normal(mean,std_dev)
        action = dist.rsample()
        # print(action.shape)
        # log_prob = dist.log_prob(action).sum(-1)
        return action,dist