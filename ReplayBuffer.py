import numpy as np
from copy import deepcopy
import torch

class ReplayBuffer(object):

    def __init__(self,buffer_size = 1024, batch_size = 128, state_dim = 34+35+160,action_dim = 28):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_dim =  state_dim
        self.action_dim = action_dim
        self.states = torch.zeros((buffer_size,state_dim))
        self.actions = torch.zeros((buffer_size,action_dim))
        self.rewards = torch.zeros((buffer_size,1))
        self.not_dones = torch.zeros((buffer_size,1))
        self.log_probs = torch.zeros((buffer_size,1))
        # self.advantages = torch.zeros((buffer_size,1))

    def store(self, t, states, actions, rewards, log_probs, dones, ):
        self.states[t] = states
        self.actions[t] = actions
        self.rewards[t] = torch.Tensor(rewards)
        self.log_probs[t] = log_probs
        self.not_dones[t] = 1 - torch.Tensor(dones)

    def sample(self):
        indices = np.random.randint(0,self.batch_size,size=self.batch_size)
        # return self.states[indices],self.actions[indices],self.rewards[indices]
        return indices
    
    def calc_returns(self,gamma = 0.99):
        self.returns = deepcopy(self.rewards)
        for t in reversed(range(self.buffer_size-1)):
            self.returns[t] += self.returns[t+1] * self.not_dones[t] * gamma

        # normalize returns
        self.returns = (self.returns - self.returns.mean()) / (self.returns.std() + 1e-8)

    def clear(self):
        self.states = torch.zeros((self.buffer_size,self.state_dim))
        self.actions = torch.zeros((self.buffer_size,self.action_dim))
        self.rewards = torch.zeros((self.buffer_size,1))
        self.not_dones = torch.zeros((self.buffer_size,1))
        self.log_probs = torch.zeros((self.buffer_size,1))
        # self.advantages = torch.zeros((self.buffer_size,1))