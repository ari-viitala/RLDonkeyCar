import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader
import random
import datetime
import time

from collections import deque

import cv2
import numpy as np

from .modules import MLP
from .ae import AE 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """ Gaussian Policy """
    def __init__(self, input_size, act_size, hidden_size):
        super().__init__()
        self.act_size = act_size
        self.net = MLP(input_size, act_size * 2, hidden_size)
        
    def forward(self, state):
        x = self.net(state)
        mean, log_std = x[:, :self.act_size], x[:, self.act_size:]

        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        normal = Normal(mean, log_std.exp())
        x = normal.rsample()

        # Enforcing action bounds
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action**2 + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def select_action(self, state):
    
        action, _ = self.sample(state)
        
        return action[0].detach().cpu().numpy()

class Critic(nn.Module):
    """ Twin Q-networks """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net1 = MLP(input_size, 1, hidden_size)
        self.net2 = MLP(input_size, 1, hidden_size)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.net1(state_action), self.net2(state_action)

class ReplayBuffer():

    def __init__(self, length):

        self.buffer = deque(maxlen=length)

    def loader(self, batch_size, gradient_steps):
        
        size = batch_size * gradient_steps
        seed = random.randint(0, 100000)
        sets = [random.Random(seed).choices(x, k=size) for x in zip(*self.buffer)]
        
        loaders = [DataLoader(s, batch_size) for s in sets]
        
        return loaders

    def sample(self, amount):
        
        return random.sample(self.buffer, amount)

    def push(self, state):
        im, control_history = state[0]
        action = state[1]
        reward = state[2]
        next_im, next_history = state[3]
        not_done = state[4]

        self.buffer.append([torch.FloatTensor(x).to(device) for x in [im, control_history, action, reward, next_im, next_history, not_done]])


class AE_SAC:

    def __init__(self, parameters={}):

        params = {
            "gamma": 0.99,
            "tau": 0.005,
            "lr": 0.0001,
            "replay_buffer_size": 1000000,
            "hidden_size": 100,
            "batch_size": 64,
            "n_episodes": 1000,
            "n_random_episodes": 10,
            "discount": 0.9,
            "horizon": 50,
            "im_rows": 40,
            "im_cols": 40,
            "linear_output": 64,
            "target_entropy": -2
        }
          
        for arg in parameters["sac"].keys():
             params[arg] = parameters["sac"][arg]


        self.gamma = params["gamma"]
        self.tau = params["tau"]
        self.lr = params["lr"]
        self.replay_buffer_size = params["replay_buffer_size"]
        self.hidden_size = params["hidden_size"]
        self.batch_size = params["batch_size"]
        self.n_episodes = params["n_episodes"]
        self.n_random_episodes = params["n_random_episodes"]
        self.discount = params["discount"]
        self.horizon = params["horizon"]
        self.im_rows = params["im_rows"]
        self.im_cols = params["im_cols"]
        self.linear_output = params["linear_output"]
        self.target_entropy = params["target_entropy"]
        self.act_size = 2

        self.encoder = AE(parameters["ae"])

        self.critic = Critic(self.linear_output + self.act_size, self.hidden_size).to(device)

        critic_parameters = list(self.critic.parameters()) + list(self.encoder.encoder.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=self.lr)

        self.critic_target = Critic(self.linear_output + self.act_size, self.hidden_size).to(device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor = Actor(self.linear_output, self.act_size, self.hidden_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.replay_buffer = ReplayBuffer(length=self.replay_buffer_size)


    def update_parameters(self, gradient_steps):
        
        training_start = time.time_ns()
#        k = min(self.batch_size, len(self.replay_buffer.buffer))
#        batch = self.replay_buffer.sample(k)

        loaders = self.replay_buffer.loader(self.batch_size, gradient_steps)
#       im, control, action, reward, next_im, next_control, not_done = [torch.FloatTensor(x).to(device) for x in zip(*batch)]
        iters = [iter(l) for l in loaders]
        
        for i in range(len(loaders[0])):
            #print("Step: {}".format(i))
            step_start = time.time_ns()
            
            im, control, action, reward, next_im, next_control, not_done = [next(it) for it in iters]
            
            #print("Preprocessing: {:.2f}ms".format((time.time_ns() - time_start) / 1e6))
            
            embedding_start = time.time_ns()

            embedding, log_sigma = self.encoder.encoder(im)
            state = torch.cat([embedding, control], axis=1)
            
            with torch.no_grad():
                next_embedding, next_log_sigma = self.encoder.encoder_target(next_im)
            
            next_state = torch.cat([next_embedding, next_control], axis=1)

            #print("Embedding: {:.2f}ms".format((time.time_ns() - embedding_start) / 1e6))

            alpha = self.log_alpha.exp().item()

            # Update critic

            train_start = time.time_ns()

            with torch.no_grad():
                next_action, next_action_log_prob = self.actor.sample(next_state)
                q1_next, q2_next = self.critic_target(next_state, next_action)
                q_next = torch.min(q1_next, q2_next)
                value_next = q_next - alpha * next_action_log_prob
                q_target = reward + not_done * self.gamma * value_next

            q1, q2 = self.critic(state, action)
            q1_loss = 0.5*F.mse_loss(q1, q_target)
            q2_loss = 0.5*F.mse_loss(q2, q_target)
            critic_loss = q1_loss + q2_loss

            #encoder loss
            #next_encoder_loss = self.encoder.loss(next_embedding, log_sigma) 

            #loss = critic_loss + encoder_loss

            self.critic_optimizer.zero_grad()

            #print("Critic loss: {:.2f}".format(critic_loss.item()))
            
            critic_loss.backward()
            #next_encoder_loss.backward()

            self.critic_optimizer.step()

            
            embedding, log_sigma = self.encoder.encoder(im)
            encoder_loss = self.encoder.loss(embedding, log_sigma)

            self.encoder.optimizer.zero_grad()
            
            encoder_loss.backward()

            self.encoder.optimizer.step()
            
            #print("Encoder loss: {:.2f}".format(encoder_loss.item()))
            

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_((1.0-self.tau)*target_param.data + self.tau*param.data)

            self.encoder.update_encoder_target()

            #print("Critic training: {:.2f}ms".format((time.time_ns() -train_start)/1e6))

            actor_start = time.time_ns()

            # Update actor
            state = state.detach()

            action_new, action_new_log_prob = self.actor.sample(state)
            q1_new, q2_new = self.critic(state, action_new)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (alpha*action_new_log_prob - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update alpha

            alpha_loss = -(self.log_alpha * (action_new_log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            #print("Actor training: {:.2f}".format((time.time_ns() - actor_start) / 1e6))
            #print("Total time: {:.2f}ms".format((time.time_ns() - time_start)/1e6))
            step_time = (time.time_ns() - step_start) / 1e6
            total_time = (time.time_ns() - training_start) / 1e9
            print("Step: {}, Step time: {:.2f}, Total time: {:.2f}, Critic loss: {:.2f}, Encoder loss: {:.2f}, Actor loss: {:.2f}, Alpha loss: {:.2f}"
                  .format(i, step_time, total_time, critic_loss.item(), encoder_loss.item(), actor_loss.item(), alpha))


    def select_action(self, state):
        #print("Selecting action")
        embedding = self.encoder.embed(state[0])
        #print("Embedded")
        action = torch.FloatTensor(state[1].reshape(1, -1)).to(device)

        state_action = torch.cat([embedding, action], axis=1)

        return self.actor.select_action(state_action)

    def push_buffer(self, state):
        self.replay_buffer.push(state)
    