import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils.ou_noise import OUNoise


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]
# hyperparameter
LINEAR_ACTOR = [400, 300]
LINEAR_CRITIC_Q1 = [256, 256]
LINEAR_CRITIC_Q2 = [256, 256]
LEARNING_RATE = 1e-3
TAU = 0.001

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, action_space = None):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, LINEAR_ACTOR[0])
		self.l2 = nn.Linear(LINEAR_ACTOR[0], LINEAR_ACTOR[1])
		self.l3 = nn.Linear(LINEAR_ACTOR[1], action_dim)

		if action_space == None:
			self.action_scale = 1.
			self.action_bias = 0.
		else:
		    self.action_scale = torch.FloatTensor((action_space.high - action_space.low)/2.)
		    self.action_bias = torch.FloatTensor((action_space.high + action_space.low)/2.)
	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = torch.tanh(self.l3(a)) * self.action_scale + self.action_bias
		return a


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, LINEAR_ACTOR[0])
		self.l2 = nn.Linear(LINEAR_ACTOR[0] + action_dim, LINEAR_ACTOR[1])
		self.l3 = nn.Linear(LINEAR_ACTOR[1], 1)


	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		return self.l3(q)


class DDPG(object):
	def __init__(self, state_dim, action_dim, discount=0.99, tau=0.001):
		self.actor = Actor(state_dim, action_dim).to(device)
		self.learning_rate = LEARNING_RATE
		self.discount = discount
		self.tau = tau
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.learning_rate)
		self.actor_loss = 0

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.learning_rate, weight_decay=1e-2)
		self.critic_loss = 0

		self.noise_added = OUNoise(action_dim)
		


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten() + self.noise_added.noise()
	def predict_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=64):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)
		self.critic_loss = critic_loss.cpu().detach().numpy()


		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()
		self.actor_loss = actor_loss.cpu().detach().numpy()

		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save_model(self, filename, agent):
		save_dir = f"weights/{agent}/"
		torch.save(self.critic.state_dict(), save_dir + filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), save_dir+ filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), save_dir + filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), save_dir + filename + "_actor_optimizer")
		print("Saved!!!")
		
	def load_model(self, filename, agent):
		load_dir = f"weights/{agent}/"
		self.critic.load_state_dict(torch.load(load_dir + filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(load_dir + filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(load_dir + filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(load_dir + filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		print("load successfully")