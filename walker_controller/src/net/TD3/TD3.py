import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameter
LINEAR_ACTOR = [256, 256]
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

        # Q1 architecture
        self.l1_1 = nn.Linear(state_dim + action_dim, LINEAR_CRITIC_Q1[0])
        self.l1_2 = nn.Linear(LINEAR_CRITIC_Q1[0], LINEAR_CRITIC_Q1[1])
        self.l1_3 = nn.Linear(LINEAR_CRITIC_Q1[1], 1)

        # Q2 architecture
        self.l2_1 = nn.Linear(state_dim + action_dim, LINEAR_CRITIC_Q2[0])
        self.l2_2 = nn.Linear(LINEAR_CRITIC_Q2[0], LINEAR_CRITIC_Q2[1])
        self.l2_3 = nn.Linear(LINEAR_CRITIC_Q2[1], 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1_1(sa))
        q1 = F.relu(self.l1_2(q1))
        q1 = self.l1_3(q1)

        q2 = F.relu(self.l2_1(sa))
        q2 = F.relu(self.l2_2(q2))
        q2 = self.l2_3(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1_1(sa))
        q1 = F.relu(self.l1_2(q1))
        q1 = self.l1_3(q1)

        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        discount = 0.99,
        policy_freq = 2,
        noise_clip = 0.5,
        policy_noise = 0.2,
        max_action = 1,
        expl_noise = 0.1
    ):

        self.tau = TAU
        self.learning_rate = LEARNING_RATE
        self.policy_freq = policy_freq
        self.discount = discount
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.max_action = max_action
        self.expl_noise = expl_noise
        self.action_dim = action_dim
        self.total_it = 0

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.learning_rate)
        self.actor_loss = 0

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.learning_rate)
        self.critic_loss = 0

        self.trainning = False
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if self.trainning == False:
            self.expl_noise = 0.2
        else:
            self.expl_noise = 0.1
        return self.actor(state).cpu().data.numpy().flatten() + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
    def predict_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
        
    def train(self, replay_buffer, batch_size = 64):
        self.trainning = True
        # sample
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise\
                ).clamp(-self.noise_clip, self.noise_clip)
                
            next_action = (self.actor_target(next_state) + noise\
                ).clamp(-self.max_action, self.max_action)

            # Compute target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q


        # Estimates current Q vallue 
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_loss = critic_loss.cpu().detach().numpy()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.total_it += 1
        # Delay policy update
        if self.total_it % self.policy_freq == 0:
            # Actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_loss = actor_loss.cpu().detach().numpy()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update parameters for target network
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
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