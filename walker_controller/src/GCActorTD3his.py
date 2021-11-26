import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import copy
from gcn import GCN
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2020)
np.random.seed(2020)
# action limit
# hipR, kneeR, ankleR, hipL, kneeL, ankleL
HIGH = np.array([1.0,     1.0,     1.0,      1.0,      1.0,   1.0])
LOW = np.array([-1.0,    -1.0,    -1.0,     -1.0,     -1.0,  -1.0])

# HIGH = np.array([10.0,     10.0,     0.2,      10.0,      10.0,   0.2])
# LOW = np.array([-10.0,    -10.0,    -1.0,     -10.0,     -10.0,  -1.0])

# hipR, hipL, kneeR, kneeL, ankleR, ankleL
# HIGH = np.array([0.9,     0.9,      0.4,      0.4,      0.15,   0.15])
# LOW = np.array([-0.8,    -0.8,     -0.8,     -0.8,     -0.35,  -0.35])

# hyperparameter
LINEAR_ACTOR = [144, 144]
LINEAR_CRITIC_Q1 = [256, 256]
LINEAR_CRITIC_Q2 = [256, 256]
LEARNING_RATE = 8e-5
TAU = 0.001

HIDDEN_DIM_GCN_ACTOR = 128
DROPOUT = 0.


class ActionSpace:
    def __init__(self, high=HIGH, low =LOW):
        self.high = high
        self.low = low
        self.maxaction = np.max(np.array([self.high,  -self.low]),axis=0)
        self.maxactionTensor = torch.FloatTensor(self.maxaction)
class Actor(nn.Module):
    def __init__(self, num_joint, feature_dim, action_dim, action_lim = None):
        super(Actor, self).__init__()

        self.gcn = GCN(num_joint, feature_dim, hidden_dim = HIDDEN_DIM_GCN_ACTOR, dropout = DROPOUT)
        state_dim = HIDDEN_DIM_GCN_ACTOR#self.gcn.cal_dim(num_joint, feature_dim)//8*5
        self.l1_1 = nn.Linear(state_dim, LINEAR_ACTOR[0])
        self.l1_2 = nn.Linear(LINEAR_ACTOR[0], LINEAR_ACTOR[0])
        self.l1_3 = nn.Linear(LINEAR_ACTOR[0], action_dim//2)

        self.l2_1 = nn.Linear(state_dim, LINEAR_ACTOR[1])
        self.l2_2 = nn.Linear(LINEAR_ACTOR[1], LINEAR_ACTOR[1])
        self.l2_3 = nn.Linear(LINEAR_ACTOR[1], action_dim//2)

        #self.lout = nn.Linear(state_dim, action_dim)

        if action_lim == None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor((action_lim.high - action_lim.low)/2.)
            self.action_bias = torch.FloatTensor((action_lim.high + action_lim.low)/2.)

    def forward(self, x, adj, training = False):
        outr, outl, _ = self.gcn(x, adj, training)
        # outr = F.relu(outr)
        # outl = F.relu(outl)
        #a = F.tanh(self.lout(a))
        
        al = F.relu(self.l1_1(outl))
        al = F.relu(self.l1_2(al))
        #al = F.relu(self.l1_3(al))
        al = (torch.tanh(self.l1_3(al)) * self.action_scale[3:] + self.action_bias[3:])*(-1)

        ar = F.relu(self.l2_1(outr))
        ar = F.relu(self.l2_2(ar))
        ar = torch.tanh(self.l2_3(ar)) * self.action_scale[:3] + self.action_bias[:3]
        #a = torch.tanh(self.l3(a)) * self.action_scale + self.action_bias
        return torch.cat([ar, al], 1)
        #return a

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


class GCActorTD3his(object):
    def __init__(
        self,
        num_joint,
        feature_dim,
        action_dim,
        adj_matrix,
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
        #self.max_action = max_action
        self.expl_noise = expl_noise
        self.action_dim = action_dim
        self.total_it = 0

        self.adj_matrix = adj_matrix
        self.action_lim = ActionSpace()
        self.max_action = torch.FloatTensor([self.action_lim.high])
        self.min_action = torch.FloatTensor([self.action_lim.low])
        self.noise_clip_min = torch.FloatTensor(self.action_lim.high/-2.0)
        self.noise_clip_max = torch.FloatTensor(self.action_lim.high/2.0)

        self.actor = Actor(num_joint, feature_dim, action_dim, self.action_lim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.learning_rate)
        self.actor_loss = 0

        state_dim = num_joint*feature_dim
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.learning_rate)
        self.critic_loss = 0

        self.training = False
        self.count = 0
        self.act = np.zeros(6)
        
    def select_action(self, state):
        state = torch.FloatTensor([state]).to(device)
        # if self.count > 15 or np.sum(self.act) == 0:
        #     self.count = 0
        #     for i in range(len(self.act)):
        #         self.act[i] = np.random.choice([self.action_lim.high[i], self.action_lim.low[i]])
            # self.act[2] = np.random.choice([self.action_lim.high[2], 0, self.action_lim.low[2]])
            # self.act[5] = np.random.choice([self.action_lim.high[5], 0, self.action_lim.low[5]])
        if self.training == False:
            self.expl_noise = 0.4
            if np.random.random_sample() > 0.8:
                for i in range(len(self.act)):
                    self.act[i] = np.random.choice([self.action_lim.high[i], self.action_lim.low[i]])
                scale = self.action_lim.maxaction*self.expl_noise
                return self.act + np.random.normal(0, scale, size=self.action_dim)                
        else:
            self.expl_noise = 0.1
        scale = self.action_lim.maxaction*self.expl_noise
        #scale = 1.0 * self.expl_noise
        action = self.actor(state, self.adj_matrix).cpu().data.numpy().flatten() + np.random.normal(0, scale, size=self.action_dim)
        return action
    def select_actionv2(self, state):
        state = torch.FloatTensor([state]).to(device)
        action = self.actor(state, self.adj_matrix).cpu().data.numpy().flatten()
        if self.training == False:
            if np.random.random_sample() > 0.6:
                return utils.modify_action(action)
            self.expl_noise = 0.3
        else:
            self.expl_noise = 0.2
        scale = np.max(np.array([self.action_lim.high,  -self.action_lim.low]),axis=0)*self.expl_noise
        #scale = 1.0 * self.expl_noise
        action =  action +  np.random.normal(0, scale, size=self.action_dim)
        return action
    def predict_action(self, state):
        state = torch.FloatTensor([state]).to(device)
        return self.actor(state, self.adj_matrix).cpu().data.numpy().flatten()
        
    def train(self, replay_buffer, batch_size = 64):
        self.training = True
        # sample
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        vec_state = state.view(state.size()[0], -1)
        vec_next_state = next_state.view(next_state.size()[0], -1)
        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise * self.action_lim.maxactionTensor
            noise = torch.max(torch.min(noise, self.noise_clip_max), self.noise_clip_min)
            next_action = self.actor_target(next_state, self.adj_matrix, training = self.training) + noise
                #.clamp(-self.max_action, self.max_action)
            # clip next_action between min_action and max_action
            next_action = torch.max(torch.min(next_action, self.max_action), self.min_action)
            # Compute target Q value
            target_Q1, target_Q2 = self.critic_target(vec_next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q


        # Estimates current Q vallue 
        current_Q1, current_Q2 = self.critic(vec_state, action)
        
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
            actor_loss = -self.critic.Q1(vec_state, self.actor(state, self.adj_matrix, training = self.training)).mean()
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

    def save_model(self, filename):
        torch.save(self.critic.state_dict(), "weights/" + filename + "_critic.pt")
        torch.save(self.critic_optimizer.state_dict(), "weights/" + filename + "_critic_optimizer.pt")
        
        torch.save(self.actor.state_dict(), "weights/" + filename + "_actor.pt")
        torch.save(self.actor_optimizer.state_dict(), "weights/" + filename + "_actor_optimizer.pt")
        print("Saved!!!")
        
    def load_model(self, filename):
        self.critic.load_state_dict(torch.load("weights/" + filename + "_critic.pt"))
        self.critic_optimizer.load_state_dict(torch.load("weights/" + filename + "_critic_optimizer.pt"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load("weights/" + filename + "_actor.pt"))
        self.actor_optimizer.load_state_dict(torch.load("weights/" + filename + "_actor_optimizer.pt"))
        self.actor_target = copy.deepcopy(self.actor)
        print("load successfully")


if __name__ == "__main__":
    adj = np.array([[0, 1, 1, 0],[1,0,1,0],[1,1,0,1],[0,1,0,0]])

    adj_hat = torch.FloatTensor(adj + np.eye(adj.shape[0], dtype=np.float64))
    input_ = torch.FloatTensor(np.array([[[1.2,0.2],[-0.1,0.2],[0.5, 0.1],[0.2,0.2]]]))


    actor = Actor(4, 2, 4)
    print(actor(input_, adj_hat))
    action_l = ActionSpace()
    scale = torch.FloatTensor((action_l.high - action_l.low)/2.)
    bias = torch.FloatTensor((action_l.high + action_l.low)/2.)
    print(scale[:3], bias[3:])