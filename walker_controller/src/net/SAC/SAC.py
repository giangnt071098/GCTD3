#based on https://github.com/pranz24/pytorch-soft-actor-critic/
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# hyperparameter
LINEAR_ACTOR = [256, 256]
LINEAR_CRITIC_Q1 = [256, 256]
LINEAR_CRITIC_Q2 = [256, 256]
LEARNING_RATE = 1e-3

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
# Deterministic policy
class Actor(nn.Module):
    def __init__(self,state_dim, action_dim, action_space = None):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, LINEAR_ACTOR[0])
        self.l2 = nn.Linear(LINEAR_ACTOR[0], LINEAR_ACTOR[1])
        self.l3 = nn.Linear(LINEAR_ACTOR[1], action_dim)

        self.log_std_linear = nn.Linear(LINEAR_ACTOR[0], action_dim)

        self.noise = torch.Tensor(action_dim)
        if action_space == None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low)/2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low)/2.)
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.l3(a)
        log_std = self.log_std_linear(a)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        #Q1 architecture
        self.l1_1 = nn.Linear(state_dim + action_dim, LINEAR_CRITIC_Q1[0])
        self.l1_2 = nn.Linear(LINEAR_CRITIC_Q1[0], LINEAR_CRITIC_Q1[1])
        self.l1_3 = nn.Linear(LINEAR_CRITIC_Q1[1], 1)

        #Q2 architecture
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


class SAC(object):
    def __init__(self, 
                state_dim, 
                action_dim,
                discount = 0.99,
                tau = 0.005,
                alpha = 0.2,
                target_update_interval = 1,
                updates = 0,
                automatic_entropy_tuning = True):
        
        self.discount = discount
        self.tau = tau
        self.learning_rate = LEARNING_RATE
        self.alpha = alpha
        self.updates = updates
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.learning_rate)
        self.actor_loss = 0

        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor((action_dim,)).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device = device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.learning_rate)
        self.critic_loss = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]
    def predict_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]
    def train(self, replay_buffer, batch_size = 64):
        # sample
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward + not_done * self.discount * (min_qf_next_target)
        qf1, qf2 = self.critic(state, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        self.critic_loss = qf_loss.cpu().detach().numpy()

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state)

        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.actor_loss = actor_loss.cpu().detach().numpy()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.updates % self.target_update_interval == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

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