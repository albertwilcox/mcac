import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import mcac.utils.pytorch_utils as ptu


LOG_STD_MAX = 2
LOG_STD_MIN = -20
epsilon = 1e-6

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim[0], 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim[0])

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, d_obs, d_act, ensemble_size=2):
        super(Critic, self).__init__()

        self.ensemble = nn.ModuleList()

        for _ in range(ensemble_size):
            q = nn.Sequential(
                nn.Linear(d_obs[0] + d_act[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            self.ensemble.append(q)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        return tuple([q(sa) for q in self.ensemble])

    def variance(self, state, action):
        if len(state.shape) == 1:
            state = state[None]
            action = action[None]
        qf = self(state, action)
        variance = torch.var(torch.cat(qf), dim=0).squeeze()
        return variance

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        return self.ensemble[0](sa)


class GaussianPolicy(nn.Module):
    def __init__(self, d_obs, d_act, hidden_dim, max_action=1):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(d_obs[0], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, d_act[0])
        self.log_std_linear = nn.Linear(hidden_dim, d_act[0])

        self.apply(weights_init_)

        # action rescaling
        # TODO: correct this if we have to deal with action spaces that aren't centered around 0
        self.action_scale = torch.FloatTensor((max_action,))
        self.action_bias = torch.FloatTensor((0.,))

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # print('------------')
        # print(log_prob.mean())
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        # print(log_prob.mean())
        log_prob = log_prob.sum(1, keepdim=True)
        # print(log_prob.mean())
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, d_obs, d_act, hidden_dim, max_action=1):
        super(DeterministicPolicy, self).__init__()

        self.linear1 = nn.Linear(d_obs[0], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, d_act[0])
        self.noise = torch.Tensor(d_act[0])

        self.apply(weights_init_)

        # action rescaling
        # TODO: correct this if we have to deal with action spaces that aren't centered around 0
        self.action_scale = torch.FloatTensor(max_action)
        self.action_bias = torch.FloatTensor(0.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


# AWAC models


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation).to(ptu.TORCH_DEVICE)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim).to(ptu.TORCH_DEVICE)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim).to(ptu.TORCH_DEVICE)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def get_logprob(self, obs, actions):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(axis=1)

        return logp_pi


class AWACMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation).to(ptu.TORCH_DEVICE)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim).to(ptu.TORCH_DEVICE)

        self.log_std_logits = nn.Parameter(
            torch.zeros(act_dim, requires_grad=True)).to(ptu.TORCH_DEVICE)
        self.min_log_std = -6
        self.max_log_std = 0
        # self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        # print("Using the special policy")
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit

        log_std = torch.sigmoid(self.log_std_logits)

        log_std = self.min_log_std + log_std * (
                self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        # print("Std: {}".format(std))

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        return pi_action, logp_pi

    def get_logprob(self, obs, actions):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit
        log_std = torch.sigmoid(self.log_std_logits)
        # log_std = self.log_std_layer(net_out)
        log_std = self.min_log_std + log_std * (
                self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)

        return logp_pi


class MLPVFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1], activation).to(ptu.TORCH_DEVICE)

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)  # Critical to ensure q has right shape.


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation).to(ptu.TORCH_DEVICE)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, d_obs, d_act, max_action, hidden_sizes=(256, 256),
                 activation=nn.ReLU, special_policy=None):
        super().__init__()

        # build policy and value functions
        if special_policy is 'awac':
            self.pi = AWACMLPActor(d_obs, d_act, (256, 256, 256, 256), activation,
                                   max_action).to(ptu.TORCH_DEVICE)
        else:
            self.pi = SquashedGaussianMLPActor(d_obs, d_act, hidden_sizes, activation,
                                               max_action).to(device)
        self.q1 = MLPQFunction(d_obs, d_act, hidden_sizes, activation).to(ptu.TORCH_DEVICE)
        self.q2 = MLPQFunction(d_obs, d_act, hidden_sizes, activation).to(ptu.TORCH_DEVICE)
        self.v = MLPVFunction(d_obs, d_act, hidden_sizes, activation).to(ptu.TORCH_DEVICE)

    def act_batch(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().data.numpy().flatten()


# CQL Stuff

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class CQLActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(CQLActor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(ptu.TORCH_DEVICE)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob
        
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(ptu.TORCH_DEVICE)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


class CQLCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(CQLCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)