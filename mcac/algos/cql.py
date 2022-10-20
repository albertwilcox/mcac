# Adapted from https://github.com/BY571/CQL/blob/main/CQL-SAC/networks.py main github for CQL in Pytorch

import mcac.utils.pytorch_utils as ptu
from mcac.algos.core import CQLActor, CQLCritic

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import math
import copy

device = ptu.TORCH_DEVICE


class CQL(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self, params):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(CQL, self).__init__()
        self.batch_size = params['batch_size']

        self.state_size = params['d_obs'][0]
        self.action_size = params['d_act'][0]

        self.do_mcac_bonus = params['do_mcac_bonus']

        self.device = device
        
        self.gamma = torch.FloatTensor([params['discount']]).to(device)
        self.tau = params['tau']
        hidden_size = params['hidden_size']
        learning_rate = params['lr']
        self.clip_grad_param = 1

        self.target_entropy = -self.action_size  

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
        
        # CQL params
        self.with_lagrange = params['with_lagrange']
        self.temp = params['temp']
        self.cql_weight = params['cql_weight']
        self.target_action_gap = params['target_action_gap']
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate) 
        
        # Actor Network 

        self.actor_local = CQLActor(self.state_size, self.action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = CQLCritic(self.state_size, self.action_size, hidden_size, 2).to(device)
        self.critic2 = CQLCritic(self.state_size, self.action_size, hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = CQLCritic(self.state_size, self.action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = CQLCritic(self.state_size, self.action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 

    def select_action(self, state, evaluate=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
            else:
                action = self.actor_local.get_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states, actions_pred.squeeze(0))   
        q2 = self.critic2(states, actions_pred.squeeze(0))
        min_Q = torch.min(q1,q2).cpu()
        actor_loss = ((alpha * log_pis.cpu() - min_Q )).mean()
        return actor_loss, log_pis

    def _compute_policy_values(self, obs_pi, obs_q):
        actions_pred, log_pis = self.actor_local.evaluate(obs_pi)
        
        qs1 = self.critic1(obs_q, actions_pred)
        qs2 = self.critic2(obs_q, actions_pred)
        
        return qs1 - log_pis.detach(), qs2 - log_pis.detach()
    
    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_size)
        return random_values - random_log_probs
    
    def update(self, replay_buffer):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        out_dict = replay_buffer.sample(self.batch_size)

        obs, act, next_obs, rew, mask, drtg = out_dict['obs'], out_dict['act'], \
                                              out_dict['next_obs'], out_dict['rew'], \
                                              out_dict['mask'], out_dict['drtg']

        states, actions, rewards, next_states, mask, drtg = ptu.torchify(obs, act, rew, next_obs, mask, drtg)
        # Here done is 1-mask

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, new_log_pi = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states, next_action)
            Q_target2_next = self.critic2_target(next_states, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * new_log_pi
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * mask * Q_target_next.squeeze(1)) 
            
            # Apply MCAC bonus
            if self.do_mcac_bonus:
                Q_targets = torch.max(Q_targets, drtg)

        Q_targets = Q_targets.unsqueeze(1)
        # Compute critic loss
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, Q_targets)
        critic2_loss = F.mse_loss(q2, Q_targets)
        
        # CQL addon
        random_actions = torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self.device)
        num_repeat = int (random_actions.shape[0] / states.shape[0])
        temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1).view(states.shape[0] * num_repeat, states.shape[1])
        temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1).view(next_states.shape[0] * num_repeat, next_states.shape[1])
        
        current_pi_values1, current_pi_values2  = self._compute_policy_values(temp_states, temp_states)
        next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)
        
        random_values1 = self._compute_random_values(temp_states, random_actions, self.critic1).reshape(states.shape[0], num_repeat, 1)
        random_values2 = self._compute_random_values(temp_states, random_actions, self.critic2).reshape(states.shape[0], num_repeat, 1)
        
        current_pi_values1 = current_pi_values1.reshape(states.shape[0], num_repeat, 1)
        current_pi_values2 = current_pi_values2.reshape(states.shape[0], num_repeat, 1)

        next_pi_values1 = next_pi_values1.reshape(states.shape[0], num_repeat, 1)
        next_pi_values2 = next_pi_values2.reshape(states.shape[0], num_repeat, 1)
        
        cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)
        
        assert cat_q1.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
        assert cat_q2.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"

        cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q1.mean()) * self.cql_weight
        cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q2.mean()) * self.cql_weight
        
        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5 
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        
        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss
        
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        info = {
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'cql1_scaled_loss': cql1_scaled_loss.item(),
            'cql2_scaled_loss': cql2_scaled_loss.item(),
            'current_alpha': current_alpha,
            'cql_alpha_loss': cql_alpha_loss.item(),
            'cql_alpha': cql_alpha.item()
        }
        return info

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)

        torch.save(self.actor_local.state_dict(), os.path.join(folder, "actor.pth"))
        torch.save(self.alpha, os.path.join(folder, "alpha.pth"))
        torch.save(self.cql_log_alpha, os.path.join(folder, "cql_log_alpha.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(folder, "critic1.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(folder, "critic2.pth"))
        
        torch.save(self.actor_optimizer.state_dict(), os.path.join(folder, "actor_optimizer.pth"))
        torch.save(self.alpha_optimizer.state_dict(), os.path.join(folder, "alpha_optimizer.pth"))
        torch.save(self.cql_alpha_optimizer.state_dict(), os.path.join(folder, "cql_alpha_optimizer.pth"))
        torch.save(self.critic1_optimizer.state_dict(), os.path.join(folder, "critic1_optimizer.pth"))
        torch.save(self.critic2_optimizer.state_dict(), os.path.join(folder, "critic2_optimizer.pth"))

    def load(self, folder):
        self.actor_local.load_state_dict(
            torch.load(os.path.join(folder, "models/actor.pth"), map_location=ptu.TORCH_DEVICE)).to(ptu.TORCH_DEVICE)
        self.critic1.load_state_dict(
            torch.load(os.path.join(folder, "models/critic1.pth"), map_location=ptu.TORCH_DEVICE)).to(ptu.TORCH_DEVICE)
        self.critic2.load_state_dict(
            torch.load(os.path.join(folder, "models/critic2.pth"), map_location=ptu.TORCH_DEVICE)).to(ptu.TORCH_DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict()).to(ptu.TORCH_DEVICE)
        self.critic2_target.load_state_dict(self.critic2.state_dict()).to(ptu.TORCH_DEVICE)

        self.alpha = torch.load(os.path.join(folder, "models/alpha.pth"), map_location=ptu.TORCH_DEVICE).to(ptu.TORCH_DEVICE)
        self.cql_log_alpha = torch.load(os.path.join(folder, "models/cql_log_alpha.pth"), map_location=ptu.TORCH_DEVICE).to(ptu.TORCH_DEVICE)

        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(folder, "models/actor_optimizer.pth"), map_location=ptu.TORCH_DEVICE)).to(ptu.TORCH_DEVICE)
        self.alpha_optimizer.load_state_dict(
            torch.load(os.path.join(folder, "models/alpha_optimizer.pth"), map_location=ptu.TORCH_DEVICE)).to(ptu.TORCH_DEVICE)
        self.cql_alpha_optimizer.load_state_dict(
            torch.load(os.path.join(folder, "models/cql_alpha_optimizer.pth"), map_location=ptu.TORCH_DEVICE)).to(ptu.TORCH_DEVICE)
        self.critic1_optimizer.load_state_dict(
            torch.load(os.path.join(folder, "models/critic1_optimizer.pth"), map_location=ptu.TORCH_DEVICE)).to(ptu.TORCH_DEVICE)
        self.critic2_optimizer.load_state_dict(
            torch.load(os.path.join(folder, "models/critic2_optimizer.pth"), map_location=ptu.TORCH_DEVICE)).to(ptu.TORCH_DEVICE)
        