"""
SAC Implementation based on https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
"""

import mcac.algos.core as core
import mcac.utils.pytorch_utils as ptu

import torch
import torch.nn.functional as F
from torch.optim import Adam

import copy
import os


class SAC:
    def __init__(self, params):

        self.tau = params['tau']
        self.alpha = params['alpha']
        self.max_action = params['max_action']
        self.discount = params['discount']
        self.batch_size = params['batch_size']
        self.do_mcac_bonus = params['do_mcac_bonus']
        self.total_it = 0
        self.running_risk = 1

        self.policy_type = params['policy']
        self.target_update_interval = params['target_update_interval']
        self.automatic_entropy_tuning = params['automatic_entropy_tuning']

        self.critic = core.Critic(params['d_obs'], params['d_act'],
                                  ensemble_size=params['q_ensemble_size']).to(ptu.TORCH_DEVICE)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=params['lr'])

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(params['d_act'])
                                                  .to(ptu.TORCH_DEVICE)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=ptu.TORCH_DEVICE)
                self.alpha_optim = Adam([self.log_alpha], lr=params['lr'])

            self.policy = core.GaussianPolicy(params['d_obs'], params['d_act'],
                                              params['hidden_size'], params['max_action']) \
                .to(ptu.TORCH_DEVICE)
            self.policy_optim = Adam(self.policy.parameters(), lr=params['lr'])

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = core.DeterministicPolicy(params['d_obs'], params['d_act'],
                                                   params['hidden_size'], params['max_action']) \
                .to(ptu.TORCH_DEVICE)
            self.policy_optim = Adam(self.policy.parameters(), lr=params['lr'])

    def select_action(self, state, evaluate=False):
        state = ptu.torchify(state).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0] * self.max_action

    def select_action_batch(self, states, evaluate=False):
        states = ptu.torchify(states)
        if evaluate is False:
            action, _, _ = self.policy.sample(states)
        else:
            _, _, action = self.policy.sample(states)
        return action.detach() * self.max_action

    def update(self, replay_buffer):
        # Sample from replay buffer
        out_dict = replay_buffer.sample(self.batch_size)
        obs, action, next_obs, reward, \
        mask, drtg, expert, succ = out_dict['obs'], out_dict['act'], \
                                    out_dict['next_obs'], out_dict['rew'], \
                                    out_dict['mask'], out_dict['drtg'], \
                                    out_dict['expert'], out_dict['succ']

        obs, action, next_obs, reward, mask, drtg, expert, succ = \
            ptu.torchify(obs, action, next_obs, reward, mask, drtg, expert, succ)

        # Compute targets using bellman backup and target function
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_obs)
            qf_list_next_target = self.critic_target(next_obs, next_state_action)
            min_qf_next_target = torch.min(torch.cat(qf_list_next_target, dim=1), dim=1)[0] \
                                    - self.alpha * next_state_log_pi.squeeze()
            min_qf_next_target = min_qf_next_target.squeeze()
            next_q_value = reward + mask * self.discount * min_qf_next_target

            # Apply MCAC bonus
            if self.do_mcac_bonus:
                next_q_value = torch.max(next_q_value, drtg)

        # Compute Q losses
        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf_list = self.critic(obs, action)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_losses = [
            F.mse_loss(qf.squeeze(), next_q_value)
            for qf in qf_list
        ]
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = sum(qf_losses)

        # Q function backward pass
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Sample from policy, compute minimum Q value of sampled action
        pi, log_pi, _ = self.policy.sample(obs)
        qf_list_pi = self.critic(obs, pi)
        min_qf_pi = torch.min(torch.cat(qf_list_pi, dim=1), dim=1)[0]

        # Calculate policy loss
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Policy backward pass
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(ptu.TORCH_DEVICE)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if self.total_it % self.target_update_interval == 0:
            ptu.soft_update(self.critic, self.critic_target, 1 - self.tau)

        info = {
            'policy_loss': policy_loss.item(),
            'alphpa_loss': alpha_loss.item(),
            'alpha_tlogs': alpha_tlogs.item()
        }
        for i, (qf, qf_loss) in enumerate(zip(qf_list, qf_losses)):
            if i > 3:
                break  # don't log absurd number of Q functions
            info['Q%d' % (i + 1)] = qf.mean().item()
            info['Q%d_loss' % (i + 1)] = qf_loss.item()

        self.total_it += 1
        return info

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)

        torch.save(self.critic.state_dict(), os.path.join(folder, "critic.pth"))
        torch.save(self.critic_optim.state_dict(), os.path.join(folder, "critic_optimizer.pth"))

        torch.save(self.policy.state_dict(), os.path.join(folder, "actor.pth"))
        torch.save(self.policy_optim.state_dict(), os.path.join(folder, "actor_optimizer.pth"))

    def load(self, folder):
        self.critic.load_state_dict(
            torch.load(os.path.join(folder, "critic.pth"), map_location=ptu.TORCH_DEVICE))
        self.critic_optim.load_state_dict(
            torch.load(os.path.join(folder, "critic_optimizer.pth"), map_location=ptu.TORCH_DEVICE))
        self.critic_target = copy.deepcopy(self.critic)

        self.policy.load_state_dict(
            torch.load(os.path.join(folder, "actor.pth"), map_location=ptu.TORCH_DEVICE))
        self.policy_optim.load_state_dict(
            torch.load(os.path.join(folder, "actor_optimizer.pth"), map_location=ptu.TORCH_DEVICE))
