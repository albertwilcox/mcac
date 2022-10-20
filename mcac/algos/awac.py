"""
AWAC implementation based on https://github.com/hari-sikchi/AWAC
"""

import mcac.algos.core as core
import mcac.utils.pytorch_utils as ptu
import mcac.utils as utils

import torch
import torch.nn.functional as F

import itertools
from torch.optim import Adam
import os

import numpy as np

device = ptu.TORCH_DEVICE


class AWAC:

    def __init__(self, params):
        self.obs_dim = params['d_obs'][0]
        self.act_dim = params['d_act'][0]
        self.act_limit = params['max_action']
        self.discount = params['discount']
        self.p_lr = params['p_lr']
        self.lr = params['lr']
        self.alpha = params['alpha']
        self.beta = params['beta']
        # # Algorithm specific hyperparams

        self.batch_size = params['batch_size']
        self.critic_batch_size = params['critic_batch_size']
        self.polyak = params['polyak']

        self.do_mcac_bonus = params['do_mcac_bonus']

        # Create actor-critic module and target networks
        self.ac = core.MLPActorCritic(self.obs_dim, self.act_dim, self.act_limit,
                                      special_policy='awac')
        self.ac_targ = core.MLPActorCritic(self.obs_dim, self.act_dim, self.act_limit,
                                           special_policy='awac')
        self.ac_targ.load_state_dict(self.ac.state_dict())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.p_lr, weight_decay=1e-4)
        self.q_optimizer = Adam(itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters()),
                                lr=self.lr)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, obs, act, rew, next_obs, mask, drtg):

        q1 = self.ac.q1(obs, act)
        q2 = self.ac.q2(obs, act)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_act, logp_next_act = self.ac.pi(next_obs)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(next_obs, next_act)
            q2_pi_targ = self.ac_targ.q2(next_obs, next_act)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = rew + mask * self.discount * (q_pi_targ - self.alpha * logp_next_act)

            # Apply MCAC bonus
            if self.do_mcac_bonus:
                backup = torch.max(backup, drtg)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy().mean(),
                      Q2Vals=q2.detach().cpu().numpy().mean())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, obs, act):
        pi, logp_pi = self.ac.pi(obs)
        q1_pi = self.ac.q1(obs, pi)
        q2_pi = self.ac.q2(obs, pi)
        v_pi = torch.min(q1_pi, q2_pi)

        q1_old_actions = self.ac.q1(obs, act)
        q2_old_actions = self.ac.q2(obs, act)
        q_old_actions = torch.min(q1_old_actions, q2_old_actions)

        adv_pi = q_old_actions - v_pi
        weights = F.softmax(adv_pi / self.beta, dim=0)
        policy_logpp = self.ac.pi.get_logprob(obs, act)
        loss_pi = (-policy_logpp * len(weights) * weights.detach()).mean()

        # Useful info for logging
        pi_info = dict(LogPi=policy_logpp.detach().cpu().numpy())

        return loss_pi, pi_info

    def update(self, replay_buffer):
        out_dict = replay_buffer.sample(self.batch_size)

        obs, act, next_obs, rew, mask, drtg = out_dict['obs'], out_dict['act'], \
                                              out_dict['next_obs'], out_dict['rew'], \
                                              out_dict['mask'], out_dict['drtg']

        obs, act, rew, next_obs, mask, drtg = ptu.torchify(obs, act, rew, next_obs, mask, drtg)

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(obs, act, rew, next_obs, mask, drtg)
        loss_q.backward()
        self.q_optimizer.step()

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(obs, act)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        info = {
            'q_loss': loss_q.item(),
            'policy_loss': loss_pi.item(),
        }
        info = utils.add_dicts(info, q_info)
        return info

    def select_action(self, obs, evaluate=False):
        act = self.ac.act(ptu.torchify(obs[None]), evaluate)

        act = np.clip(act, -self.act_limit, self.act_limit)

        return act

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)

        torch.save(self.ac.state_dict(), os.path.join(folder, "ac.pth"))
        torch.save(self.pi_optimizer.state_dict(), os.path.join(folder, "pi_optimizer.pth"))
        torch.save(self.q_optimizer.state_dict(), os.path.join(folder, "q_optimizer.pth"))

    def load(self, folder):
        self.ac.load_state_dict(
            torch.load(os.path.join(folder, "ac.pth"), map_location=ptu.TORCH_DEVICE))
        self.ac_targ.load_state_dict(self.ac.state_dict())
        self.pi_optimizer.load_state_dict(
            torch.load(os.path.join(folder, "pi_optimizer.pth"), map_location=ptu.TORCH_DEVICE))
        self.q_optimizer.load_state_dict(
            torch.load(os.path.join(folder, "q_optimizer.pth"), map_location=ptu.TORCH_DEVICE))
