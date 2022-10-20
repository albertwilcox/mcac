"""
TD3 implementation based on https://github.com/sfujim/TD3
"""

import mcac.algos.core as core
import mcac.utils.pytorch_utils as ptu

import torch
import torch.nn.functional as F

import copy
import os


class TD3:
    def __init__(self, params):

        self.max_action = params['max_action']
        self.discount = params['discount']
        self.tau = params['tau']
        self.policy_noise = params['policy_noise']
        self.noise_clip = params['noise_clip']
        self.policy_freq = params['policy_freq']
        self.batch_size = params['batch_size']
        self.batch_size_demonstrator = params['batch_size_demonstrator']
        self.do_bc_loss = params['do_bc_loss']
        self.bc_weight = params['bc_weight']
        self.bc_decay = params['bc_decay']
        self.do_q_filter = params['do_q_filter']
        self.do_mcac_bonus = params['do_mcac_bonus']

        self.total_it = 0
        self.running_risk = 1

        self.actor = core.Actor(params['d_obs'], params['d_act'], self.max_action).to(ptu.TORCH_DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params['lr_actor'])

        self.critic = core.Critic(params['d_obs'], params['d_act']).to(ptu.TORCH_DEVICE)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params['lr_critic'])

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(ptu.TORCH_DEVICE)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer):
        # Sample from replay buffer
        out_dict = replay_buffer.sample(self.batch_size)
        obs, action, next_obs, reward, mask, drtg, expert = out_dict['obs'], out_dict['act'], \
                                                            out_dict['next_obs'], out_dict['rew'], \
                                                            out_dict['mask'], out_dict['drtg'], \
                                                            out_dict['expert']
        obs, action, next_obs, reward, mask, drtg, expert = \
            ptu.torchify(obs, action, next_obs, reward, mask, drtg, expert)

        info = {}
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise)\
                .clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_obs) + noise)\
                .clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2).squeeze()
            target_Q = reward + mask * self.discount * target_Q
            target_Q = target_Q.squeeze()

            # Apply MCAC bonus
            if self.do_mcac_bonus:
                target_Q = torch.max(target_Q, drtg)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        current_Q1, current_Q2 = current_Q1.squeeze(), current_Q2.squeeze()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        info['critic_loss'] = critic_loss.item()
        info['Q1'] = current_Q1.mean().item()
        info['Q2'] = current_Q2.mean().item()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_q_loss = -self.critic.Q1(obs, self.actor(obs)).mean()

            # Behavior cloning auxiliary loss, inspired by DDPGfD paper
            if self.do_bc_loss:
                # Sample expert actions from the replay buffer
                out_dict = replay_buffer.sample_positive(self.batch_size_demonstrator, 'expert')
                obs, action = out_dict['obs'], out_dict['act']
                obs, action = ptu.torchify(obs, action)

                # Calculate loss as negative log prob of actions
                act_hat = self.actor(obs)
                losses = F.mse_loss(act_hat, action, reduction='none')

                # Optional Q filter
                if self.do_q_filter:
                    with torch.no_grad():
                        q_agent = self.critic.Q1(obs, act_hat)
                        q_expert = self.critic.Q1(obs, action)
                        q_filter = torch.gt(q_expert, q_agent).float()

                    if torch.sum(q_filter) > 0:
                        bc_loss = torch.sum(losses * q_filter) / torch.sum(q_filter)
                    else:
                        bc_loss = actor_q_loss * 0
                else:
                    bc_loss = torch.mean(losses)

            else:
                bc_loss = actor_q_loss * 0

            lambda_bc = self.bc_decay ** self.total_it * self.bc_weight
            actor_loss = actor_q_loss + lambda_bc * bc_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            info['actor_loss'] = actor_loss.item()
            info['actor_q_loss'] = actor_q_loss.item()
            info['actor_bc_loss'] = bc_loss.item()

            # Update the frozen target models
            ptu.soft_update(self.critic, self.critic_target, 1 - self.tau)
            ptu.soft_update(self.actor, self.actor_target, 1 - self.tau)

        self.total_it += 1
        return info

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)

        torch.save(self.critic.state_dict(), os.path.join(folder, "critic.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(folder, "critic_optimizer.pth"))

        torch.save(self.actor.state_dict(), os.path.join(folder, "actor.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(folder, "actor_optimizer.pth"))

    def load(self, folder):
        self.critic.load_state_dict(torch.load(os.path.join(folder, "critic.pth")))
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(folder, "critic_optimizer.pth")))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(folder, "actor.pth")))
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(folder, "actor_optimizer.pth")))

