'''
Built on cartrgipper implementation in
https://github.com/SudeepDasari/visual_foresight
'''

import numpy as np
import copy
from mcac.envs.base_mujoco_env import BaseMujocoEnv
from gym.spaces import Box
import os
"""
Constants associated with the Object Extraction env.
"""

# FIXED_ENV = False
GT_STATE = True
EARLY_TERMINATION = True


def no_rot_dynamics(prev_target_qpos, action):
    target_qpos = np.zeros_like(prev_target_qpos)
    target_qpos[:3] = action[:3] + prev_target_qpos[:3]
    target_qpos[4] = action[3]
    return target_qpos


def clip_target_qpos(target, lb, ub):
    target[:len(lb)] = np.clip(target[:len(lb)], lb, ub)
    return target


class ObjExtraction(BaseMujocoEnv):
    def __init__(self, fixed=True):
        parent_params = super()._default_hparams()
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        self.reset_xml = os.path.join(envs_folder, 'assets', 'extraction.xml')
        super().__init__(self.reset_xml, parent_params)
        self._adim = 4
        self.substeps = 500
        self.low_bound = np.array([-0.4, -0.4, -0.05])
        self.high_bound = np.array([0.4, 0.4, 0.15])
        self.action_multiplier = np.array([0.02, 0.02, 0.02, 0.1])
        self.ac_high = np.ones_like(self.action_multiplier)
        self.ac_low = -self.ac_high
        self.action_space = Box(self.ac_low, self.ac_high)
        self._previous_target_qpos = None
        self.target_height_thresh = 0.03
        self.object_fall_thresh = -0.03
        self.obj_y_dist_range = np.array([0.05, 0.05])
        self.obj_x_range = np.array([-0.2, -0.05])
        self.randomize_objects = not fixed
        self.gt_state = GT_STATE

        if self.gt_state:
            self.observation_space = Box(low=-np.inf,
                                         high=np.inf,
                                         shape=(27, ))
        else:
            self.observation_space = (48, 64, 3)
        self.reset()

    def render(self):
        # cameras are flipped in height dimension
        return super().render()[:, ::-1].copy().squeeze()

    def reset(self):
        self._reset_sim(self.reset_xml)

        #clear our observations from last rollout
        self._last_obs = None

        state = self.sim.get_state()
        pos = np.copy(state.qpos[:])
        pos[6:] = self.object_reset_poses().ravel()
        state.qpos[:] = pos
        self.sim.set_state(state)

        self.sim.forward()

        self._previous_target_qpos = copy.deepcopy(
            self.sim.data.qpos[:5].squeeze())
        self._previous_target_qpos[-1] = self.low_bound[-1]

        if self.gt_state:
            return pos
        else:
            return self.render()

    def step(self, action):
        action = np.clip(action, self.ac_low, self.ac_high) * self.action_multiplier
        target_qpos = self._next_qpos(action)
        if self._previous_target_qpos is None:
            self._previous_target_qpos = target_qpos

        for st in range(self.substeps):
            alpha = st / (float(self.substeps) - 1)
            self.sim.data.ctrl[:] = alpha * target_qpos + (
                1. - alpha) * self._previous_target_qpos
            self.sim.step()

        self._previous_target_qpos = target_qpos
        reward = self.reward_fn()
        goal = self.in_goal()

        if EARLY_TERMINATION:
            done = goal
        else:
            done = False

        info = {
            "goal": goal
        }

        if self.gt_state:
            return self.position, reward, done, info
        else:
            return self.render(), reward, done, info

    @property
    def jaw_width(self):
        pos = self.position
        return 0.08 - (pos[4] - pos[5])

    def set_y_range(self, bounds):
        self.obj_y_dist_range[0] = bounds[0]
        self.obj_y_dist_range[1] = bounds[1]

    def expert_action(self, noise_std=0.0, demo_quality='high'):
        cur_pos = self.position[:3]
        cur_pos[1] += 0.05  # compensate for length of jaws
        target_obj_pos = self.object_poses[1][:3]
        action = np.zeros(self._adim)
        delta = target_obj_pos - cur_pos

        if demo_quality == 'high':
            thresh1 = 0.02
            thresh2 = 0.04
        else:
            thresh1 = 0.03
            thresh2 = 0.05

        if np.abs(delta[0]) > thresh1:
            action[0] = delta[0]
            action[3] = 0.02
        elif np.abs(delta[1]) > thresh2:
            action[1] = delta[1]
            action[3] = 0.02
        elif self.jaw_width > 0.06:
            action[3] = 0.06
        else:
            action[3] = 0.06
            action[2] = 0.05
        action = action + np.random.randn(self._adim) * noise_std
        action = action / self.action_multiplier
        action = np.clip(action, self.ac_low, self.ac_high)
        return action

    def in_goal(self):
        return bool(self.target_object_height >= self.target_height_thresh)

    def reward_fn(self):
        return int(self.in_goal() - 1)

    def object_reset_poses(self):
        new_poses = np.zeros((3, 7))
        new_poses[:, 3] = 1
        if self.randomize_objects:
            x = np.random.uniform(self.obj_x_range[0], self.obj_x_range[1])
            y1 = np.random.randn() * 0.05
            y0 = y1 - np.random.uniform(self.obj_y_dist_range[0],
                                        self.obj_y_dist_range[1])
            y2 = y1 + np.random.uniform(self.obj_y_dist_range[0],
                                        self.obj_y_dist_range[1])
            new_poses[0, 0:2] = np.array([y0, x])
            new_poses[1, 0:2] = np.array([y1, x])
            new_poses[2, 0:2] = np.array([y2, x])
        else:
            x = np.mean(self.obj_x_range)
            y1 = 0.
            y0 = y1 - np.mean(self.obj_y_dist_range)
            y2 = y1 + np.mean(self.obj_y_dist_range)
            new_poses[0, 0:2] = np.array([y0, x])
            new_poses[1, 0:2] = np.array([y1, x])
            new_poses[2, 0:2] = np.array([y2, x])
        return new_poses

    @property
    def position(self):
        return np.copy(self.sim.get_state().qpos[:])

    @property
    def object_poses(self):
        pos = self.position
        num_objs = (self.position.shape[0] - 6) // 7
        poses = []
        for i in range(num_objs):
            poses.append(np.copy(pos[i * 7 + 6:(i + 1) * 7 + 6]))
        return np.array(poses)

    @property
    def target_object_height(self):
        return self.object_poses[1, 2] - 0.072

    def _next_qpos(self, action):
        assert action.shape[0] == self._adim, action
        target = no_rot_dynamics(self._previous_target_qpos, action)
        target = clip_target_qpos(target, self.low_bound, self.high_bound)
        return target