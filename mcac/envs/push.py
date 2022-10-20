'''
Built on cartrgipper implementation in
https://github.com/SudeepDasari/visual_foresight
'''

import numpy as np
import moviepy.editor as mpy
import copy
from .base_mujoco_env import BaseMujocoEnv
from gym.spaces import Box
import os

FIXED_ENV = True
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


class Push(BaseMujocoEnv):
    def __init__(self, denser_reward=False):
        parent_params = super()._default_hparams()
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        self.reset_xml = os.path.join(envs_folder, 'assets', 'push.xml')
        super().__init__(self.reset_xml, parent_params)
        self._adim = 2
        self.substeps = 500
        self.low_bound = np.array([-0.4, -0.4, -0.05])
        self.high_bound = np.array([0.4, 0.4, 0.15])
        self.ac_high = np.ones(self._adim)
        self.ac_low = -self.ac_high
        self.action_space = Box(self.ac_low, self.ac_high)
        self.action_multiplier = 0.05
        self._previous_target_qpos = None
        self.target_height_thresh = 0.03
        self.object_fall_thresh = -0.03
        self.obj_y_dist_range = np.array([0.05, 0.05])
        self.obj_x_range = np.array([-0.03, -0.03])
        self.randomize_objects = not FIXED_ENV
        self.gt_state = GT_STATE
        # self._max_episode_steps = 150
        self.horizon = 150
        self._num_steps = 0
        self.denser_reward = denser_reward

        # Expert params
        self._expert_block = 0
        # self._expert_reset_done = False

        # self._viol = False

        if self.gt_state:
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(27,))
        else:
            self.observation_space = self.observation_space = Box(0, 1, shape=(3, 64, 64),
                                                                  dtype='float32')
        self.reset()
        self.num_objs = (self.position.shape[0] - 6) // 7

    def render(self):
        x = super().render()[:, ::-1].copy().squeeze()
        return x

    def reset(self, **kwargs):
        self._reset_sim(self.reset_xml)
        # clear our observations from last rollout
        self._last_obs = None
        self._expert_block = 0

        state = self.sim.get_state()
        pos = np.copy(state.qpos[:])
        pos[6:] = self.object_reset_poses().ravel()
        state.qpos[:] = pos
        self.sim.set_state(state)
        self._num_steps = 0
        # self._viol = False

        self.sim.forward()

        self._previous_target_qpos = copy.deepcopy(
            self.sim.data.qpos[:5].squeeze())
        self._previous_target_qpos[-1] = self.low_bound[-1]

        if self.gt_state:
            return pos
        else:
            return self.render()

    def step(self, action):
        # print("ACTION: ", action)
        action = np.clip(action, self.ac_low, self.ac_high) * self.action_multiplier
        # Add extra action dimensions that we have artificially removed
        action = np.array(list(action) + [0, 0])
        target_qpos = self._next_qpos(action)
        if self._previous_target_qpos is None:
            self._previous_target_qpos = target_qpos

        for st in range(self.substeps):
            alpha = st / (float(self.substeps) - 1)
            self.sim.data.ctrl[:] = alpha * target_qpos + \
                                    (1. - alpha) * self._previous_target_qpos
            self.sim.step()

        # self.sim.data.ctrl[:] = target_qpos
        # self.sim.step()

        self._previous_target_qpos = target_qpos
        # constraint = self._viol or self.topple_check()
        # self._viol = constraint
        # constraint = self.constraint_fn()
        goal = self.in_goal()
        reward = self.reward_fn()
        # if constraint:
        #     reward = -1

        self._num_steps += 1
        if EARLY_TERMINATION:
            done = goal
        else:
            done = self._num_steps >= self._max_episode_steps

        info = {
            "goal": goal,
        }

        if self.gt_state:
            return self.position, reward, done, info
        else:
            return self.render(), reward, done, info

    def topple_check(self, debug=False):
        quat = self.object_poses[:, 3:]
        phi = np.arctan2(
            2 *
            (np.multiply(quat[:, 0], quat[:, 1]) + quat[:, 2] * quat[:, 3]),
            1 - 2 * (np.power(quat[:, 1], 2) + np.power(quat[:, 2], 2)))
        theta = np.arcsin(2 * (np.multiply(quat[:, 0], quat[:, 2]) -
                               np.multiply(quat[:, 3], quat[:, 1])))
        psi = np.arctan2(
            2 * (np.multiply(quat[:, 0], quat[:, 3]) + np.multiply(
                quat[:, 1], quat[:, 2])),
            1 - 2 * (np.power(quat[:, 2], 2) + np.power(quat[:, 3], 2)))
        euler = np.stack([phi, theta, psi]).T[:, :2] * 180. / np.pi
        if debug:
            return np.abs(euler).max() > 15 or np.isnan(euler).sum() > 0, euler
        return np.abs(euler).max() > 15 or np.isnan(euler).sum() > 0

    @property
    def jaw_width(self):
        pos = self.position
        return 0.08 - (pos[4] - pos[5])  # 0.11 might be slightly off

    def set_y_range(self, bounds):
        self.obj_y_dist_range[0] = bounds[0]
        self.obj_y_dist_range[1] = bounds[1]

    def expert_action(self, noise_std=0.001, step_size=0.05):
        ac, reset_done = self.expert_action_old(block=self._expert_block,
                                                noise_std=noise_std,
                                                step_size=step_size)
        if reset_done:
            self._expert_block += 1

        return ac / self.action_multiplier

    def expert_action_old(self, block=0, noise_std=0.001, step_size=0.05):
        # Can make step_size smaller to make demos more slow
        # Can also increase noise_std but this may make demos less reliable
        # print("BLOCK ID: ", block)
        cur_pos = self.position[:3]
        cur_pos[1] += 0.05  # compensate for length of jaws

        block_done = False
        block_reset_done = False
        block_pos = self.object_poses[block][:3]
        action = np.zeros(self._adim)
        delta = block_pos - cur_pos
        if not block_done:
            if abs(delta[0]) > 1e-3:
                action[0] = delta[0]
            else:
                if abs(block_pos[1]) < 0.1:
                    action[1] = -step_size
                else:
                    block_done = True
        if block_done:
            if cur_pos[1] < 0.04:
                action[1] = step_size
            else:
                block_reset_done = True

        action = action + np.random.randn(self._adim) * noise_std
        action = np.clip(action, self.ac_low, self.ac_high)
        return action, block_reset_done

    def get_demo(self, noise_std=0.001):
        im_list = []
        obs_list = [self.reset()]
        ac_list = []
        block_id = 0
        num_steps = 0
        while block_id < self.num_objs and num_steps < self._max_episode_steps:
            ac, reset_done = self.expert_action(block=block_id, noise_std=noise_std)
            if reset_done:
                block_id += 1
            ns, r, done, info = self.step(ac)
            obs_list.append(ns)
            ac_list.append(ac)
            im_list.append(self.render().squeeze())
            num_steps += 1

        while num_steps < self._max_episode_steps:
            ac = 5 * np.random.randn(self._adim) * noise_std
            ns, r, done, info = self.step(ac)
            obs_list.append(ns)
            ac_list.append(ac)
            im_list.append(self.render().squeeze())
            num_steps += 1

        # npy_to_gif(im_list, "out") # vis stuff for debugging
        return obs_list, ac_list, im_list

    def get_rand_rollout(self):
        if np.random.random() < 0.6:  # TODO: may need to tune this
            obs_list, ac_list, im_list = self.get_demo(noise_std=0.01)
        else:
            im_list = []
            obs_list = [self.reset()]
            ac_list = []
            num_steps = 0
            while num_steps < self._max_episode_steps:
                ac = self.action_space.sample()
                ns, r, done, info = self.step(ac)
                obs_list.append(ns)
                ac_list.append(ac)
                im_list.append(self.render().squeeze())
                num_steps += 1

        npy_to_gif(im_list, "out")  # vis stuff for debugging
        return obs_list, ac_list, im_list

    def get_block_dones(self):
        block_dones = np.zeros(self.num_objs)
        for block in range(self.num_objs):
            block_pos = self.object_poses[block][:3]
            # TODO: maybe make this an interval rather than a threshold later
            if 0.075 < abs(block_pos[1]) and abs(block_pos[0]) < .2:
                block_dones[block] = 1
        return block_dones

    # def constraint_fn(self):
    #     block_constr = []
    #     for block in range(self.num_objs):
    #         block_pos = self.object_poses[block][:3]
    #         block_constr.append(abs(block_pos[2]) > .2)
    #     return any(block_constr)

    def in_goal(self):
        block_dones = self.get_block_dones()
        return bool(np.sum(block_dones) == len(block_dones)) # believe it or not the bool() is necessary

    def reward_fn(self):
        if self.denser_reward:
            block_dones = self.get_block_dones()
            return np.sum(block_dones) - 3
        else:
            return int(self.in_goal() - 1)

    def object_reset_poses(self):
        new_poses = np.zeros((3, 7))
        new_poses[:, 3] = 1
        if self.randomize_objects == True:
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
        target = no_rot_dynamics(self._previous_target_qpos, action)
        target = clip_target_qpos(target, self.low_bound, self.high_bound)
        return target


def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


if __name__ == '__main__':
    env = PushEnv()
    # env.get_demo()
    env.get_rand_rollout()
