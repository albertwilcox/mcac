"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise. State
representation is (x, y). Action representation is (dx, dy).
"""

import numpy as np
from gym import Env
from gym import utils
from gym import Wrapper
from gym.spaces import Box

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os



class SimplePointBot(Env, utils.EzPickle):

    def __init__(self,
                 walls=(((75, 55), (100, 95)),),
                 window_width=180,
                 window_height=150,
                 max_force=3,
                 start_pos=(30, 75),
                 end_pos=(150, 75),
                 horizon=100,
                 constr_penalty=-100,
                 goal_thresh=3,
                 noise_scale=0.125):
        utils.EzPickle.__init__(self)
        self._max_episode_steps = horizon
        self.start = start_pos
        self.goal = end_pos
        self._goal_thresh = goal_thresh
        self._noise_scale = noise_scale
        self._constr_penalty = constr_penalty
        self.window_width = window_width
        self.window_height = window_height
        self.max_force = max_force
        self.action_space = Box(-np.ones(2), np.ones(2))
        self.observation_space = Box(-np.ones(2) * np.float('inf'), np.ones(2) * np.float('inf'))
        self.walls = [self._complex_obstacle(wall) for wall in walls]
        self.wall_coords = walls

        self._done = self.state = None
        self._episode_steps = 0

    def step(self, a):
        a = self._process_action(a)
        old_state = self.state.copy()
        next_state = self._next_state(self.state, a)
        cur_reward = self.step_reward(self.state, a)
        self.state = next_state
        self._episode_steps += 1
        constr = self.obstacle(next_state)
        self._done = self._episode_steps >= self._max_episode_steps
        mask = 1
        if constr:
            self._done = True
            cur_reward = self._constr_penalty
            mask = 0
        if cur_reward == 0:
            self._done = True
            mask = 0

        obs = self.state / (self.window_width, self.window_height)
        return obs, cur_reward, self._done, {
            "constraint": constr,
            'goal': cur_reward == 0,
            'mask': mask
        }

    def reset(self, random_start=False):
        if random_start:
            self.state = np.random.random(2) * (self.window_width, self.window_height)
            if self.obstacle(self.state):
                self.reset(True)
        else:
            self.state = self.start + np.random.randn(2)
        self._done = False
        self._episode_steps = 0
        obs = self.state / (self.window_width, self.window_height)
        return obs

    def render(self, mode='human'):
        return self._draw_state(self.state)

    def _next_state(self, s, a, override=False):
        if self.obstacle(s):
            return s

        next_state = s + a + self._noise_scale * np.random.randn(len(s))
        next_state = np.clip(next_state, (0, 0), (self.window_width, self.window_height))
        return next_state

    def step_reward(self, s, a):
        """
        Returns 1 if in goal otherwise 0
        """
        return int(np.linalg.norm(np.subtract(self.goal, s)) < self._goal_thresh) - 1

    def obstacle(self, s):
        return any([wall(s) for wall in self.walls])

    @staticmethod
    def _complex_obstacle(bounds):
        """
        Returns a function that returns true if a given state is within the
        bounds and false otherwise
        :param bounds: bounds in form [[X_min, Y_min], [X_max, Y_max]]
        :return: function described above
        """
        min_x, min_y = bounds[0]
        max_x, max_y = bounds[1]

        def obstacle(state):
            if type(state) == np.ndarray:
                lower = (min_x, min_y)
                upper = (max_x, max_y)
                state = np.array(state)
                component_viol = (state > lower) * (state < upper)
                return np.product(component_viol, axis=-1)
            if type(state) == torch.Tensor:
                lower = torch.from_numpy(np.array((min_x, min_y)))
                upper = torch.from_numpy(np.array((max_x, max_y)))
                component_viol = (state > lower) * (state < upper)
                return torch.prod(component_viol, dim=-1)

        return obstacle

    def _process_action(self, a):
        return np.clip(a, -1, 1) * self.max_force

    def draw(self, trajectories=None, heatmap=None, points=None, points2=None, point_colors=None, plot_starts=False, remove_axes=False,
             board=True,
             file=None,
             show=False):
        """
        Draws the desired trajectories and heatmaps (probably would be a safe set) to pyplot
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if heatmap is not None:
            assert heatmap.shape == (self.window_height, self.window_width)
            # heatmap = np.flip(heatmap, axis=0)
            im = plt.imshow(heatmap, cmap='cividis')
            plt.colorbar(im)

        if board:
            self.draw_board(ax)

        if trajectories is not None and type(trajectories) == list:
            if type(trajectories[0]) == list:
                self.plot_trajectories(ax, trajectories, plot_starts)
            if type(trajectories[0]) == dict:
                self.plot_trajectory(ax, trajectories, plot_starts)

        if points is not None:
            if points.max() <= 1:
                points = points * (self.window_width, self.window_height)
            if point_colors is not None:
                c = point_colors
            else:
                c = 'dimgrey'
            plt.scatter(points[:, 0], points[:, 1], marker=',', s=3, linewidths=0.1, c=c, cmap='inferno')
        if points2 is not None:
            plt.scatter(points2[:, 0], points2[:, 1], marker=',', linewidths=0.1, s=1,
                        color='tab:blue')

        ax.set_aspect('equal')
        ax.autoscale_view()
        plt.gca().invert_yaxis()

        if remove_axes:
            plt.tick_params(left=False, bottom=False)
            plt.xticks([])
            plt.yticks([])

        if file is not None:
            plt.savefig(file)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_trajectory(self, ax, trajectory, plot_start=False):
        self.plot_trajectories(ax, [trajectory], plot_start)

    def plot_trajectories(self, ax, trajectories, plot_start=False):
        """
        Renders a trajectory to pyplot. Assumes you already have a plot going
        :param ax:
        :param trajectories: Trajectories to impose upon the graph
        :param plot_start: whether or not to draw a circle at the start of the trajectory
        :return:
        """

        for trajectory in trajectories:
            states = np.array([frame['obs'] for frame in trajectory])
            plt.plot(states[:, 0], self.window_height - states[:, 1])
            if plot_start:
                start = states[0]
                start_circle = plt.Circle((start[0], self.window_height - start[1]), radius=2,
                                          color='lime')
                ax.add_patch(start_circle)

    def draw_board(self, ax):
        plt.xlim(0, self.window_width)
        plt.ylim(0, self.window_height)

        for wall in self.wall_coords:
            width, height = np.subtract(wall[1], wall[0])
            ax.add_patch(
                patches.Rectangle(
                    xy=wall[0],  # point of origin.
                    width=width,
                    height=height,
                    linewidth=1,
                    color='red',
                    fill=True
                )
            )

        circle = plt.Circle(self.start, radius=3, color='k')
        ax.add_patch(circle)
        circle = plt.Circle(self.goal, radius=3, color='k')
        ax.add_patch(circle)


class SlitPointBot(SimplePointBot):
    def __init__(self):
        super(SlitPointBot, self).__init__(window_width=180,
                                           window_height=150,
                                           start_pos=(20, 75),
                                           end_pos=(160, 75),
                                           walls=[
                                               ((80, 0), (100, 40)),
                                               ((80, 45), (100, 150)),
                                           ],
                                           horizon=100,
                                           constr_penalty=-100)


def spb_expert(obs):
    obs = obs * (180, 150)
    x, y = obs
    if x < 101:
        if 40 < y < 45:
            goal = (105, 42.5)
        else:
            goal = (78, 42.5)
    else:
        goal = (160, 75)

    act = np.subtract(goal, obs) / 2
    act = act / np.max(np.abs(act))
    return act
