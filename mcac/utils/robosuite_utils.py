import gym
from gym.wrappers import TimeLimit
import numpy as np
import moviepy.editor as mpy
import os
from skimage.transform import resize

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite import load_controller_config
from robosuite.utils.transform_utils import pose2mat


horizons = {
    'NutAssembly': 200,
    'Lift': 50,
    'Door': 50,
    'TwoArmPegInHole': 50
}


class RSWrapper(gym.Wrapper):
    """
    Infers goal info, adds mask info (always 1), and shifts reward down by -1
    """
    def step(self, action):
        next_obs, rew, done, info = super(RSWrapper, self).step(action)
        info['goal'] = rew
        rew -= 1
        done = int(rew == 0)
        return next_obs, rew, done, info


class RSGymWrapper(GymWrapper):
    """
    Modified version of
    https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/wrappers/gym_wrapper.py
    to do the following:
        > add state data to the info array returned alongside the image-based observation
    """

    def __init__(self, env, keys=None):
        # Run super method
        if keys is None:
            keys = [
                'object-state', 'robot0_proprio-state'
            ]
        super().__init__(env=env, keys=keys)
        self.state = None

    def reset(self, **kwargs):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.
        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        self.state = self._flatten_obs(ob_dict)
        return self._process_image(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment
        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        info['next_state'] = self._flatten_obs(ob_dict)
        self.state = info['next_state']
        return self._process_image(ob_dict), reward, done, info

    @staticmethod
    def _process_image(ob_dict):
        im = ob_dict['agentview_image']
        im = np.flip(im, axis=0)
        im = (resize(im, (64, 64)) * 255).astype(np.uint8)
        im = im.transpose((2, 0, 1))
        return im


def get_config(env_name, camera_obs=False):
    controller_config = load_controller_config(default_controller='OSC_POSE')
    env_kwargs = {
        "env_name": env_name,
        "controller_configs": controller_config,
        "robots": [
            "UR5e" if env_name == 'NutAssembly' else 'Panda'
        ],
        "control_freq": 20,
        "hard_reset": False,
        "horizon": 1000,
        "ignore_done": True,
        "reward_scale": 1.0,
        'has_renderer': False,
        'has_offscreen_renderer': camera_obs,
        'use_object_obs': True,
        'use_camera_obs': camera_obs,
        'reward_shaping': False,
        'render_camera': "agentview",
        'keys': [
            'object-state', 'robot0_proprio-state',
        ]
    }
    if env_name == 'NutAssembly':
        env_kwargs['nut_type'] = 'round'
        env_kwargs['single_object_mode'] = 2
    if env_name == 'TwoArmPegInHole':
        env_kwargs['robots'] = [
            'Panda',
            'Panda'
        ]
        env_kwargs['keys'] = [
            'object-state', 'robot0_proprio-state', 'robot1_proprio-state',
        ]
    return env_kwargs


def make_env(env_name, from_images=False):
    env_kwargs = get_config(env_name, camera_obs=False)
    keys = env_kwargs.pop('keys')
    env = suite.make(
        **env_kwargs
    )
    if from_images:
        env = RSGymWrapper(env, keys=keys)
    else:
        env = GymWrapper(env, keys)
    env = RSWrapper(env)
    hor = horizons[env_name]
    env = TimeLimit(env, hor)
    return env
