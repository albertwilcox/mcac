import os
import torch
import numpy as np
import random
from datetime import datetime
from collections.abc import Iterable


def seed(s, envs=None):
    if s == -1:
        return

    # torch.set_deterministic(True)
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)

    if envs is not None:
        if isinstance(envs, Iterable):
            for env in envs:
                env.seed(s)
                env.action_space.seed(s)
        else:
            envs.seed(s)
            envs.action_space.seed(s)


def get_file_prefix(params=None):
    if params is not None and params['exper_name'] is not None:
        folder = os.path.join('outputs', params['exper_name'])
    else:
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d/%H-%M-%S")
        folder = os.path.join('outputs', date_string)
    if params is not None and params['seed'] != -1:
        folder = os.path.join(folder, str(params['seed']))
    return folder


def get_data_dir(params):
    return os.path.join('data', params['env'], str(params['supervisor']))


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    color2num = dict(
        gray=30,
        red=31,
        green=32,
        yellow=33,
        blue=34,
        magenta=35,
        cyan=36,
        white=37,
        crimson=38
    )

    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def shift_reward(rew, params):
    return (rew + params['reward_shift']) * params['reward_scale']


def add_dicts(*args):
    out = {}
    for arg in args:
        for k, v in arg.items():
            out[k] = v
    return out


__all__ = ['seed', 'get_file_prefix', 'get_data_dir', 'colorize', 'shift_reward', 'add_dicts']