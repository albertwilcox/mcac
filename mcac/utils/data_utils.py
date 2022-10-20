import mcac.utils as utils

import os
import numpy as np
import json


def save_trajectory(trajectory, data_folder, i):
    # If the observations are images save them as separate numpy arrays
    do_image_filtering = len(trajectory[0]['obs'].shape) == 3
    if do_image_filtering:
        im_fields = ('obs', 'next_obs')
        for field in im_fields:
            if field in trajectory[0]:
                dat = np.array([frame[field] for frame in trajectory], dtype=np.uint8)
                np.save(os.path.join(data_folder, "%d_%s.npy" % (i, field)), dat)
        traj_save = [{key: frame[key] for key in frame if key not in im_fields}
                     for frame in trajectory]
    else:
        traj_save = trajectory

    for frame in traj_save:
        for key in frame:
            if type(frame[key]) == np.ndarray:
                frame[key] = tuple(frame[key].tolist())

    with open(os.path.join(data_folder, "%d.json" % i), "w") as f:
        json.dump(traj_save, f)


def load_trajectory(data_folder, i):
    with open(os.path.join(data_folder, '%d.json' % i), 'r') as f:
        trajectory = json.load(f)

    # Add images stored as .npy files if there is no obs in the json
    add_images = 'obs' not in trajectory[0]
    if add_images:
        im_fields = ('obs', 'next_obs')
        im_dat = {}

        for field in im_fields:
            f = os.path.join(data_folder, "%d_%s.npy" % (i, field))
            if os.path.exists(data_folder):
                dat = np.load(f)
                im_dat[field] = dat.astype(np.uint8)

        for j, frame in list(enumerate(trajectory)):
            for key in im_dat:
                frame[key] = im_dat[key][j]

    return trajectory


def load_replay_buffer(params):
    replay_buffer = utils.ReplayBuffer(params['buffer_size'])
    for i in range(params['n_demos']):
        trajectory = load_trajectory(params['data_folder'], i)
        x = utils.shift_reward(trajectory[-1]['rew'], params) / (1 - params['discount'])
        for transition in reversed(trajectory):
            transition['rew'] = utils.shift_reward(transition['rew'], params)
            x = transition['rew'] + transition['mask'] * params['discount'] * x
            transition['drtg'] = x
            transition['succ'] = 1

        replay_buffer.store_trajectory(trajectory)

    return replay_buffer


def generate_expert_trajectory(env, expert_policy, params):
    obs, total_ret, done, t, completed = env.reset(), 0, False, 0, False
    trajectory = []
    while not done:
        act = expert_policy(obs)
        if act is None:
            done, rew = True, 0
            continue
        next_obs, rew, done, info = env.step(act)

        trajectory.append({
            'obs': obs,
            'next_obs': next_obs,
            'act': act.astype(np.float64),
            'rew': rew,
            'done': done,
            'mask': info['mask'] if 'mask' in info
            else (1 if t + 1 == params['horizon'] else float(not done)),
            'expert': 1
        })

        total_ret += rew
        if 'goal' in info:
            completed = completed or info['goal']
        else:
            completed = True
        t += 1
        obs = next_obs

    return trajectory, completed, total_ret


def generate_offline_data(env, expert_policy, params):
    # Runs expert policy in the environment to collect data
    i = 0
    total_rews = []
    act_limit = env.action_space.high[0]
    try:
        os.makedirs(params['data_folder'])
    except FileExistsError:
        x = input(
            'Data already exists. Type `o` to overwrite, type anything else to skip data collection... > ')
        if x.lower() != 'o':
            return
    while i < params['n_demos']:
        print('Collecting demo %d' % i)
        trajectory, completed, total_ret = generate_expert_trajectory(env, expert_policy, params)

        if completed:  # only count successful episodes
            save_trajectory(trajectory, params['data_folder'], i)
            i += 1
        else:
            print('Trajectory unsuccessful, redoing')
        env.close()
        total_rews.append(total_ret)


