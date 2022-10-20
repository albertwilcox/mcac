from mcac.algos import SAC, TD3, GQE, AWAC, CQL
import mcac.utils as utils
import mcac.utils.env_utils as eu
import mcac.utils.data_utils as du
import mcac.utils.pytorch_utils as ptu
from mcac.utils.arg_parser import parse_args
from mcac.utils.logx import EpochLogger

import numpy as np
from tqdm import trange
import os
import json


def main():
    params = parse_args()

    logdir = utils.get_file_prefix(params)
    params['data_folder'] = utils.get_data_dir(params)
    params['logdir'] = logdir

    utils.seed(params['seed'])
    os.makedirs(logdir)
    ptu.setup(params['device'])
    with open(os.path.join(logdir, 'hparams.json'), 'w') as f:
        json.dump(params, f)

    env, test_env = eu.make_env(params)

    logger = EpochLogger(output_dir=logdir, exp_name=params['exper_name'])

    if params['algo'] == 'sac':
        agent = SAC(params)
    elif params['algo'] == 'td3':
        agent = TD3(params)
    elif params['algo'] == 'gqe':
        agent = GQE(params)
    elif params['algo'] == 'awac':
        agent = AWAC(params)
    elif params['algo'] == 'cql':
        agent = CQL(params)

    if params['gen_data']:
        expert_policy = eu.make_expert_policy(params, test_env)
        du.generate_offline_data(test_env, expert_policy, params)
    replay_buffer = du.load_replay_buffer(params)

    if params['checkpoint'] is not None:
        agent.load(params['checkpoint'])
    else:
        print('Pretraining Policy')
        os.makedirs(os.path.join(logdir, 'pretrain_plots'))
        for i in trange(params['init_iters']):
            info = agent.update(replay_buffer)
        if params['init_iters'] > 0:
            agent.save(os.path.join(logdir, 'pretrain'))

    if params['rb_checkpoint'] is not None:
        replay_buffer.load(params['rb_checkpoint'])

    # Run training loop
    # Prepare for interaction with environment
    i = 0
    n_episodes = 0
    epoch = 0
    robosuite = params['env'] in eu.robosuite_envs

    total_timesteps = params['total_timesteps']

    while i < total_timesteps:
        # Collect one trajectory
        obs, done, t = env.reset(), False, 0
        ep_buf, rets = [], []
        while not done and t < params['horizon']:
            # Every params['eval_freq'] timesteps, run the evaluation loop and output logs
            if i % params['eval_freq'] == 0:
                do_eval(agent, test_env, logger, params['num_eval_episodes'], epoch, i, robosuite)
                epoch += 1
            
            if i % params['save_freq'] == 0:
                agent.save(os.path.join(logdir, f'models/{i}'))
                replay_buffer.save(os.path.join(logdir, f'rb/{i}'))

            # Begin policy updates
            if i < params['start_timesteps']:
                act = env.action_space.sample()
            else:
                act = agent.select_action(obs)
                if params['algo'] == 'td3':
                    act = (agent.select_action(obs) +
                       np.random.normal(0, params['max_action'] * params['expl_noise'],
                                        size=params['d_act']))\
                    .clip(-params['max_action'], params['max_action'])

            next_obs, rew, done, info = env.step(act)
            ep_buf.append({
                'obs': obs,
                'next_obs': next_obs,
                'act': act,
                'rew': utils.shift_reward(rew, params),
                'done': done,
                'expert': 0,
                'goal': info['goal'] if 'goal' in info else 0,
                'mask': info['mask'] if 'mask' in info
                else (1 if t == params['horizon'] else float(not done))

            })
            obs = next_obs

            i += 1
            t += 1
            rets.append(rew)

            # grad steps
            if i >= params['start_timesteps']:
                for _ in range(params['update_n_steps']):
                    if len(replay_buffer) == 0:
                        break
                    update_info = agent.update(replay_buffer)
                    logger.store(**update_info)

        # Calculate discounted reward to go for each transition in the episode
        x, succ = 0, 0
        for j, transition in enumerate(reversed(ep_buf)):
            if j == 0:
                succ = succ or transition['goal']
                if not transition['mask']:
                    x = transition['rew']
                else:
                    # Set drtg to infinite discounted reward sum.
                    # reward_estimate = np.median(rets)
                    reward_estimate = ep_buf[-1]['rew']
                    if params['discount'] < 1:
                        x = reward_estimate / (1 - params['discount'])
                    else:
                        x = reward_estimate * float('inf')
            else:
                x = transition['rew'] + transition['mask'] * params['discount'] * x

            transition['drtg'] = x
            transition['succ'] = succ
            del transition['goal']

        for transition in ep_buf:
            replay_buffer.store_transition(transition)

        if robosuite:
            env.close()

        logger.store(TrainEpRet=sum(rets), TrainEpLen=len(rets))
        n_episodes += 1

def do_eval(agent, test_env, logger, num_eval_episodes, epoch, i, robosuite):
    print('Testing Agent')
    for _ in range(num_eval_episodes):
        obs, done, ep_ret, ep_len = test_env.reset(), False, 0, 0
        while not done:
            # Take deterministic actions at test time (noise_scale=0)
            act = agent.select_action(obs, evaluate=True)
            next_obs, rew, done, info = test_env.step(act)
            ep_ret += rew
            ep_len += 1
            obs = next_obs
        if robosuite:
            test_env.close()
        logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Log info about epoch
    logger.log_tabular('Epoch', epoch)
    logger.log_tabular('TotalEnvInteracts', i)
    logger.log_tabular('TestEpRet')
    logger.log_tabular('TestEpLen', average_only=True)
    if epoch == 0:
        logger.log_tabular('AverageTrainEpRet', 0)
        logger.log_tabular('StdTrainEpRet', 0)
        logger.log_tabular('TrainEpLen', 0)
        logger.log_tabular('Q1', 0)
        logger.log_tabular('Q2', 0)
    else:
        logger.log_tabular('TrainEpRet')
        logger.log_tabular('TrainEpLen', average_only=True)
        logger.log_tabular('Q1', average_only=True)
        logger.log_tabular('Q2', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    main()
