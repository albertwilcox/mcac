import mcac.envs.simple_point_bot as spb
from mcac.utils.sac_supervisor import SacSupervisor
import mcac.utils.pytorch_utils as ptu

import os
import gym
from gym.wrappers import TimeLimit

robosuite_envs = ('Lift', 'Door')


def find_max_episode_steps(env):
    try:
        x = env._max_episode_steps
        return x
    except AttributeError:
        if hasattr(env, 'env'):
            return find_max_episode_steps(env.env)
        else:
            return False


def make_env(params):
    env_name = params['env']

    if env_name == 'navigation':
        env = spb.SlitPointBot()
        test_env = spb.SlitPointBot()
    elif env_name == 'push':
        from mcac.envs.push import Push

        env = TimeLimit(Push(denser_reward=True), max_episode_steps=150)
        test_env = TimeLimit(Push(denser_reward=True), max_episode_steps=150)
    elif env_name == 'extraction':
        from mcac.envs.obj_extraction import ObjExtraction

        env = TimeLimit(ObjExtraction(fixed=False), max_episode_steps=50)
        test_env = TimeLimit(ObjExtraction(fixed=False), max_episode_steps=50)
    elif env_name in robosuite_envs:
        import mcac.utils.robosuite_utils as ru

        env = ru.make_env(params['env'])
        test_env = ru.make_env(params['env'])
    else:
        env = gym.make(params['env'])
        test_env = gym.make(params['env'])

    mes = find_max_episode_steps(env)

    if mes:
        params['horizon'] = mes
    else:
        raise ValueError('unable to find time horizon')

    params['d_obs'] = env.observation_space.shape
    params['d_act'] = env.action_space.shape
    params['max_action'] = env.action_space.high[0]

    return env, test_env


def make_expert_policy(params, env):
    env_name = params['env']
    use_robosuite = env_name in robosuite_envs

    if env_name == 'navigation':
        expert_pol = spb.spb_expert
    elif env_name in ('push', 'extraction'):
        expert_pol = lambda _: env.expert_action(noise_std=0.004)
    elif use_robosuite:
        experts = {
            'Lift': 'lift.pkl',
            'Door': 'door.pkl',
            'TwoArmPegInHole': 'tapih.pkl'
        }
        expert = SacSupervisor(env.observation_space.shape[0], env.action_space.shape[0])
        expert.load_supervisor(os.path.join('supervisors', experts[env_name]))
        expert = expert.to(ptu.TORCH_DEVICE)
        expert_pol = lambda obs: expert.get_action(obs, True)
    else:
        raise ValueError('cannot load expert for that environment')

    return expert_pol


