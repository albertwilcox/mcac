import mcac.utils as utils

import argparse
import numpy as np


def parse_args():
    initial_parser = argparse.ArgumentParser()
    initial_parser.add_argument('--algo', required=True)
    initial_args, _ = initial_parser.parse_known_args()
    algo = initial_args.algo

    parser = argparse.ArgumentParser()

    # Experiment arguments
    parser.add_argument('--algo')
    parser.add_argument('--exper-name', type=str, default=None,
                        help='Experiment name to be used for output directory')
    parser.add_argument('--env', type=str, default='spb',
                        help='environment name')
    parser.add_argument('--do-dense', action='store_true')
    parser.add_argument('--supervisor', type=int, default=0,
                        help='The index of the supervisor you would like to use')
    parser.add_argument('--expert-file', type=str, default='experts/spb.pth',
                        help='path to saved expert for algorithmic supervisors')
    parser.add_argument('--no-offline-data', action='store_true',
                        help='If true, will not load offline data')
    parser.add_argument('--seed', '-s', type=int, default=-1,
                        help='random seed')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA devide to use')
    parser.add_argument('--gen-data', action='store_true',
                        help='add flag if we need to generate data')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint from which to load models')
    parser.add_argument('--rb-checkpoint', type=str, default=None)

    # Training arguments
    parser.add_argument('--train-episodes', type=int, default=100,
                        help='Number of training episodes')
    parser.add_argument('--total-timesteps', type=int, default=int(1e6))
    parser.add_argument('--init-iters', type=int, default=0,
                        help='How many iterations of pretraining to run')
    parser.add_argument('--buffer-size', type=int, default=int(1e6),
                        help='Replay buffer size')
    parser.add_argument('--n-demos', type=int, default=50,
                        help='How many demos to collect/load')
    parser.add_argument('--update-n-steps', type=int, default=1,
                        help='How many gradient steps to take each timestep')
    parser.add_argument('--eval-freq', type=int, default=int(1e3),
                        help='How many interacts between evaluating policy')
    parser.add_argument('--save-freq', type=int, default=int(1e4))
    parser.add_argument('--num-eval-episodes', default=10, type=int,
                        help='How many episodes to evaluate over')
    parser.add_argument('--start-timesteps', default=0, type=int,
                        help='How many timesteps to use initial random policy')
    parser.add_argument('--reward-shift', type=float, default=0)
    parser.add_argument('--reward-scale', type=float, default=1)
    
    # MCAC
    parser.add_argument('--do-mcac-bonus', action='store_true')


    if algo == 'sac':
        add_sac_args(parser)

    if algo == 'td3':
        add_td3_args(parser)

    if algo == 'gqe':
        add_gqe_args(parser)

    if algo == 'awac':
        add_awac_args(parser)

    if algo == 'cql':
        add_cql_args(parser)

    args = parser.parse_args()

    params = vars(args)
    params['algo'] = algo

    return params


def add_sac_args(parser):
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--discount', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic-entropy-tuning', action='store_true',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--target-update-interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--q-ensemble-size', type=int, default=2)


def add_td3_args(parser):
    parser.add_argument("--expl-noise", type=float, default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch-size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument('--batch-size-demonstrator', type=int, default=128,
                        help='Batch size for demonstrator BC loss, N_D in overcoming exploration paper')
    parser.add_argument("--discount", type=float, default=0.99)  # Discount factor
    parser.add_argument("--tau", type=float, default=0.005)  # Target network update rate
    parser.add_argument("--policy-noise", type=float, default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise-clip", type=float, default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy-freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=3e-4)

    parser.add_argument('--bc-weight', type=float, default=1,
                        help='weight for behavior cloning loss, recommended 1000/batch_size')
    parser.add_argument('--do-bc-loss', action='store_true', help='Whether or not to do a bc loss')
    parser.add_argument('--do-q-filter', action='store_true',
                        help='whether or not to use a q-filter for BC loss')
    parser.add_argument('--bc-decay', type=float, default=1)


def add_awac_args(parser):
    parser.add_argument('--update-freq', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--critic-batch-size', type=int, default=256)
    parser.add_argument('--beta', type=float, default=2.0)

    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--p_lr', type=float, default=3e-4)
    parser.add_argument('--polyak', type=float, default=0.995)


def add_gqe_args(parser):
    add_sac_args(parser)
    parser.add_argument('--gqe', action='store_true')
    parser.add_argument('--gqe-lambda', type=float, default=.9)
    parser.add_argument('--gqe-n', type=int, default=32)

def add_cql_args(parser):
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--update-freq', type=int, default=1) # Not sure if we should use that one for CQL, set to 1.
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument("--hidden_size", type=int, default=32) # 256 both in the github and in awc for comparaison's purposes
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--cql-weight", type=float, default=1.0)
    parser.add_argument("--target-action-gap", type=float, default=10)
    parser.add_argument("--with-lagrange", type=int, default=0)
    parser.add_argument("--tau", type=float, default=5e-3)



