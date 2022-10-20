# Monte Carlo Augmented Actor-Critic for Sparse Reward Deep Reinforcement Learning from Suboptimal Demonstrations

Author implementation of 'Monte Carlo Augmented Actor-Critic for Sparse Reward Deep Reinforcement Learning from Suboptimal Demonstrations'

Read the paper [here](https://arxiv.org/abs/2210.07432).

## Installation

1. Make a python3.7+ virtualenv: `virtualenv venv --python=/path/to/python3.7`
2. Activate it: `source venv/bin/activate`
3. Install `pip install -e .`. Requires python 3.7+.
4. (Optional) If you want to run robotics experiments, download the mujoco200 binary and licence [here](https://www.roboti.us/index.html). Run `pip install mujoco-py==2.0.2.13`.
5. (Optional) If you want to run Robosuite experiments (Lift and Door), `pip install robosuite`

## Running

Simply run the commands in `commands.sh`. 
The first time you run commands for a particular
enviornment you'll need to add the `--gen-data` flag
in order to generate the necessary offline data.

## Bibtex

```
@inproceedings{
    wilcox2022monte,
    title={Monte Carlo Augmented Actor-Critic for Sparse Reward Deep Reinforcement Learning from Suboptimal Demonstrations},
    author={Albert Wilcox and Ashwin Balakrishna and Jules Dedieu and Wyame Benslimane and Daniel S. Brown and Ken Goldberg},
    booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
    year={2022},
    url={https://openreview.net/forum?id=FLzTj4ia8BN}
}
```