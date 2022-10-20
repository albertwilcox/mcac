from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 7, \
    "You should have Python 3.7 and greater." 

setup(
    name='mcac',
    py_modules=['mcac'],
    version='0.0.1',
    install_requires=[
        'numpy',
        'gym',
        'joblib',
        'matplotlib==3.1.1',
        'torch',
        'tqdm',
        'moviepy',
        'opencv-python',
        'torchvision==0.7.0',
        'dotmap',
        'scikit-image',
        'mujoco-py==2.0.2.13',
        'robosuite==1.2.1',

    ],
    description="Code for Monte Carlo Augmented Actor Critic.",
    author="Albert Wilcox",
)
