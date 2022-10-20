'''
All cartgripper env modules built on cartrgipper implementation in
https://github.com/SudeepDasari/visual_foresight
'''
from abc import ABC

from mujoco_py import load_model_from_path, MjSim
import numpy as np
from .base_env import BaseEnv


class BaseMujocoEnv(BaseEnv, ABC):
    def __init__(self, model_path, _hp):
        super(BaseMujocoEnv, self).__init__()
        self._frame_height = _hp.viewer_image_height
        self._frame_width = _hp.viewer_image_width

        self._reset_sim(model_path)

        self._base_adim, self._base_sdim = None, None  #state/action dimension of Mujoco control
        self._adim, self._sdim = None, None  #state/action dimension presented to agent
        self.num_objects, self._n_joints = None, None
        self._goal_obj_pose = None
        self._goaldistances = []

        self._ncam = _hp.ncam
        if self._ncam == 2:
            self.cameras = ['maincam', 'leftcam']
        elif self._ncam == 1:
            self.cameras = ['maincam']
        else:
            raise ValueError

        self._last_obs = None
        self._hp = _hp

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params['viewer_image_height'] = 256
        parent_params['viewer_image_width'] = 256
        parent_params['ncam'] = 1

        return parent_params

    def set_goal_obj_pose(self, pose):
        self._goal_obj_pose = pose

    def _reset_sim(self, model_path):
        """
        Creates a MjSim from passed in model_path
        :param model_path: Absolute path to model file
        :return: None
        """
        self._model_path = model_path
        self.sim = MjSim(load_model_from_path(self._model_path))

    def reset(self):
        self._goaldistances = []

    def render(self):
        """ Renders the enviornment.
        Implements custom rendering support. If mode is:

        - dual: renders both left and main cameras
        - left: renders only left camera
        - main: renders only main (front) camera
        :param mode: Mode to render with (dual by default)
        :return: uint8 numpy array with rendering from sim
        """
        images = np.zeros(
            (self._ncam, self._frame_height, self._frame_width, 3),
            dtype=np.uint8)
        for i, cam in enumerate(self.cameras):
            images[i] = self.sim.render(
                self._frame_width, self._frame_height, camera_name=cam)
        return images


    @property
    def adim(self):
        return self._adim

    @property
    def sdim(self):
        return self._sdim

    @property
    def ncam(self):
        return self._ncam
