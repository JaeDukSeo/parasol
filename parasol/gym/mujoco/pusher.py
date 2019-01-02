from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_nips import  SawyerPushAndReachXYEnv
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from deepx import T
import numpy as np
import os
import cv2
from gym import utils
from gym.spaces import Box
import scipy.misc

from ..gym_wrapper import GymWrapper

__all__ = ['Pusher']

class SawyerPushXY(utils.EzPickle, SawyerPushAndReachXYEnv):

    def __init__(self, 
                image=False,
                image_dim=84,
                reward_info=None,
                frame_skip=50,
                pos_action_scale=2. / 100,
                randomize_goals=False,
                hide_goal=False,
                init_block_low=(-0.05, 0.6),
                init_block_high=(0.05, 0.65),
                puck_goal_low=(-0.05, 0.55),
                puck_goal_high=(0.05, 0.65),
                hand_goal_low=(-0.05, 0.55),
                hand_goal_high=(0.05, 0.65),
                fixed_puck_goal=(0.0, 0.75),
                fixed_hand_goal=(-0.05, 0.6),
                mocap_low=(-0.25, 0.2, -0.0),
                mocap_high=(0.25, 0.8, 0.2),
                **kwargs
        ):
        # utils.EzPickle.__init__(self)
        self.quick_init(locals())
        self.reward_info = reward_info
        self.randomize_goals = randomize_goals
        self._pos_action_scale = pos_action_scale
        self.hide_goal = hide_goal
        self.init_block_low = np.array(init_block_low)
        self.init_block_high = np.array(init_block_high)
        self.puck_goal_low = np.array(puck_goal_low)
        self.puck_goal_high = np.array(puck_goal_high)
        self.hand_goal_low = np.array(hand_goal_low)
        self.hand_goal_high = np.array(hand_goal_high)
        self.fixed_puck_goal = np.array(fixed_puck_goal)
        self.fixed_hand_goal = np.array(fixed_hand_goal)
        self.mocap_low = np.array(mocap_low)
        self.mocap_high = np.array(mocap_high)
        self._goal_xyxy = self.sample_goal_xyxy()
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)
        self.action_space = Box(
            np.array([-1, -1]),
            np.array([1, 1]),
        )
        self.obs_box = Box(
            np.array([-0.2, 0.5, -0.2, 0.5]),
            np.array([0.2, 0.7, 0.2, 0.7]),
        )
        goal_low = np.concatenate((self.hand_goal_low, self.puck_goal_low))
        goal_high = np.concatenate((self.hand_goal_high, self.puck_goal_high))
        self.goal_box = Box(
            goal_low,
            goal_high,
        )
        self.image = image
        self.image_dim = image_dim
        self.INIT_HAND_POS = np.array([0.0, 0.5, 0.02])
        if self.image:
            self.observation_space = Box(0, 1, (3*self.image_dim**2,), dtype=np.float32)
        else:
            self.observation_space = Box(0, 1, (4,), dtype=np.float32)
        self.reset()
        self.reset_mocap_welds()

    def sample_puck_xy(self):
        pos = np.random.uniform(self.init_block_low, self.init_block_high)
        while np.linalg.norm(self.get_endeff_pos()[:2] - pos) < 0.05:
            pos = np.random.uniform(self.init_block_low, self.init_block_high)
        return pos

    @property
    def init_angles(self):
        return [ 1.84605891e+00, -5.29572854e-01, -4.20794719e-01,  2.16590819e+00,
              1.63634186e+00,  3.61181024e-01,  1.72814749e+00, -1.09706637e-02,
              6.09265205e-01,  2.59895360e-02,  9.99999990e-01,  2.80573332e-05,
             -3.47280097e-06,  1.38684523e-04, -5.00000000e-02,  6.00000000e-01,
              2.09686080e-02,  7.07106781e-01,  1.54372126e-14,  7.07106781e-01,
             -1.54484227e-14, -3.88789679e-18,  7.50000000e-01,  2.09686080e-02,
              7.07106781e-01,  4.46056581e-16,  7.07106781e-01, -4.60012116e-16]

    @property
    def model_name(self):
        return os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets", "sawyer_pusher.xml")

    @property
    def endeff_id(self):
        return self.model.body_names.index('center')

    def _get_obs(self):
        if self.image:
            image_obs = self.get_image(camera_name="robotview").transpose()/255.0
            image_obs = image_obs.reshape((3, 84, 84))
            image_obs = np.rot90(image_obs,  axes=(-2, -1))
            image_obs = np.transpose(image_obs, [1, 2, 0])[10:74,10:74,:]
            return image_obs.flatten()
        else:
            end_eff = self.get_endeff_pos()[:2]
            puck = self.get_puck_pos()[:2]
            return np.concatenate((end_eff, puck))

    def reset(self):
        velocities = self.data.qvel.copy()
        angles = np.array(self.init_angles)

        self.set_state(angles.flatten(), velocities.flatten())
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.INIT_HAND_POS)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)
        # set_state resets the goal xy, so we need to explicit set it again
        self._goal_xyxy = self.sample_goal_for_rollout()
        self.set_goal_xyxy(self._goal_xyxy)
        self.set_puck_xy(self.sample_puck_xy())
        self.reset_mocap_welds()
        return self._get_obs()

    def step(self, a):
        a = np.clip(a, -1, 1)
        mocap_delta_z = 0.02 - self.data.mocap_pos[0, 2]
        new_mocap_action = np.hstack((
            a,
            np.array([mocap_delta_z])
        ))
        self.mocap_set_action(new_mocap_action[:3] * self._pos_action_scale)
        u = np.zeros(7)
        self.do_simulation(u, self.frame_skip)
        obs = self._get_obs()
        done = False

        hand_distance = np.linalg.norm(
            self.get_hand_goal_pos() - self.get_endeff_pos()
        )
        puck_distance = np.linalg.norm(
            self.get_puck_goal_pos()[:2] - self.get_puck_pos()[:2])
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_puck_pos())
        info = dict(
            hand_distance=hand_distance,
            puck_distance=puck_distance,
            touch_distance=touch_distance,
            success=float(puck_distance < 0.03),
        )
        reward = -puck_distance
        return obs, reward, done, info


class Pusher(GymWrapper):

    environment_name = 'Pusher'
    entry_point = "parasol.gym.mujoco.pusher:SawyerPushXY"
    max_episode_steps = 50
    reward_threshold = -np.inf

    def __init__(self, **kwargs):
        config = {
            'image': kwargs.pop('image', True),
            'sliding_window': kwargs.pop('sliding_window', 0),
            'default_goal': kwargs.pop('default_goal', [0.0, 0.75]),
            'image_dim': kwargs.pop('image_dim', 64),
        }
        super(Pusher, self).__init__(config)

    def make_summary(self, observations, name):
        if self.image:
            observations = T.reshape(observations, [-1] + self.image_size())
            T.core.summary.image(name, observations)

    def is_image(self):
        return self.image

    def image_size(self):
        if self.image:
            return [self.image_dim, self.image_dim, 3]
        return None
    
    def cost_fn(self, s, a):
        return np.linalg.norm(s[:,-2:] - self.fixed_puck_goal, axis=-1)

