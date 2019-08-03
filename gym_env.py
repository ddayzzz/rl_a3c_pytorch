# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import numpy as np
import cv2
import gym


def process_frame(frame):
    """
    处理原始的 A3C 的图像
    :param frame: 210x160x3
    :return: 处理过后的一帧图像
    """
    # 去掉分数和边缘
    x_t = frame.astype(np.float32)  # 生成一个新的对象
    x_t = x_t[34:34 + 160, :160, :]  # 截取
    x_t = cv2.resize(x_t, (84, 84))  # 转换为普通的图像
    x_t /= 255.0  # 标准化
    x_t = np.transpose(x_t, [2, 0, 1])
    return x_t


class GymEnvironment(object):

    @staticmethod
    def get_action_size(env_name):
        env = gym.make(env_name)
        action_size = env.action_space.n
        env.close()
        return action_size

    @staticmethod
    def get_observation_size(env_name):
        return [3, 84, 84]  # todo: 结果需压迫更加泛化!

    def __init__(self, env_name, human_control=False, seed=None, frame_skip=4,
                 enable_debug_observation=False):
        """
        GYM 定义环境， 其他参数可能与 基本的有冲突
        :param env_name:
        :param human_control:
        :param seed:
        :param frame_skip:
        :param state_is_rgb:
        :param enable_debug_observation:
        :param enable_encode_depth:
        """
        self.env_name = env_name
        self.enable_debug_observation = enable_debug_observation
        self.env = gym.make(env_name)
        self.reset()

    def reset(self):
        rgb_state_uint8 = self.env.reset()
        state = process_frame(rgb_state_uint8)

        self.last_state = state
        self.last_action = 0
        self.last_reward = 0

    def close(self):
        self.env.close()
        print("gym environment stopped")

    # pixel change
    def _subsample(self, a, average_width):
        s = a.shape
        sh = s[0] // average_width, average_width, s[1] // average_width, average_width
        return a.reshape(sh).mean(-1).mean(1)

    def _calc_pixel_change(self, state, last_state):
        d = np.absolute(state[:, 2:-2, 2:-2] - last_state[:, 2:-2, 2:-2])
        # (3,80,80)
        m = np.mean(d, 0)
        c = self._subsample(m, 4)
        return c

    def step(self, action):
        obs, reward, terminal, info = self.env.step(action)

        state = process_frame(obs)

        pixel_change = self._calc_pixel_change(state, self.last_state)
        self.last_state = state
        self.last_action = action
        self.last_reward = reward
        if self.enable_debug_observation:
            return state, reward, terminal, pixel_change, obs  # 显示带有分数的图像
        else:
            return state, reward, terminal, pixel_change

    def key_to_action(self):
        env = gym.make(self.env_name)
        r = env.unwrapped.get_keys_to_action()
        env.close()
        return r
