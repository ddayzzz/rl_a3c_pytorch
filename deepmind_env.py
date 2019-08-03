# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import Pipe, Process

try:
    import deepmind_lab
    from pygame import locals
except ImportError:
    pass

import numpy as np


def _action(*entries):
    return np.array(entries, dtype=np.intc)


class LabEnvironment(object):
    # 有数值的位置 是 action spec 的对应动作的位置. RIGHT -LEFT, FORWEARD-BACKWARD 相对
    """
    Action spec:
  [{'max': 512, 'min': -512, 'name': 'LOOK_LEFT_RIGHT_PIXELS_PER_FRAME'},
   {'max': 512, 'min': -512, 'name': 'LOOK_DOWN_UP_PIXELS_PER_FRAME'},
   {'max': 1, 'min': -1, 'name': 'STRAFE_LEFT_RIGHT'},
   {'max': 1, 'min': -1, 'name': 'MOVE_BACK_FORWARD'},
   {'max': 1, 'min': 0, 'name': 'FIRE'},
   {'max': 1, 'min': 0, 'name': 'JUMP'},
   {'max': 1, 'min': 0, 'name': 'CROUCH'}]
  nav_static_maze_01
    """
    ACTION_LIST = [
        _action(-20, 0, 0, 0, 0, 0, 0),  # look_left
        _action(20, 0, 0, 0, 0, 0, 0),  # look_right
        # _action(  0,  10,  0,  0, 0, 0, 0), # look_up
        # _action(  0, -10,  0,  0, 0, 0, 0), # look_down
        _action(0, 0, -1, 0, 0, 0, 0),  # strafe_left
        _action(0, 0, 1, 0, 0, 0, 0),  # strafe_right
        _action(0, 0, 0, 1, 0, 0, 0),  # forward
        _action(0, 0, 0, -1, 0, 0, 0),  # backward
        # _action(  0,   0,  0,  0, 1, 0, 0), # fire
        # _action(  0,   0,  0,  0, 0, 1, 0), # jump
        # _action(  0,   0,  0,  0, 0, 0, 1)  # crouch
        # _action(0, 0, 0, 0, 0, 0, 0)  # -1, default action
    ]
    DEFAULT_ACTION = _action(0, 0, 0, 0, 0, 0, 0)

    ACTION_MEANING = {
        0: "LOOK_LEFT",
        1: "LOOK_RIGHT",
        2: "STRAFE_LEFT",
        3: "STRAFE_RIGHT",
        4: "FORWARD",
        5: "BACKWARD"
    }

    ACTION_NAME_TO_KEY = {
        'LOOK_LEFT': locals.K_4,
        'LOOK_RIGHT': locals.K_6,
        'STRAFE_LEFT': locals.K_LEFT,
        'STRAFE_RIGHT': locals.K_RIGHT,
        'FORWARD': locals.K_UP,
        'BACKWARD': locals.K_DOWN
    }
    ACTION_TO_KEY = {
        0: locals.K_4,
        1: locals.K_6,
        2: locals.K_LEFT,
        3: locals.K_RIGHT,
        4: locals.K_UP,
        5: locals.K_DOWN
    }
    KEY_TO_ACTION = {
        (locals.K_4,): 0,
        (locals.K_6,): 1,
        (locals.K_LEFT,): 2,
        (locals.K_RIGHT,): 3,
        (locals.K_UP,): 4,
        (locals.K_DOWN,): 5
    }

    @staticmethod
    def get_action_size(env_name):
        return len(LabEnvironment.ACTION_LIST)

    @staticmethod
    def get_observation_size(env_name):
        return [3, 84, 84]  # todo: 结果需压迫更加泛化!

    def __init__(self,
                 env_name,
                 human_control=False,
                 seed=None,
                 frame_skip=4,
                 state_is_rgb=True,  # 默认值仍然是 RGB 的 state
                 enable_debug_observation=False):

        self.frame_skip = frame_skip
        self.human_control = human_control
        basic_obs = [
            'RGB_INTERLEAVED'
        ]  # 基本的观察输出, RGBD, 相对的速度, 相对的角速度

        config = {
            'fps': str(60),
            'width': str(84),
            'height': str(84),
        }
        if enable_debug_observation:
            basic_obs.append('DEBUG.CAMERA.TOP_DOWN')
            config.update(maxAltCameraWidth=str(160), maxAltCameraHeight=str(160))
        # del basic_obs[0]  # todo 可能会有问题，删除 RGB

        self.env = deepmind_lab.Lab(env_name, basic_obs, config=config)

        self.enable_debug_observation = enable_debug_observation

        if seed == None:
            self.seed = np.random.randint(0, np.power(2, 32) - 1)
        else:
            self.seed = seed

        self.reset()

        print('lab started')

    def _process_frame(self, frame):
        frame = frame.astype(np.float32)
        frame = frame / 255.0
        frame = np.transpose(frame, [2, 0, 1])
        return frame

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

    def process_top_down(self, top_down):
        """
        奇怪了。 deepin 下需要转换；ubuntu 不需要
        :param top_down:
        :return:
        """
        if top_down.shape[0] == 3:
            top_down = np.transpose(top_down, [1, 2, 0])  # CHW TO HWC
            top_down = np.ascontiguousarray(top_down)  # C
        return top_down

    def reset(self):
        self.env.reset()
        obs = self.env.observations()
        rgb = obs['RGB_INTERLEAVED']
        # 处理
        self.last_action = 0
        self.last_reward = 0
        self.last_state = self._process_frame(rgb)

        if self.enable_debug_observation:
            self.last_debug_top_down_state = self.process_top_down(obs['DEBUG.CAMERA.TOP_DOWN'])

    def step(self, action):
        """
        进行一个动作
        :param action:
        :return: state, 回报, 是否终止, pixel_change. 如过开启深度预测 + depth-map; 最后为 debug 的 top-down 视图
        """

        if action < 0:
            real_action = self.DEFAULT_ACTION
        else:
            real_action = self.ACTION_LIST[action]

        reward = self.env.step(real_action, num_steps=self.frame_skip)
        terminated = not self.env.is_running()
        obs = None if terminated else self.env.observations()

        # step 返回的信息
        if terminated:
            state = self.last_state
        else:
            rgb = obs['RGB_INTERLEAVED']
            state = self._process_frame(rgb)

        pixel_change = self._calc_pixel_change(state, self.last_state)
        self.last_state = state
        self.last_action = action
        self.last_reward = reward

        returned = [state, reward, terminated, pixel_change]

        # 其他的 observation
        if self.enable_debug_observation:
            top_down = None
            if terminated:
                top_down = self.last_debug_top_down_state
            else:
                top_down = self.process_top_down(obs['DEBUG.CAMERA.TOP_DOWN'])
            self.last_debug_top_down_state = top_down

            returned.append(top_down)
        return returned

    def close(self):
        self.env.close()
        print("lab environment stopped")

    def key_to_action(self):
        return self.KEY_TO_ACTION


if __name__ == "__main__":
    # human control
    pass
