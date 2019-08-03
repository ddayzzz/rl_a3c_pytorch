# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import Pipe, Process
from environments.utils import LabWorkCommand
from gym import spaces
import gym


try:
    import deepmind_lab
    from pygame import locals
except ImportError:
    pass

import numpy as np
def _action(*entries):
    return np.array(entries, dtype=np.intc)
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

# todo 布局信息来自于 deepmind 自带的脚本. 一个方法是把 level 拷贝到本地目录, 修改 debug_observation, 使其包含 layout 信息
MAZE_INFO = {
        'nav_maze_random_goal_02': {
            'maze': '******************************** *               *       *   ** * *** *********** *** * * * **   *   *             * *   * ** *** *** ************* ***** **   *   * *         *   *     **** *** * *         * ***** * **     *   *         * *     * ** *** *****         * *     **** *     *           * *     * ** *     * *         * *     * **       * *           *     * ** *     * *********   *     * ** *     *       *     *     * ** ***** ******* * *** ******* **   *         * * *     *     **** *** ***** * * ******* ******       *       *             ********************************'
            , 'height': 19
            , 'width': 31},
        'nav_maze_random_goal_01':{
            'maze': '**********************   *   *           ** * * ***** ***** * ** * * *   *     * * ** * * *   *     * * ** *   *   *     * * ** *****   * *   * * **     *   * *   * * ****** * *********** **                   **********************',
            'width': 21,
            'height': 11
        },
        'nav_maze_random_goal_03':{
            'maze': '******************************************                                       ** ***************** *********** ******* ** *   *           * *   *             * ** *** *           * * * *             * **   *             * * * *             * **** * *           * *** *             * ** *   *           *   * *             * ** *** *           *** * *             * **   * *           *   * *             * ** *** *           * *** *             * **     *           *     *             * ** ********************* *************** ** *         *         *       *         ** * * *************** * ******* * ********   * *             * *       * * *     ****** *             * *       * *** *** ** *   *             * *       *     *   ** * ***             * *       * ***** * **   * *             * *       * *   * * ** *** *             * *       * * * * * **   *               * *       *   * * * **** * *************** *       ***** * * **   *   *     *     * *       *   * * * ** *** * *** * *** * * * ******* * *** * **     *     *     * *           *     * ******************************************',
            'width': 41,
            'height': 27
        }
}


# 环境在另外的进程上运行, 经过验证, 可以提高CPU的利用率
def LabWorkProcess(conn, env_name, frame_skip, enable_vel=False, enable_debug_observation=False,
                   enable_encode_depth=False, maxAltCameraWidth=160, maxAltCameraHeight=160):
    # lab
    basic_obs = [
        'RGB_INTERLEAVED'
    ]  # 基本的观察输出, RGBD

    config = {
        'fps': str(60),
        'width': str(84),
        'height': str(84),

    }
    if enable_debug_observation:
        basic_obs.extend(['DEBUG.CAMERA.TOP_DOWN', 'DEBUG.POS.TRANS', ])
        config.update(maxAltCameraWidth=str(maxAltCameraWidth), maxAltCameraHeight=str(maxAltCameraHeight), hasAltCameras='true')
    if enable_encode_depth:
        basic_obs.append('RGBD_INTERLEAVED')
        # del basic_obs[0]  # todo 可能会有问题，删除 RGB
    if enable_vel:
        basic_obs.extend(['VEL.TRANS', 'VEL.ROT'])  # 速度

    env = deepmind_lab.Lab(env_name,
                           basic_obs,
                           config=config)

    conn.send(0)

    while True:
        # 程序接受消息的循环
        command, arg = conn.recv()

        if command == LabWorkCommand.RESET:
            env.reset()
            obs = env.observations()
            conn.send(obs)
        elif command == LabWorkCommand.ACTION:
            reward = env.step(arg, num_steps=frame_skip)
            terminal = not env.is_running()
            if not terminal:
                obs = env.observations()
            else:
                obs = 0
            conn.send([obs, reward, terminal])
        elif command == LabWorkCommand.TERMINATE:
            break
        else:
            print("bad command: {}".format(command))
    env.close()
    conn.send(0)
    conn.close()



class DeepmindLabEnvironment(gym.Env):

    metadata = {'render.modes': ['rgb_array', 'human']}


    def __init__(self,
                 env_name,
                 frame_skip=4,
                 enable_encode_depth=False,
                 enable_vel=False,
                 enable_debug_observation=False,
                 debug_width=160,
                 debug_height=160,
                 seed=None,
                 **kwargs):
        super(DeepmindLabEnvironment, self).__init__()
        # 相关的属性
        self.env_name = env_name
        self.env_type = 'lab'
        self.viewer = None
        self.enable_debug_observation = enable_debug_observation
        self.enable_vel = enable_vel
        # 编码深度
        self.enable_encode_depth = enable_encode_depth
        # 打开进程和相关的管道
        self.connection, child_connection = Pipe()
        # 配置其他的参数
        self.process = Process(target=LabWorkProcess, args=(
            child_connection, env_name, frame_skip, enable_encode_depth, enable_debug_observation, enable_encode_depth,
            debug_width, debug_height))
        # 初始化 lab 环境
        self.process.start()
        self.connection.recv()

        # 配置相关的输出的 observation
        self.action_space = gym.spaces.Discrete(len(ACTION_LIST))
        obs_spaces = {'rgb': spaces.Box(0, 255, shape=[84, 84, 3], dtype=np.uint8)}

        if enable_debug_observation:
            # top_down = [w, h, 3]
            # word_pos = [x, y, z]: double
            obs_spaces['top_down'] = spaces.Box(0, 255, shape=[debug_width, debug_height, 3], dtype=np.uint8)
            obs_spaces['word_position'] = spaces.Box(low=0.0, high=np.finfo(np.float).max, shape=[1, 3], dtype=np.float)
        if enable_encode_depth:
            obs_spaces['depth'] = spaces.Box(0, 255, shape=[84, 84], dtype=np.uint8)
        if enable_vel:
            obs_spaces['vel'] = spaces.Box(0.0, high=np.finfo(np.float).max, shape=[1, 6], dtype=np.float)

        self.observation_space = spaces.Dict(spaces=obs_spaces)

    def reset(self):
        self.connection.send([LabWorkCommand.RESET, 0])
        obs = self.connection.recv()
        state = self._get_state(obs)
        return state

    def _get_state(self, obs):
        returned = dict(rgb=obs['RGB_INTERLEAVED'])

        if self.enable_debug_observation:
            returned['top_down'] = obs['DEBUG.CAMERA.TOP_DOWN']
            returned['word_position'] = obs['DEBUG.POS.TRANS']

        if self.enable_encode_depth:
            returned['depth'] = obs['RGBD_INTERLEAVED'][:, :, 3]

        if self.enable_vel:
            returned['vel'] = np.concatenate([obs['VEL.TRANS'], obs['VEL.ROT']])
        return returned

    def step(self, action):
        """
        进行一个动作
        :param action:
        :return: state,
        """

        if action < 0:
            real_action = DEFAULT_ACTION
        else:
            real_action = ACTION_LIST[action]

        self.connection.send([LabWorkCommand.ACTION, real_action])
        obs, reward, terminated = self.connection.recv()

        # step 返回的信息
        if terminated:
            state = None
        else:
            state = self._get_state(obs=obs)

        return state, reward, terminated, None

    def close(self):
        self.connection.send([LabWorkCommand.TERMINATE, 0])
        ret = self.connection.recv()
        self.connection.close()
        self.process.join()
        print("lab environment stopped, returned ", ret)


    def get_keys_to_action(self):
        return KEY_TO_ACTION

    @staticmethod
    def get_maze_info(env_name):
        return MAZE_INFO.get(env_name, dict())


class DeepmindLabEnvironmentForUnreal(gym.Wrapper):

    def __init__(self, env: gym.Env, channel_first=False, enable_pixel_change=False):
        super(DeepmindLabEnvironmentForUnreal, self).__init__(env=env)
        # 添加额外的属性
        self.last_state = None  # 上一次的 state 对象
        self.last_action = 0
        self.last_reward = 0
        # 重新定义 observation 空间
        self.channel_first = channel_first
        self.enable_pixel_change = enable_pixel_change
        if channel_first:
            obs_spaces = {'rgb': spaces.Box(0, 1, shape=[3, 84, 84], dtype=np.float32)}
        else:
            obs_spaces = {'rgb': spaces.Box(0, 1, shape=[84, 84, 3], dtype=np.float32)}
        # 重新设置状态空间
        # todo 这里访问了 Wrapper 的 get_attr 方法, gym < 0.13 没有
        if self.env.enable_debug_observation:
            pass  # 这个还不管
        if self.env.enable_encode_depth:
            obs_spaces['depth'] = spaces.Box(0, 1, shape=[84, 84], dtype=np.float32)
        if self.env.enable_vel:
            pass
        # pixel change
        if enable_pixel_change:
            obs_spaces['pixel_change'] = spaces.Box(0, 1, shape=[20, 20], dtype=np.float32)
        self.observation_space = spaces.Dict(spaces=obs_spaces)

    # pixel change
    @staticmethod
    def _subsample(a, average_width):
        s = a.shape
        sh = s[0] // average_width, average_width, s[1] // average_width, average_width
        return a.reshape(sh).mean(-1).mean(1)

    def calc_pixel_change(self, state, last_state):
        if self.channel_first:
            d = np.absolute(state[:, 2:-2, 2:-2] - last_state[:, 2:-2, 2:-2])
            # (3,80,80)
            m = np.mean(d, 0)
        else:
            d = np.absolute(state[2:-2, 2:-2, :] - last_state[2:-2, 2:-2, :])
            # (80,80,3)
            m = np.mean(d, 2)
        c = self._subsample(m, 4)
        return c

    def process_frame(self, frame):
        frame = frame.astype(np.float32)
        frame = frame / 255.
        if self.channel_first:
            frame = np.transpose(frame, [2, 0, 1])
        return frame

    def process_state(self, state, last_frame=None):
        new_state = dict(state)
        new_state['rgb'] = self.process_frame(state['rgb'])

        if self.env.enable_encode_depth:
            new_state['depth'] = self.process_frame(state['depth'])

        if self.enable_pixel_change and last_frame is not None:
            frame = new_state['rgb']
            pc = self.calc_pixel_change(frame, last_frame)
            new_state['pixel_change'] = pc
        # top_down 显示可能有问题
        if self.env.enable_debug_observation:
            top_down = state['top_down']
            if top_down.shape[0] == 3:
                if not self.channel_first:
                    top_down = np.transpose(top_down, [1, 2, 0])  # CHW TO HWC
                    top_down = np.ascontiguousarray(top_down)  # C
            else:
                pass  # todo channelfirst 应该没有需要
            new_state['top_down'] = top_down
        return new_state

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        # 同样需要处理
        self.last_state = self.process_state(state)  # todo: 其实一般不会使用 reset 的 pc
        self.last_action = 0
        self.last_reward = 0
        return state

    def step(self, action):
        # 是否使用了 pc
        if self.enable_pixel_change:
            last_frame = self.last_state['rgb']  # 一次 step 后会覆盖, 所以需要记录一下

        # 运行环境的命令
        state, reward, terminated, info = self.env.step(action)

        if terminated:
            state = self.last_state  # last_state 已经处理过了
        else:
            # 构造新的 state
            state = self.process_state(state=state, last_frame=last_frame)

        self.last_action = action
        self.last_reward = reward
        self.last_state = state
        return state, reward, terminated, info



    def render(self, mode='human', **kwargs):
        """
        由于 依赖 last_state
        :param mode:
        :param kwargs:
        :return:
        """
        rgb = self.last_state['rgb']
        if mode == 'rgb_array':
            return rgb
        elif mode is 'human':
            # pop up a window and render
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow((rgb * 255).astype(np.uint8))
            return self.viewer.isopen
        else:
            super(DeepmindLabEnvironmentForUnreal, self).render(mode=mode)  # just raise an exception



if __name__ == "__main__":
    # human control
    pass
