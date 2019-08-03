import numpy as np
import cv2
from itertools import count
from environments.utils import make_environment
from utils import make_env


def deepmind_lab(level):
    env = make_environment(env_type='lab', env_name=level, channel_first=False)
    try:
        while True:
            env.reset()
            episodic_reward = 0.0
            time = 0
            for t in count():
                env.render(mode='human')
                action = np.random.choice(env.action_space.n)
                obs, reward, terminal, pc = env.step(action)
                episodic_reward += reward
                time = t
                if terminal:
                    break
            print('time: {}, reward: {}'.format(time, episodic_reward))
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def gym_env(level):
    env = make_env(env_type='gym', env_name=level, args=dict())
    try:
        while True:
            env.reset()
            episodic_reward = 0.0
            time = 0
            for t in count():
                env.render(mode='human')
                action = np.random.choice(env.action_space.n)
                obs, reward, terminal, pc = env.step(action)
                episodic_reward += reward
                time = t
                if terminal:
                    break
            print('time: {}, reward: {}'.format(time, episodic_reward))
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == '__main__':
    deepmind_lab(level='nav_maze_static_03')
    # gym_env(level='Pong-v0')