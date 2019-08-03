from models.unreal import UNREAL
from environments.utils import make_environment
from torch.autograd import Variable
import torch


if __name__ == '__main__':
    import numpy as np
    env = make_environment(env_type='lab', env_name='nav_maze_static_01', channel_first=True)
    model = UNREAL(3, action_size=6)

    env.reset()
    #
    cx = Variable(torch.zeros(1, 1, 256))  # layers, batch_size, output
    hx = Variable(torch.zeros(1, 1, 256))
    observation, reward, terminal, pixel_change = env.step(0)
    observation = observation['rgb']
    # observation = np.transpose(observation, [2, 0, 1])
    ls = torch.from_numpy(np.array([[0,0,0,0,0,0,0], [1,1,1,1,1,1,0]], dtype=np.float32))
    v, pi, (hx, cx) = model(task_type='a3c', states=torch.from_numpy(np.array([observation, observation.copy()])), hx=hx, cx=cx, last_action_reward=ls)
    #
    # pi, v, (hx, cx) = model(task_type='a3c', states=torch.from_numpy(np.zeros([3, 3, 84, 84], dtype=np.float32)), cx=cx, hx=hx)

    print('v={0}'.format(v))
    print('pi={0}'.format(pi))
    print('hx={0}'.format(hx))
    print('cx={0}'.format(cx))
    # 可视化
    # from tensorboardX import SummaryWriter
    # with SummaryWriter('/tmp/unreal') as w:
    #     w.add_graph(model, input_to_model=['rp', torch.from_numpy(np.zeros([3, 3, 84, 84], dtype=np.float32))])
    env.close()


