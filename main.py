from __future__ import print_function, division
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp


from train import Trainer
from shared_optim import SharedRMSprop, SharedAdam
#from gym.configuration import undo_logger_setup

from models.unreal import UNREAL
import time
import signal

# from torch.utils.tensorboard import SummaryWriter

# torch.backends.cudnn.benchmark = True
from tensorboardX import SummaryWriter

#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=32,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')

parser.add_argument(
    '--env_type',
    default='lab',
    metavar='ENV',
    help='type of environment to train on (default: lab)')

parser.add_argument(
    '--env_name',
    default='nav_maze_static_01',
    metavar='ENV',
    help='environment to train on (default: nav_maze_static_01)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='RMSProp',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

def log_uniform(lo, hi, rate):
    import math
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)

class Application(object):

    def __init__(self, args):
        self.args = args
        self.trainers = []
        self.global_t = 0
        # summary write

    def train_function(self, parallel_index, init_env, enable_summary_writer=None):
        """

        :param parallel_index:
        :param init_env: 主要在保存模型之后是的进程继续执行
        :param enable_summary_writer: 由于只能同一个process共享数据, 所以定义 writer 在 process 中
        :return:
        """

        trainer = self.trainers[parallel_index]
        # 设置时间
        # set start_time
        trainer.set_start_time(self.start_time)

        if enable_summary_writer:
            summary_writer = SummaryWriter(log_dir='tmp/unreal_log')
        else:
            summary_writer = None
        if init_env:
            trainer.init_env()  # 必须这样实现. 否则 deepmind  lab 无法启动!
        while True:
            if self.stop_requested:
                break
            if self.terminate_reqested:
                trainer.close()
                break
            if self.global_t > 10 * 10 ** 7:  # max_time_step
                trainer.close()
                break
            # if parallel_index == 0 and self.global_t > self.next_save_steps:
            #     # Save checkpoint
            #     self.save()

            diff_global_t = trainer.train(self.global_t, summary_writer)
            # if parallel_index == 0 and tags:
            #     for tag, value in tags.items():
            #         # self.summary_writer.add_scalar(tag=tag, scalar_value=value, global_step=self.global_t)
            #         pass
            self.global_t += diff_global_t

    def run(self):

        torch.manual_seed(args.seed)
        if args.gpu_ids == -1:
            args.gpu_ids = [-1]
        else:
            torch.cuda.manual_seed(args.seed)
            mp.set_start_method('spawn')

        # env = make_env(env_type=args.env_type, env_name=args.env_name, args=args)
        shared_model = UNREAL(in_channels=3, action_size=6, enable_pixel_control=True)

        if args.load:
            saved_state = torch.load(
                '{0}{1}.dat'.format(args.load_model_dir, args.env),
                map_location=lambda storage, loc: storage)
            shared_model.load_state_dict(saved_state)
        shared_model.share_memory()

        lr = log_uniform(1e-4, 5e-3, 0.5)

        if args.shared_optimizer:
            if args.optimizer == 'RMSprop':
                optimizer = SharedRMSprop(shared_model.parameters(), lr=lr, eps=0.1)
            if args.optimizer == 'Adam':
                optimizer = SharedAdam(
                    shared_model.parameters(), lr=lr, amsgrad=args.amsgrad)
            optimizer.share_memory()
        else:
            optimizer = None



        # p = mp.Process(target=train, args=(args, shared_model, env_conf))
        # p.start()
        # processes.append(p)
        # time.sleep(0.1)

        self.stop_requested = False
        self.terminate_reqested = False

        for rank in range(0, args.workers):
            trainer = Trainer(rank, args, shared_model=shared_model, optimizer=optimizer, lr=lr)
            self.trainers.append(trainer)

            # time.sleep(0.1)
        # 设置运行起始的时间
        # set start time
        self.start_time = time.time() - 0  # wall_t

        processes = []
        for rank in range(0, args.workers):
            if rank == 0:
                p = mp.Process(target=self.train_function, args=(rank, True, True))
            else:
                p = mp.Process(target=self.train_function, args=(rank, True))
            p.start()
            processes.append(p)

        # 注册终止信号
        signal.signal(signal.SIGINT, self.signal_handler)


        print('Press Ctrl+C to stop')
        for rank in range(0, args.workers):
            time.sleep(0.01)
            processes[rank].join()



    def save(self):
        pass

    def signal_handler(self, signal, frame):
        # self.summary_writer.close()
        print('You pressed Ctrl+C!')
        self.terminate_reqested = True


if __name__ == '__main__':
    args = parser.parse_args()
    app = Application(args=args)
    app.run()
