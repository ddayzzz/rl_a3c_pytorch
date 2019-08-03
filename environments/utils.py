# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import numpy as np
from enum import Enum


class LabWorkCommand(Enum):

    RESET = 0
    ACTION = 1
    TERMINATE = 2


def make_environment(env_type, env_name, channel_first=False, **kwargs):
    if env_type == 'lab':
        from environments.lab_environment import DeepmindLabEnvironment, DeepmindLabEnvironmentForUnreal
        env = DeepmindLabEnvironment(env_name=env_name, **kwargs)
        env = DeepmindLabEnvironmentForUnreal(env=env, channel_first=channel_first, enable_pixel_change=True)
        return env
    elif env_type == 'gym':
        from environments.gym_environment import GymEnvironment
        return GymEnvironment(env_name=env_name, **kwargs)
    raise ValueError('environment args')

def get_maze_info(env_type, env_name):
    if env_type == 'lab':
        from environments.lab_environment import DeepmindLabEnvironment
        return DeepmindLabEnvironment.get_maze_info(env_name)
    else:
        raise ValueError('not found!')




