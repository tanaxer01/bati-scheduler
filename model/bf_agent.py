import math
import random
from re import L
from typing import Optional
import numpy as np
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .network import DQN, DDQN

from .agent import Agent
from .metrics import MetricLogger, MonitorsInterface
from .replay_memory import ReplayMemory, Transition


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BFAgent(Agent):
    def _process_obs(self, obs):

        '''
        for i, (wait, res, wall, flops, deps) in enumerate(obs["jobs"]):
            ## Waiting time
            state[i, 0] = wait
            ## Resources
            state[i, 1] = res
            ## Walltime
            state[i, 2] = wall
            ## Flops
            state[i, 3] = flops
            ## Dependencies
            state[i, 4] = deps
            ## Speed
            #state[i, 5] = speed
        '''

        return torch.from_numpy(obs["jobs"]).type(torch.float32)

