import random
import math
import numpy as np

from typing import Optional
from collections import namedtuple, deque

from .FreeSpaces import FreeSpaceContainer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from itertools import combinations

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
TAU = 0.005
LR = 1e-4

def action_to_num(machines: list[int], action: list[int]) -> int:
    all_actions = sum([ list(combinations(machines, i))  for i in range(1, len(machines)+1) ], [])
    action_dict = { j: i for i, j in enumerate([()] + all_actions) }

    if action in all_actions:
        return action_dict[action]
    return -1

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps_done = 0
        self.listFreeSpace : Optional[FreeSpaceContainer] = None

    def _get_free_space_list(self, total_cores: Optional[int] = None, max_time: float = 20.0):
        if self.listFreeSpace is None:
            assert total_cores is not None
            self.listFreeSpace = FreeSpaceContainer(total_cores, max_time)

        return self.listFreeSpace

    def act(self, obs) -> int:
        # 1.  Are there any tasks to alloc in listFreeSpace ?
        queue = obs['queue']

        # 2.  Is queue len > 0 and not listFreeSpace.full
        # 2a. No jobs in queue
        if queue["size"] == 0:
            return 0

        # 2b. No FreeSpaces available
        job = queue["jobs"][0]
        posible_actions = self._get_free_space_list.get_spaces(job[3], job[2])
        posible_actions = [ action_to_num(obs["platform"]["ids"], i) for i in posible_actions ]
        print(posible_actions)

        if len(posible_actions) == 0:
            return 0

        # 2c. Add queue[0] to listFreeSpace
        #     Return the "best" space for the task
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                queue_data    = np.concatenate([obs["queue"]["jobs"].ravel(), [obs["queue"]["size"]]])
                platform_data = np.concatenate([obs["platform"]["status"].ravel(),obs["platform"]["agenda"].ravel()])
                state_data = np.concatenate([queue_data, platform_data,  [obs["current_time"]]])
                size = obs["queue"]["size"]

                obs_arr = torch.tensor(state_data, device=self.device, dtype=torch.float)
                results = self.policy_net(obs_arr).tolist()
                masked_results = [ i if j in posible_actions else -1*float('inf') for i, j in enumerate(results) ]

                return masked_results.index(max(masked_results))
        else:
            #agenda = obs['platform']['agenda']
            #queue = obs['queue']['jobs']

            #nb_available = len(agenda) - sum(1 for j in agenda if j[1] != 0)

            #jobs =  [i for i, j in enumerate(queue) if 0 < j[1] <= nb_available]

            #job_pos = next((i for i, j in enumerate(queue) if 0 < j[1] <= nb_available), -1)
            #job_pos = -1 if len(jobs) == 0 else random.choice(jobs)

            return random.choice(posible_actions)
            return int(torch.tensor([[job_pos + 1]], device=self.device, dtype=torch.long).item())

    def play(self, env, verbose=True) -> None:
        history = { 'score': 0, 'steps': 0, 'info': None }
        obs, done, info = env.reset(), False, {}

        # queue
        queue_s = obs["queue"]["jobs"].ravel().shape[0] + 1
        print(obs["queue"].keys(), queue_s)

        platform_s = obs["platform"]["status"].ravel().shape[0] + obs["platform"]["agenda"].ravel().shape[0]
        print(obs["platform"].keys(), platform_s)

        state_s = queue_s + platform_s + 1
        obs_s   = 2**len(obs["platform"]["status"])
        print(">>", obs_s)
        print(">>", state_s, env.action_space.n)

        self.policy_net = DQN(state_s, obs_s).to(self.device)
        #self.target_net = DQN(state_s, env.action_space.n).to(self.device)
        #self.target_net.load_state_dict(self.policy_net.state_dict())

        while not done:
            obs, reward, done, info = env.step(self.act(obs))
            history['score'] += reward
            history['steps'] += 1
            history['info'] = info

            if history["score"] < -3000:
                print(f"[ERROR] Simulation stopped.")

        if verbose:
            print(f"[DONE] Score: {history['score']} - Steps: {history['steps']}")



