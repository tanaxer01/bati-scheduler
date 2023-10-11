import random
import math 
import numpy as np

from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
TAU = 0.005
LR = 1e-4

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

    def act(self, obs) -> int:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)

        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                queue_data = np.concatenate([obs["queue"]["jobs"].ravel(), [obs["queue"]["size"]]])
                platform_data = np.concatenate([obs["platform"]["status"].ravel(),obs["platform"]["agenda"].ravel()])

                state_data = np.concatenate([queue_data, platform_data,  [obs["current_time"]]])
                size = obs["queue"]["size"]
                if size == 0:
                    return 0

                obs_queue = torch.tensor(state_data, device=self.device, dtype=torch.float)

                results = self.policy_net(obs_queue)
                results = [ j if i <= obs["queue"]["size"] else -1*float('inf') for i, j in enumerate(results.tolist()) ]
                res = results.index(max(results))

                return res

                #return int(self.policy_net(obs_queue).max().item())
                return self.policy_net(obs_queue).max(1)[1].view(1, 1)
        else:
            agenda = obs['platform']['agenda']
            queue = obs['queue']['jobs']
            nb_available = len(agenda) - sum(1 for j in agenda if j[1] != 0)

            jobs =  [i for i, j in enumerate(queue) if 0 < j[1] <= nb_available]

            job_pos = next((i for i, j in enumerate(queue) if 0 < j[1] <= nb_available), -1)
            job_pos = -1 if len(jobs) == 0 else random.choice(jobs)

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
        print(">>", state_s)

        self.policy_net = DQN(state_s, env.action_space.n).to(self.device)
        self.target_net = DQN(state_s, env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        while not done:
            obs, reward, done, info = env.step(self.act(obs))
            history['score'] += reward
            history['steps'] += 1
            history['info'] = info

            if history["score"] < -3000:
                print(f"[ERROR] Simulation stopped.")

        if verbose:
            print(f"[DONE] Score: {history['score']} - Steps: {history['steps']}")
