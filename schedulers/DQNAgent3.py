import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.decomposition import PCA

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
TAU = 0.005
LR = 1e-4

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps_done = 0

    def act(self, obs) -> int:
        platform = obs['platform']
        queue    = obs["queue"]
        nb_available = len(platform["agenda"]) - sum(1 for j in platform["agenda"] if j[1] != 0)

        # If no tasks in queue, we can't choose anything.
        if queue["size"] == 0:
            return 0

        self.steps_done += 1
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        if sample > eps_threshold:
            # EXPLOTATION

            ## Queue Data
            queue_len  = queue["size"]
            queue_wait = obs["current_time"] - queue["jobs"][:,0]
            queue_res  = queue["jobs"][:,1]
            queue_wall = queue["jobs"][:,2]
            queue_data = np.concatenate([ [queue_len],queue_wait,queue_res,queue_wall ])

            ## Platform Data
            host_status = np.zeros(5)
            for i in platform["status"]:
                host_status[i-1] += 1
            host_status /= platform["agenda"].shape[0]

            host_remaining_time = obs["current_time"] - (platform["agenda"][:,0] + platform["agenda"][:,1])
            host_data  = np.concatenate([ host_status, host_remaining_time ]) 

            all_data   = np.concatenate([queue_data, host_data])
            all_tensor = torch.tensor(all_data, device=self.device, dtype=torch.float)

            with torch.no_grad():
                results = self.policy_net(all_tensor)
                valid_results = [ j if i < queue_len else -1 * float('inf') for i,j in enumerate(results.tolist())]

                chosen_job = valid_results.index(max(valid_results))
                print(f"A{chosen_job}")
                return chosen_job + 1
        else:
            # EXPLORATION

            #jobs = [i for i, j in enumerate(queue["jobs"]) if 0 < j[1] <= nb_available]
            #job_pos = -1 if len(jobs) == 0 else random.choice(jobs)
            job_pos = next((i for i, j in enumerate(queue["jobs"]) if 0 < j[1] <= nb_available), -1)
            
            print(f"B{job_pos}")
            return job_pos + 1

    def play(self, env, verbose=True) -> None:
        history = { 'score': 0, 'steps': 0, 'info': None }
        obs, done, info = env.reset(), False, {}

        # Space size
        queue_size    = obs["queue"]["jobs"].shape[0] * 3 + 1
        platform_size = 5 + obs["platform"]["agenda"].shape[0]

        self.policy_net = DQN(queue_size + platform_size, env.action_space.n).to(self.device)

        while not done:
            obs, reward, done, info = env.step(self.act(obs))
            history['score'] += reward
            history['steps'] += 1
            history['info'] = info

            if history["score"] < -3000:
                print(f"[ERROR] Simulation stopped.")

        if verbose:
            print(f"[DONE] Score: {history['score']} - Steps: {history['steps']}")