import math
import random
import numpy as np
from collections    import namedtuple, deque
from itertools      import count

from torch.cuda import memory
from envs.QueueEnv  import QueueEnv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


env = QueueEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
                workloads_dir = "/data/workloads/test",
                t_action = 10,
                queue_max_len = 20,
                t_shutdown = 500,
                hosts_per_server = 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

MAX_WALL = 2000

class QAgent:
    def __init__(self, n_observations: int, n_actions: int) -> None:
        self.n_observations = n_observations
        self.n_actions = n_actions

        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.episode_durations = []

    def act(self, obs):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)

        if obs.shape[0] == 0:
            return torch.tensor([0], device=device, dtype=torch.long)

        if np.random.uniform(0, 1) <= eps_threshold:
            idx  = np.random.choice(obs.shape[0])
            res = obs[idx][0].unsqueeze(0)
            res += 1

            return res


        with torch.no_grad():
            scores, _  = self.policy_net(obs).max(1)
            max_row = scores.argmax().item()

        res = obs[max_row][0].unsqueeze(0)
        res += 1
        return res

    def _process_state(self, state):
        platform = state["platform"]
        queue = state["queue"]

        # Get valid jobs for this step.
        nb_available = len(platform["agenda"]) - sum(1 for i in platform["agenda"] if i[1] != 0)
        job_pos = [ i for i, j in enumerate(queue["jobs"]) if 0 < j[1] <= nb_available ]

        ## Platform info
        # 1. Resource state
        states = np.zeros(5)
        for host in platform['status']:
            states[int(host) - 1] += 1
        states /= platform['agenda'].shape[0]

        ## Job specific info
        # TODO - Refactore ones ready
        job_cant  = len(job_pos)
        job_state = np.zeros( (job_cant, 4) )

        if job_cant == 0:
            return torch.tensor([], device=device, dtype=torch.float32)

        res = torch.zeros(job_cant, 9)
        for i, j in enumerate(job_pos):
            res[i, 0] = j
            res[i, 1] = state["current_time"] - queue["jobs"][j][0]
            res[i, 2] = queue["jobs"][j][1] /platform["nb_hosts"]
            # TODO - Change 3 for Estim. vs Wall
            res[i, 3] = queue["jobs"][j][2] /MAX_WALL
            res[i, 4:] = torch.from_numpy(states)

        return res


    def _optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s.shape[0] != 0,
                                                    batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = [ s for s in batch.next_state if s.shape[0] != 0 ]

        state_batch = batch.state
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = [ self.policy_net(i) for i in state_batch ]
        state_action_values = torch.cat([ state_action_values[i.max(1)[1][0]]
                                            for i in state_action_values ], dim=0)
        state_action_values = state_action_values.gather(1,
                                            action_batch.unsqueeze(1).type(torch.int64))


        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        '''
        with torch.no_grad():
            next_state_values[non_final_mask] = torch.cat([ self.target_net(i).max(1)[0]
                                                for i in non_final_next_states ], dim=0)
        '''

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, env):
        #num_episodes = 600 if torch.cuda.is_available() else 50
        num_episodes = 1
        print("[START]")

        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            state = env.reset()

            state = self._process_state(state)

            for t in count():
                action = self.act(state)

                obs, reward, done, _ = env.step( action.item() )
                obs = self._process_state(obs)

                # Store the transition in memory
                reward = torch.tensor([reward], device=device, dtype=torch.float32)

                if action.item() != 0:
                    self.memory.push(state, action, obs, reward)

                # Move to the next state
                #state = next_state

                state = obs

                # Perform one step of the optimization (on the policy network)
                self._optimize_model()

                # Soft update of the target network's weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                            target_net_state_dict[key] * (1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    # plot_durations()
                    break

        print("Complete")
        # plot_durations(show_result=True)
        # plt.ioff()
        # plt.show()

    def play(self, env, verbose=True) -> None:

        eps_start = 1.0
        eps_end   = 0.1
        eps_decay = 0.996

        history = { "score": 0, "steps": 0, "info": None }
        obs, done, info = env.reset(), False, {}
        obs = self._process_state(obs)

        while not done:
            obs, reward, done, info = env.step( self.act(obs) )
            obs = self._process_state(obs)

            history['score'] += reward
            history['steps'] += 1
            history['info']   = info

            #print(f"STEP {history['steps']}")

            eps_start = max(eps_start * eps_decay, eps_end)

        if verbose:
            print(f"[DONE] Score: {history['score']} - Steps: {history['steps']}")


if __name__ == "__main__":
    env = QueueEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
                workloads_dir = "/data/workloads/test",
                t_action = 10,
                queue_max_len = 20,
                t_shutdown = 500,
                hosts_per_server = 1)

    n_actions = env.action_space.n
    n_observations = 9

    agent = QAgent(n_observations, n_actions)
    agent.train(env)

    import batsim_py
    env = QueueEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
                workloads_dir = "/data/workloads/test",
                t_action = 10,
                queue_max_len = 20,
                t_shutdown = 500,
                hosts_per_server = 1)

    jobs_mon = batsim_py.monitors.JobMonitor(env.simulator)
    sim_mon  = batsim_py.monitors.SimulationMonitor(env.simulator)
    schedule_mon  = batsim_py.monitors.SchedulerMonitor(env.simulator)

    agent.play(env)

    jobs_mon.to_csv("/data/expe-out/jobs-DQN.out")
    sim_mon.to_csv("/data/expe-out/sim-DQN.out")
    schedule_mon.to_csv("/data/expe-out/schedule-DQN.out")

