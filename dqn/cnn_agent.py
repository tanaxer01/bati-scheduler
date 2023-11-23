import math
import random
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from .model import ConvDQN
from .replay_memory import ReplayMemory, Transition

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
# BATCH_SIZE = 128
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CnnAgent():
    def __init__(self, state_size, action_size) -> None:
        self.state_size = state_size
        self.action_size = action_size

        # Q Networks
        self.policy_net = ConvDQN(state_size, action_size)
        self.target_net = ConvDQN(state_size, action_size)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        # Replay Memory
        self.memory = ReplayMemory(20000)
        self.steps_done = 0

    def act(self, obs):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        nb_jobs = sum( 1 for i in obs[1,:] if i != -1 )

        # Exploration
        if random.random() <= eps_threshold:
            res = random.randint(0,nb_jobs)
            print("AAAA", nb_jobs)
            print(obs[1,:])
            return torch.tensor([res], device=device, dtype=torch.float32).unsqueeze(0)

        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(1)
        print("BBBB")
        # Explotation
        with torch.no_grad():
            action = self.policy_net(state)
            print(action.shape)
            action = action[:nb_jobs:,1].max(1)[1].view(1,1)

            #.max(1)[1].view(1, 1)

        return action - 1

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
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
        num_episodes = 1

        for i_episode in range(num_episodes):
            state = env.reset()
            state = self._process_obs(state)

            for t in count():
                action = self.act(state)
                obs, reward, terminated, truncated, info = env.step(action)
                reward = torch.tensor([reward], device=device)
                done = truncated or terminated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(self._process_obs(obs))

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    break
        print("[ Training complete ]")


    def _process_obs(self, obs):
        queue = obs["queue"]
        platform = obs["platform"]

        # State matrix
        state = np.full( (self.action_size, 6), -1 )
        for i, (sub, res, wall) in enumerate(queue["jobs"]):
            # Task stuff
            ## Waiting time
            state[i, 0] = obs["current_time"] - sub
            ## Resources
            state[i, 1] = res
            #state[i, 1] = res / platform["nb_hosts"]
            ## Walltime
            state[i, 2] = wall
            ## Queue len
            state[i, 3] = 0
            state[i, 4] = 0
            state[i, 5] = 0


            candidates = np.delete(queue["jobs"], i-1, 0)
            candidates = candidates[candidates[:, 1] <= res]

            # Queue stuff
            ## Waiting time
            state[i, 3] = candidates[:,0].mean() if candidates.shape[0] != 0 else 0.
            ## Resources
            #state[i, 4] = len(candidates)/ platform["nb_hosts"]
            ## Resources
            state[i, 4] = candidates[:,1].mean() if candidates.shape[0] != 0 else 0.
            ## Walltime
            state[i, 5] = candidates[:,2].mean() if candidates.shape[0] != 0 else 0.

            assert state.T.shape == (6,20)

        return state.T
