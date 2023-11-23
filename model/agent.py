import math
import random
import numpy as np
import datetime
from pathlib import Path
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from .network import DQN
from .metrics import MetricLogger
from .replay_memory import ReplayMemory, Transition

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size):
        self.state_size  = state_size
        self.action_size = 1

        # Q Network
        self.policy_net = DQN(state_size, 1)
        self.target_net = DQN(state_size, 1)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        # Replay Memory
        self.memory = ReplayMemory(20000)

        self.steps_done = 0

    def act(self, state) -> int:
        """Given a state, choose an epsilon-greedy action"""
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)

        # EXPLORE
        if random.random() < eps_threshold:
            action_idx = 0 if state.shape[0] == 1 else  random.randint(0, state.shape[0]-1)
        # EXPLOIT
        else:
            with torch.no_grad():
                action_idx = self.policy_net(state).max(1).indices.view(1,1).item()

        # increment step
        self.steps_done += 1
        return action_idx

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return None, None

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = [s for s in batch.next_state if s is not None]
        state_batch  = batch.state
        action_batch = torch.cat(batch.action).unsqueeze(0)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        #state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        state_action_values = torch.cat([ self.policy_net(i).max(1)[0]
                                            for i in state_batch ]).unsqueeze(0)
        state_action_values = state_action_values.gather(1, action_batch.type(torch.int64))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
            next_state_values[non_final_mask] = torch.tensor([ self.policy_net(i).max(1).values for i in non_final_next_states ])

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values.T, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

         # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return (loss.item(), expected_state_action_values.mean().item())

    def train(self, env):
        #num_episodes = NUM_EPISODES if torch.cuda.is_available() else 1
        num_episodes = 1

        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            state = env.reset()
            state = self._process_obs(state)

            for t in count():
                action = self.act(state)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
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

        print("[Training complete]")

    def play(self, env, save=False):
        use_cuda = torch.cuda.is_available()
        print(f"Using CUDA: {use_cuda}", end="\n\n")

        logger = None
        if save:
            save_dir = Path("/data/expe-out/checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            save_dir.mkdir(parents=True)

            logger = MetricLogger(save_dir)

        #episodes = 40
        episodes = 10
        for e in range(episodes):
            state, _ = env.reset()
            state = self._process_obs(state)

            # Play the game!
            while True:
                assert state.size(0) != 0, f"+ {len(env.simulator.queue)} {state.shape}"

                # Run agent on the state
                action = self.act(state)

                # Agent perform action
                next_state, reward, done, trunc, info = env.step(action)
                next_state = self._process_obs(next_state)

                # Remember
                action = torch.tensor([action], device=device, dtype=torch.float32)
                reward = torch.tensor([reward], device=device, dtype=torch.float32)

                if action == 0:
                    self.memory.push(state, action, None, reward)
                else:
                    self.memory.push(state, action, next_state, reward)

                # Learn
                loss, q = self.optimize_model()

                # Logging
                if save and logger:
                    logger.log_step(reward.item(), loss, q)

                # Update state
                state = next_state

                # Update state
                if done:
                    break

            if save and logger:
                logger.log_episode()

                #if (e % 20 == 0) or (e == episodes - 1):
                eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
                logger.record(episode=e, epsilon=eps, step=self.steps_done)

    def _process_obs(self, obs):
        queue = obs["queue"]
        platform = obs["platform"]

        # State matrix
        state = torch.zeros(queue["jobs"].shape[0], 8)
        for i, (sub, res, wall, flops) in enumerate(queue["jobs"]):
            last_alloc = platform["hosts"][int(res)-1,1]

            # Task stuff
            ## Waiting time
            state[i, 0] = obs["current_time"] - sub
            ## Resources
            state[i, 1] = res / platform["nb_hosts"]
            ## Walltime
            state[i, 2] =  wall
            ## Flops
            state[i, 3] =  flops

            candidates = np.delete(queue["jobs"], i-1, 0)
            candidates = candidates[candidates[:, 1] <= res]

            # Queue stuff
            ## Queue length
            # state[i, 4] = candidates.size
            ## Mean Waiting time
            state[i, 5] = candidates[:,0].mean() if candidates.shape[0] != 0 else 0.
            ## Mean Resources needed
            #state[i, 4] = len(candidates)/ platform["nb_hosts"]
            state[i, 6] = candidates[:,1].mean() / platform["nb_hosts"] if candidates.shape[0] != 0 else 0.
            ## Mean Walltime
            #state[i, 7] = candidates[:,2].mean() / platform["hosts"][:,1].mean() if candidates.shape[0] != 0 else 0.
            state[i, 7] = candidates[:,2].mean() if candidates.shape[0] != 0 else 0.

        return state
