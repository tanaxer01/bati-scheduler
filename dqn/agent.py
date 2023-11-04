
import math
import random
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from .model import DQN
from .replay_memory import ReplayMemory, Transition

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

NUM_EPISODES = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size=1):
        self.state_size = state_size
        self.action_size = action_size

        # Q Network
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        # Replay Memory
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.episode_durations = []


    def act(self, obs):

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                    math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if random.random() <= eps_threshold:
            # TODO - Random action
            pass

        # TODO - Explotation
        return torch.zeros(2)


    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # TODO - Non final ???

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        # TODO - FIX
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)


        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            # TODO - FIX
            next_state_values = self.target_net(next_state_batch).max(1)[0]

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
        num_episodes = NUM_EPISODES if torch.cuda.is_available() else 1

        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            state = self._process_obs(env.reset())

            for t in count():
                print(f"-> {env.simulator.current_time} {t}")
                action = self.act(state)
                obs, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)

                next_state = self._process_obs(obs)

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
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                            target_net_state_dict[key] * (1 -TAU)
                    self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    # self.plot_durations()
                    break

                print("REMOVE RETURN")
                return



        print("[Training Complete]")
        # self.plot_durations(show_result=True)
        # plt.ioff()
        # plt.show()

    def _process_obs(self, obs):
        queue = obs["queue"]


        print(obs)
        res = torch.zeros(len(queue["jobs"]), 3)



        res = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return res




