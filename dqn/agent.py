
import math
import random
import matplotlib.pyplot as plt
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
MAX_WALL = 2000

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

        # No jobs to assign
        if obs.shape[0] == 0:
            print("| 1 -->", 0, " (", eps_threshold,")")
            return torch.tensor([0],
                                device=device, dtype=torch.float32).unsqueeze(0)

        if random.random() <= eps_threshold:
            # TODO - Exploration
            res = random.randint(1,obs.shape[0])
            print(f"| 2 -> {res}/{obs.shape[0]}")
            return torch.tensor([res],
                                device=device, dtype=torch.float32).unsqueeze(0)

        # TODO - DQN -> ConvDQN

        # TODO - Explotation
        action = self._predict_scores(obs)
        print(f"| 3 -> {action}/{obs.shape[0]}")
        return torch.tensor([action - 1],
                                device=device, dtype=torch.float32).unsqueeze(0)


    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title("result")
        else:
            plt.clf()
            plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001) # pause a bit so that plots are updated

        if show_result:
            plt.savefig("/data/expe-out/training.png")


    def _predict_scores(self, states):
        with torch.no_grad():
            scores, _ = self.policy_net(states).max(1)
            max_job   = scores.argmax()

        return max_job + 1


    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            print(f"| nono {len(self.memory)}")
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))


        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                    batch.next_state)), device=device, dtype=torch.bool)
        #non_final_next_states = torch.cat([s for s in batch.next_state
        #                                            if s is not None])

        # TODO - Non final ???
        next_state_batch = batch.next_state

        #state_batch = torch.cat(batch.state)
        state_batch  = batch.state
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        # TODO - FIX
        #state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        state_action_values = [ self.policy_net(i).max().unsqueeze(0) for i in state_batch ]
        state_action_values = torch.concat(state_action_values).unsqueeze(1)
        state_action_values = state_action_values.gather(0, action_batch.type(torch.int64))
        # ----------


        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            # TODO - FIX
            # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
            next_state_values = [ self.policy_net(i).max().unsqueeze(0) for i in state_batch ]
            next_state_values = torch.concat(next_state_values)
            # ----------

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
        #num_episodes = NUM_EPISODES if torch.cuda.is_available() else 1
        num_episodes = 1

        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            state = self._process_obs(env.reset())

            for t in count():
                print(f"|--------{env.simulator.current_time} {t} {state.shape[0]}")
                action = self.act(state)
                obs, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)

                # TODO - Check for terminal states
                next_state = self._process_obs(obs)

                # Store the transition in memory
                if action != 0:
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
                    self.plot_durations()
                    break

        print("[Training Complete]")
        self.plot_durations(show_result=True)
        # plt.ioff()
        plt.show()

        torch.save(self.policy_net.state_dict(), "/data/expe-out/policy_weights.pth")
        torch.save(self.target_net.state_dict(), "/data/expe-out/target_weights.pth")


    def play(self, env, verbose=True):
        history = { "score": 0, "steps": 0, "info": None }
        obs, done, info = env.reset(), False, {}
        obs = self._process_obs(obs)

        while not done:
            obs, reward, done, info = env.step( self.act(obs) )
            obs = self._process_obs(obs)

            history['score'] += reward
            history['steps'] += 1
            history['info']   = info

        if verbose:
            print(f"[DONE] Score: {history['score']} - Steps: {history['steps']}")


    def _process_obs(self, obs):
        platform = obs["platform"]["hosts"].ravel()
        queue    = obs["queue"]

        res = torch.zeros(queue["jobs"].shape[0], 4 + platform.shape[0] )
        for i, job in enumerate(queue["jobs"]):
            # TODO - Rev MAX_WALL value
            res[i, 0] = i
            res[i, 1] = obs["current_time"] - job[0]         # Waiting time
            res[i, 2] = job[1] / obs["platform"]["nb_hosts"] # Res needed
            res[i, 3] = job[2] / MAX_WALL                    # Wall

        return res


