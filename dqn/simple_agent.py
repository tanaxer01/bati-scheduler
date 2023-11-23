import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
from torch.nn.modules import loss
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
# BATCH_SIZE = 128
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleAgent():
    def __init__(self, state_size, action_size=1):
        self.state_size = state_size
        self.action_size = action_size

        # Q Network
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        # Replay Memory
        self.memory = ReplayMemory(20000)

        self.steps_done = 0

    def act(self, obs):
        match obs.shape[0]:
            case 0:
                return torch.tensor([0], device=device, dtype=torch.float32).unsqueeze(0)
            case 1:
                return torch.tensor([1], device=device, dtype=torch.float32).unsqueeze(0)
            case _:
                return self._choose_best(obs)

    def _choose_best(self, obs):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        # Exploration
        if random.random() <= eps_threshold:
            res = random.randint(0,obs.shape[0]-1)
            return torch.tensor([res], device=device, dtype=torch.float32).unsqueeze(0)

        # Explotation
        action = self._predict_scores(obs)
        return torch.tensor([action], device=device, dtype=torch.float32).unsqueeze(0)

    def _predict_scores(self, states):
        with torch.no_grad():
            scores, _ = self.policy_net(states).max(1)
            max_job   = scores.argmax()

        return max_job

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

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = [s for s in batch.next_state if s is not None]

        #state_batch = torch.cat(batch.state)
        state_batch  = batch.state
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        # TODO - FIX
        #state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # ----------
        state_action_values = torch.concat([ self.policy_net(i).max().unsqueeze(0) for i in state_batch ]).unsqueeze(1)
        state_action_values = state_action_values.gather(0, action_batch.type(torch.int64))
        # ----------


        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)


        with torch.no_grad():
            # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

            # ----------
            next_state_values[non_final_mask] = torch.concat([ self.policy_net(i).max().unsqueeze(0) for i in non_final_next_states if i is not None])
            # ----------

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.logs[0]['loss'].append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

         # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, env):
        #num_episodes = NUM_EPISODES if torch.cuda.is_available() else 1
        num_episodes = 1

        self.logs = {}

        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            state, _ = env.reset()
            state = self._process_obs(state)
            self.logs[i_episode] = { "scores": [], "queue_len": [], "wait": [], "loss": [] }

            for t in count():
                action = self.act(state)
                obs, reward, terminated, truncated, info = env.step(action)
                reward = torch.tensor([reward], device=device)
                done = truncated or terminated

                waits = env.simulator.current_time - obs["queue"]["jobs"][:,0]
                self.logs[i_episode]["scores"].append(float(reward))
                self.logs[i_episode]["queue_len"].append(obs["queue"]["jobs"].shape[0])
                self.logs[i_episode]["wait"].append( waits.mean() if waits.shape[0] else 0. )

                if terminated or obs["queue"]["size"] == 0:
                    next_state = None
                else:
                    next_state = self._process_obs(obs)

                # Store the transition in memory
                #if action != 0 or info["prev_action"] != 0:
                if action != 0 and (next_state != None and next_state.shape[0] > 1):
                    self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = self._process_obs(obs)

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
                    break

        with open('/data/expe-out/scores.json', 'w') as fp:
            json.dump(self.logs, fp)

        print(f"[Training Complete]")
        # plt.ioff()
        #plt.show()

        torch.save(self.policy_net.state_dict(), "/data/expe-out/policy_weights.pth")
        torch.save(self.target_net.state_dict(), "/data/expe-out/target_weights.pth")

    def play(self, env, verbose=True):
        #checkpoint = torch.load("/data/expe-out/policy_weights.pth")
        #self.policy_net.load_state_dict(checkpoint)

        logs = { "scores": [], "queue_len": [], "wait": [] }

        history = { "score": 0, "steps": 0, "info": None }
        (obs, _), done, info = env.reset(), False, {}
        obs = self._process_obs(obs)

        while not done:
            obs, reward, done, _, info = env.step( self.act(obs) )
            obs = self._process_obs(obs)

            if obs.shape[0] != 0:
                ###
                waits = env.simulator.current_time - obs[:,0]

                logs["scores"].append(float(reward))
                logs["queue_len"].append(obs.shape[0])
                #logs["wait"].append( waits.mean() if waits.shape[0] else 0. )
                logs["wait"].append( waits.mean().item() )

            history['score'] += reward
            history['steps'] += 1
            history['info']   = info
            ###

        with open('/data/expe-out/play_scores.json', 'w') as fp:
            json.dump(logs, fp)

        if verbose:
            print(f"\n[DONE] Score: {history['score']} - Steps: {history['steps']} {len(env.simulator.queue)}")

    def _process_obs(self, obs):
        queue = obs["queue"]
        platform = obs["platform"]

        # State matrix
        state = torch.zeros(queue["jobs"].shape[0], 6)
        for i, (sub, res, wall) in enumerate(queue["jobs"]):
            # Task stuff
            ## Waiting time
            state[i, 0] = obs["current_time"] - sub
            ## Resources
            state[i, 1] = res / platform["nb_hosts"]
            ## Walltime
            state[i, 2] = wall
            ## Queue len
            state[i, 3] = queue["jobs"].shape[0]

            candidates = np.delete(queue["jobs"], i-1, 0)
            candidates = candidates[candidates[:, 1] <= res]

            # Queue stuff
            ## Waiting time
            state[i, 3] = candidates[:,0].max() if candidates.shape[0] != 0 else 0.
            ## Resources
            #state[i, 4] = len(candidates)/ platform["nb_hosts"]
            ## Resources
            state[i, 4] = candidates[:,1].max() / platform["nb_hosts"] if candidates.shape[0] != 0 else 0.
            ## Walltime
            state[i, 5] = candidates[:,2].max() if candidates.shape[0] != 0 else 0.

        return state

