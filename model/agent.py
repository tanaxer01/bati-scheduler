import math
import random
from typing import Optional
import numpy as np
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .network import DQN, DDQN
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

class Agent():
    def __init__(self, state_size, monitors: Optional[MonitorsInterface] = None):
        # Q Network
        self.state_size  = state_size
        self.action_size = 1

        #self.net = DDQN(state_size, 1)

        self.policy_net = DQN(state_size, 1)
        self.target_net = DQN(state_size, 1)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        for p in self.target_net.parameters():
            p.requires_grad = False

        # Replay Memory
        self.memory = ReplayMemory(20000)
        self.steps_done = 0

        # Model update fn
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn   = nn.SmoothL1Loss()

        # Learning parameters
        self.burnin      = BATCH_SIZE # min. experiences before training
        self.learn_every = 2   # no. of experiences between updates to Q_online
        self.sync_every  = 1 # no. of experiences between Q_target & Q_online sync
        # self.sync_every  = 1e4
        self.save_every  = 5e5 # no. of experiences between saving Net

        self.monitors = monitors

    def td_estimate(self, state, action):
        ''' Computes Q(s_t, a), and then '''
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = torch.cat([ self.policy_net(i).max(1)[0] for i in state ]).unsqueeze(0)
        #state_action_values = torch.cat([ self.policy_net(i)[0,int(j)].unsqueeze(0) for i,j in zip(state, action.tolist()[0]) ])

        return state_action_values.gather(1, action.type(torch.int64)).T
        #return state_action_values.unsqueeze(1)

    @torch.no_grad()
    def td_target(self, reward, next_state):
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)), device=device, dtype=torch.bool)
        non_final_next_states = [s for s in next_state if s is not None]

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            #next_state_values[non_final_mask] = torch.tensor([ self.policy_net(i).max(1).values for i in non_final_next_states ])
            next_state_values[non_final_mask] = torch.tensor([ self.target_net(i).max(1).values for i in non_final_next_states ])

        return (next_state_values * GAMMA) + reward

    def _update_online_Q(self, td_estimate, td_target):
        """ Calculates the loss in the actual step, and updates the weights accordingly"""

        loss = self.loss_fn(td_estimate, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def _sync_Q_target(self):
        """ Syncronize the weights of the target_net with the ones from the online_net. """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

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

        self.steps_done += 1
        return action_idx

    def learn(self):
        if self.steps_done % self.sync_every == 0:
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            self._sync_Q_target()

        # if self.steps_done % self.save_every == 0:
        #     #self.save()

        if self.steps_done < self.burnin:
            return None, None

         # if self.steps_done % self.learn_every != 0:
         #    return None, None

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Sample from memory
        next_state_batch  = batch.next_state
        state_batch  = batch.state
        action_batch = torch.cat(batch.action).unsqueeze(0)
        reward_batch = torch.cat(batch.reward)

        td_est = self.td_estimate(state_batch, action_batch)
        td_tgt = self.td_target(reward_batch, next_state_batch)

        loss = self._update_online_Q(td_est, td_tgt.unsqueeze(1))

        return loss, td_est.mean().item()

    '''
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


        state_action_values = self.td_estimate(state_batch, action_batch)

        expected_state_action_values = self.td_target(reward_batch, batch.next_state)

        loss = self._update_online_Q(state_action_values, expected_state_action_values.unsqueeze(1))

        return (loss, expected_state_action_values.mean().item())
    '''

    def save(self):
        save_path = Path("/data/expe-out/network.chkpt/netwrk")
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        torch.save( dict(model=self.policy_net.state_dict(), exploration_rate = eps), save_path)
        print(f"Saved network checkpoing to {save_path} at step{self.steps_done}")

    def load(self, checkpoint_fn):
        checkpoint = torch.load(checkpoint_fn)

        self.policy_net.load_state_dict(checkpoint["model"])
        self.target_net.load_state_dict(checkpoint["model"])

    def play(self, env, episodes=40,save=False):
        use_cuda = torch.cuda.is_available()
        print(f"Using CUDA: {use_cuda}", end="\n\n")

        logger = None
        if save:
            save_dir = Path("/data/expe-out/checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            save_dir.mkdir(parents=True)

            logger = MetricLogger(save_dir)

        for e in range(episodes):
            state, _ = env.reset()
            state = self._process_obs(state)

            # Play the game!
            while True:
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
                loss, q = self.learn()

                # Logging
                # for m in self.monitors:
                #    if type(m) == TrainingMonitor:
                #        m.log_step(reward.item(), loss, q)

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

        if save:
            self.save()

    def test(self, env):
        if self.monitors:
            self.monitors.init_episode(env.simulator, True)

        history = { "score": 0, "steps": 0, "info": None }
        done, info = False, {}
        state, _ = env.reset()
        state = self._process_obs(state)

        if state.size(0) == 0:
            print("NO BACKFILL NEEDED")
            if self.monitors:
                print(f"SAVING IN {self.monitors.save_dir}")
                self.monitors.record()
            return


        print(state.shape)

        # Play the game!
        while not done:
            state, reward, done, _, info = env.step( self.act(state) )
            state = self._process_obs(state)

            history['score'] += reward
            history['steps'] += 1
            history['info']   = info

        if self.monitors:
            print(f"SAVING IN {self.monitors.save_dir}")
            self.monitors.record()

        print(f"[DONE] Score: {history['score']} - Steps: {history['steps']}")

    def _process_obs(self, obs):
        queue = obs["jobs"]

        # State matrix
        state = torch.zeros(queue.shape[0], 8)
        for i, (wait, res, wall, flops, deps) in enumerate(queue):
            # Task stuff
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

            '''
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
            '''

        #return state
        return torch.from_numpy(queue).type(torch.float32)

