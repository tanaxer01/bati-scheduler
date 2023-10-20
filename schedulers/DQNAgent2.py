from typing import Optional, Tuple
import numpy as np
import torch

from collections import deque
from itertools   import combinations

from .dqn import Agent
from .FreeSpaces import JobAgenda, FreeSpace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentWrapper:
    def __init__(self, state_size, action_size, seed):
        self.seed  = np.random.seed(seed)
        self.agent = Agent(state_size, action_size, seed)

    def act(self, obs, eps):
        posible_spaces = obs["posible_spaces"]
        current_job    = obs["current_job"]

        ### Invalid actions
        if (current_job[1] != 0 and current_job[1] == current_job[2]) or len(posible_spaces) == 0:
            print(f"act: {'Ready' if len(posible_spaces) != 0 else 'No spaces' }")
            return -1, -1

        ### Exploration v Explotation
        if np.random.uniform(0,1) > eps:
            choice = np.random.choice(posible_spaces.shape[0])
            print("act: exploration", posible_spaces[choice,0:2])
            return tuple(posible_spaces[choice,0:2])

        max_score, best_act  = None, None
        scores = self._predict_scores(posible_spaces, current_job)
        for i, (core, space, *_) in enumerate(posible_spaces):
            score = scores[i]
            if not max_score or score > max_score:
                max_score = score
                best_act = (core, space)

        print("act: explotation", best_act)
        return best_act

    def _predict_scores(self, spaces, curr_job):
        inputs  = np.hstack((spaces, np.tile(curr_job, (spaces.shape[0], 1))))
        outputs = np.zeros(len(inputs))

        for i, j in enumerate(inputs):
            with torch.no_grad():
                obs  = torch.from_numpy(j).float().unsqueeze(0).to(device)
                pred = self.agent.qnetwork_local(obs)
                outputs[i] = np.argmax(pred)

        return outputs
        # TODO: Prep obs for self.agent.act

        ## Queue stadistical data
        ## Plaform Data
        # Host utilization
        # Host next free start
        # Host consumption
        ## Agenda  Data

        queue_len  = queue["size"]
        queue_wait = obs["current_time"] - queue["jobs"][:,0]
        queue_res  = queue["jobs"][:,1]
        queue_wall = queue["jobs"][:,2]

        host_status = np.zeros(5)
        for i in platform["status"]:
            host_status[i-1] += 1
        host_status /= platform["agenda"].shape[0]
        host_remaining_time = obs["current_time"] - (platform["agenda"][:,0] + platform["agenda"][:,1])

        all_data = np.concatenate([
            [queue_len],
            queue_wait,
            queue_res,
            queue_wall,
            host_status,
            host_remaining_time
        ])

        obs = torch.from_numpy(all_data).float().unsqueeze(0).to(device)
        self.agent.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.agent.qnetwork_local(obs)
            self.agent.qnetwork_local.train()
            valid_actions = np.array([ j if i < queue_len else -1 * float('inf') for i,j in enumerate(action_values.tolist()[0])])


        return np.argmax(valid_actions) - 1

    '''
    def _score_spaces(self, spaces):
        scores = np.zeros(len(spaces))

        # wait_time
        for i in range(len(spaces)):
            scores[i] += spaces[i].start - self.listFreeSpaces.curr_time
        # TODO: estimate_exec_time

        # EnergyConsumption
        #

        exp_scores = np.exp(scores)
        actions_prob = exp_scores / exp_scores.sum()

        return actions_prob
    '''


    # TODO FIX TRAIN ALGO
    def train(self, env, n_episodes = 200, max_t = 1000, eps_start = 1.0, eps_end = 0.1, eps_decay = 0.996, verbose=True) -> None:
        scores = [] # list containing score from each episode
        scores_window = deque(maxlen = 100) # last 100 scores
        eps = eps_start

        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0

            for t in range(max_t):
                action = self.act(state, eps)
                next_state, reward, done, _ = env.step(action)

                ## above step decides whether we will train(learn) the network
                ## actor (local_qnetwork) or we will fill the replay buffer
                ## if len replay buffer is equal to the batch size then we will
                ## train the network or otherwise we will add experience tuple in our
                ## replay buffer.
                state = next_state
                score += reward
                if done:
                    break

                scores_window.append(score)
                scores.append(score)

                eps = max(eps*eps_decay, eps_end)
                print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
                if i_episode %100==0:
                    print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))

                if np.mean(scores_window)>=200.0:
                    print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                           np.mean(scores_window)))
                    torch.save(self.agent.qnetwork_local.state_dict(),'checkpoint.pth')
                    break
        return scores

    def play(self, env, load_data=False, verbose=True) -> None:
        if load_data:
            self.agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

        eps = 1.0
        eps_end = 0.1
        eps_decay = 0.996

        history = { 'score': 0, 'steps': 0, 'info': None }
        obs, done, info = env.reset(), False, {}

        while not done:
            print("STEP", history["steps"])
            obs, reward, done, info = env.step(self.act(obs, eps))
            history['score'] += reward
            history['steps'] += 1
            history['info'] = info

            #if history["steps"] == 74:
            #    break
            #if history["score"] < -100:
            #    break

            eps = max(eps*eps_decay, eps_end)


        if verbose:
            print(f"[DONE] Score: {history['score']} - Steps: {history['steps']}")

