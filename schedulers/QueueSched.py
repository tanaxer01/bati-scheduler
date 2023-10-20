from typing import List
from .dqn import Agent

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO - refactor dis
MAX_WALL = 2000

class AgentWrapper:
    def __init__(self, state_size: int, action_size: int, seed) -> None:
        self.seed  = np.random.seed(seed)
        self.agent = Agent(state_size, action_size, seed)

    def act(self, obs, eps) -> int:
        queue        = obs["queue"]
        platform     = obs["platform"]
        current_time = obs["current_time"]

        # No queue nothing to do.
        if queue["size"] == 0:
            return 0

        # Get valid jobs for this step.
        nb_available = len(platform["agenda"]) - sum(1 for i in platform["agenda"] if i[1] != 0)
        job_pos = ( i for i, j in enumerate(queue["jobs"]) if 0 < j[1] <= nb_available)

        # Exploration vs Explotation
        if np.random.uniform(0, 1) > eps:
            return np.random.choice(list(job_pos)) + 1

        ## Job specific info
        # TODO - Refactore ones ready
        job_cant  = sum(1 for _ in job_pos)
        job_state = np.zeros( (job_cant, 2) )

        for i, j in enumerate(job_pos):
            job_state[i, 0] = j                                             # Job idx
            job_state[i, 1] = obs["current_time"] - obs["queue"][0]         # Wait time
            job_state[i, 2] = obs["queue"][1] / obs["platform"]["nb_hosts"] # Resources
            # TODO - Change 3 for Estim. vs Wall
            job_state[i, 3] = obs["queue"][2] / MAX_WALL                    # Wall time

        ## Queue general info

        ## Platform info
        # 1. Resource state
        state = np.zeros(5)
        for host in obs['platform']['status']:
            state[int(host) - 1] += 1
        state /= obs['platform']['agenda'].shape[0]

        inputs = np.hstack( (job_state, np.tile(state, (job_state.shape[0], 1))) )
        # Calc scores.
        max_score, best_act = None, None
        scores = self._score_jobs(inputs)
        for i, (act, *_) in enumerate(inputs):
            score = scores[i]
            if not max_score or score > max_score:
                max_score = score
                best_act  = act

        # Choose the best job
        assert best_act != None, "Todo check dis case"
        return best_act

    def _score_jobs(self, queue):
        outputs = np.zeros( queue.shape[0] )

        for i, *j in enumerate(queue):
            with torch.no_grad():
                obs  = torch.from_numpy(j).float().unsqueeze(0).to(device)
                pred = self.agent.qnetwork_local(obs)
                outputs[i] = np.argmax(pred)

        return outputs


        return []

    def train(self) -> None:
        raise NotImplementedError("TODO - implement dis")

    def play(self, env, verbose=True) -> None:

        eps_start = 1.0
        eps_end   = 0.1
        eps_decay = 0.996

        history = { "score": 0, "steps": 0, "info": None }
        obs, done, info = env.reset(), False, {}

        while not done:
            obs, reward, done, info = env.step( self.act(obs, eps_start) )

            history['score'] += reward
            history['steps'] += 1
            history['info']   = info

            print(f"STEP {history['steps']}")

            #if history["steps"] == 10:
            #    break

            eps_start = max(eps_start * eps_decay, eps_end)

        if verbose:
            print(f"[DONE] Score: {history['score']} - Steps: {history['steps']}")

