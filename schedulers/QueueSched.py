from typing import List
from .dqn import Agent 

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Calc scores.
        max_score, best_act = None, None
        scores = self._score_jobs(queue["jobs"])

        # Choose the best job


        return 0

    def _score_jobs(self, queue) -> List[float]:

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

