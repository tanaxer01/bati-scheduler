
import numpy as np
from typing import Any, Optional, Tuple

import batsim_py

import gymnasium as gym
from gymnasium import error

from .base_env import SchedulingEnv

INF = float('inf')

class BackfillEnv(SchedulingEnv):

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        if not self.simulator.is_running or not self.simulator.platform:
            raise error.ResetNeeded("Simulation not running.")

        reward = 0.

        obs = self._get_state()
        done = not self.simulator.is_running
        info = { "workload": self.workload_fn }
        return (obs, reward, done, False, info)
