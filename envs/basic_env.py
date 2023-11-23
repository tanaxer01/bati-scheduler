from typing import Any, List, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.utils import seeding

import numpy as np
import batsim_py
import os

INF = float('inf')

class BasicEnv(gym.Env):
    """Basic Environment"""

    metadata = { "render.modes": [] }

    def __init__(self,
                 platform_fn: str,
                 workload_fn: str,
                 t_action: int = 1,
                 t_shutdown: int = 0,
                 queue_max_len: int = 20,
                 seed: Optional[int] = None,
                 simulation_time: Optional[float] = None,
                 verbosity: batsim_py.simulator.BatsimVerbosity = 'quiet') -> None:

        super().__init__()

        if not platform_fn:
            raise error.Error("Expected `platform_fn` argument to be a non "
                              f"empty string, got {platform_fn}.")
        elif not os.path.exists(platform_fn):
            raise error.Error(f"File {platform_fn} does not exist.")
        else:
            self.platform_fn = platform_fn

        if not workload_fn:
            raise error.Error("Expected `workload_fn` argument to be a non "
                              f"empty string, got {platform_fn}.")
        elif not os.path.exists(workload_fn):
            raise error.Error(f"File {workload_fn} does not exist.")
        else:
            self.workload_fn = workload_fn

        if t_action < 0:
            raise error.Error("Expected `t_action` argument to be greater "
                              f"than zero, got {t_action}.")

        self.seed(seed)
        self.simulator = batsim_py.SimulatorHandler()
        self.simulation_time = simulation_time
        self.workload: Optional[str] = None
        self.verbosity: batsim_py.simulator.BatsimVerbosity = verbosity
        self.t_action = t_action
        self.queue_max_len = queue_max_len
        self.observation_space, self.action_space = self._get_spaces()

    def reset(self, seed=None, options=None):
        self._close_simulator()
        self._start_simulator()

        self.observation_space, self.action_space = self._get_spaces()
        return self._get_state(), {}

    def render(self, mode: str = 'human'):
        raise error.Error(f"Not supported.")

    def close(self):
        self._close_simulator()

    def seed(self, seed: Optional[int] = None) -> Sequence[int]:
        self.np_random, s = seeding.np_random(seed)
        return [s]

    def _close_simulator(self) -> None:
        self.simulator.close()

    def _start_simulator(self) -> None:
        self.simulator.start(platform=self.platform_fn,
                             workload=self.workload_fn,
                             verbosity=self.verbosity,
                             simulation_time=self.simulation_time)

###############################################################################
# Custom Functions
###############################################################################

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        if not self.simulator.is_running or not self.simulator.platform:
            raise error.ResetNeeded("Simulation not running.")

        print("-->", self.simulator.current_time)

        # Get the job object from the sim.
        available_hosts = self.simulator.platform.get_not_allocated_hosts()
        posible_jobs = [ j for j in self.simulator.queue if j.res <= len(available_hosts) ]

        reward = 0.
        if len(posible_jobs):
            # Assignate the chosen job into best alloc.
            print("!!", action.item(), len(posible_jobs) )
            job = posible_jobs[ int(action.item()) ]
            res = [ h.id for h in available_hosts[:job.res] ]

            self.simulator.allocate(job.id, res)
            reward = self._get_reward()

        # Proceed time til next assignation.
        done = obs = None
        info = { "workload": self.workload }

        while not done :
            self.simulator.proceed_time(self.t_action)
            done = not self.simulator.is_running
            obs = self._get_state()

            if obs.shape[0]:
                break

        return (obs, reward, done, False, info)

    def _get_reward(self) -> float:
        running = [ j for j in self.simulator.jobs if j.is_running ]
        w_running  = [ self.simulator.current_time - j.subtime for j in running ]

        return -1 * sum( 100 for i in w_running if i > 1 )

    def _get_state(self):
        available_hosts = self.simulator.platform.get_not_allocated_hosts()
        posible_jobs = [ j for j in self.simulator.queue if j.res < len(available_hosts) ]

        res = np.zeros( (self.queue_max_len, 3, 1) , dtype=np.float32)
        '''
        if res.shape[0] != 0:
            # Job info
            ## Wait time
            res[:, 0] = [ self.simulator.current_time - j.subtime for j in posible_jobs ]
            ## Resources
            res[:, 1] = [ j.res / len(available_hosts) for j in posible_jobs ]
            ## Walltime
            res[:, 2] = [ j.walltime or -1 for j in posible_jobs ]
        '''

        return res

    def _get_spaces(self):
        obs_shape = (self.queue_max_len, 3, 1)

        observation_space = spaces.Box(low=0, high=INF, shape=obs_shape, dtype=np.float32)
        action_space = spaces.Discrete(100)

        return observation_space, action_space





