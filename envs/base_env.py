from abc import abstractmethod
from typing import Any, Optional, Sequence, Tuple
import os

import batsim_py
from batsim_py.simulator import BatsimVerbosity

import gymnasium as gym
from gymnasium import error
from gymnasium.utils import seeding

class SchedulingEnv(gym.Env):
    """Basic Environment that implement all basic parts of the scheduling environment"""

    def __init__(
            self,
            platform_fn : str,
            workload_fn : str,
            t_action : int = 1,
            seed : Optional[int] = None,
            simulation_time : Optional[float] = None,
            verbosity : BatsimVerbosity = "quiet") -> None:
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
        self.verbosity : BatsimVerbosity = verbosity
        self.t_action  = t_action
        self.simulation_time = simulation_time
        self.observation_space, self.action_space = self._get_spaces()

    def reset(self, seed=None, options=None) -> Any:
        self._close_simulator()
        self._start_simulator()

        self.observation_space, self.action_space = self._get_spaces()
        return self._get_state(), {}

    def render(self, mode: str = 'human'):
        raise error.Error(f"Not supported.")

    def close(self) -> None:
        self._close_simulator()

    def seed(self, seed : Optional[int] = None) -> Sequence[int]:
        self.np_random, s = seeding.np_random(seed)
        return [s]

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        raise NotImplemented

    @abstractmethod
    def _get_state(self) -> Any:
        raise NotImplemented

    @abstractmethod
    def _get_spaces(self) -> Tuple[Any, Any]:
        raise NotImplemented

    def _close_simulator(self):
        self.simulator.close()

    def _start_simulator(self):
        if os.path.isdir(self.workload_fn):
            workloads = os.listdir(self.workload_fn)
            workloads = [ os.path.join(self.workload_fn, w) for w in workloads if w.endswith('.json') and "dependencies" not in w ]

            workload = self.np_random.choice(workloads)
        else:
            workload = self.workload_fn

        self.simulator.start(platform=self.platform_fn,
                             workload=workload,
                             verbosity=self.verbosity,
                             simulation_time=self.simulation_time,
                             allow_compute_sharing=False,
                             allow_storage_sharing=False)



