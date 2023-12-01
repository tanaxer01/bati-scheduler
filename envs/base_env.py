from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import xml.etree.ElementTree as ET
from abc import abstractmethod
import os

import batsim_py
from batsim_py.jobs import Job
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

        self.host_speeds : Dict[str, float] = self._get_host_speeds()
        self.completed_jobs : Set[int] = {*()}
        self.simulator.subscribe(batsim_py.events.JobEvent.COMPLETED, self._on_job_completed)

    @property
    def valid_jobs(self) -> List[Job]:
        """  Filter input and returns all jobs that have their dependencies fulfilled."""
        dependencies_met = lambda x: "real_subtime" in x.metadata or "dependencies" not in x.metadata

        # Jobs real subtime needs to take into account when they became available
        valid_jobs  = filter(lambda x: dependencies_met(x), self.simulator.queue)
        sorted_jobs = sorted(valid_jobs, key=lambda x: x.metadata["real_subtime"] if "dependencies" in x.metadata else x.subtime)
        return sorted_jobs

    def _on_job_completed(self, job: Job) -> None:
        """
        This function keeps track of completed jobs and updates the real time when jobs with
        dependencies became available.
        """

        self.completed_jobs.add( int(job.name) )

        child_nodes = filter(lambda j: "dependencies" in j.metadata and int(job.name) in j.metadata["dependencies"], self.simulator.jobs)
        for j in child_nodes:
            # This ensures that if the job has more than one dependency, `real_subtime` will take the value of the last one.
            j.metadata["real_subtime"] = self.simulator.current_time

    def _get_host_speeds(self) -> Dict[str, float]:
        """ Reads the compputing seeds of each host, and stores them in Kflops """

        root = ET.parse(self.platform_fn).getroot()

        prefixes = { "G": 10e9, "M": 10e6, "K": 10e3 }
        lower_bound = prefixes["K"]

        str_speeds = { h.attrib["id"]: h.get("speed").split(",")[0][:-1] for h in root.iter("host") }
        int_speeds = { k: float(h[:-1]) * prefixes[h[-1]] for k,h in str_speeds.items() }
        norm_speeds = { k: v/lower_bound for k, v in int_speeds.items() }

        return norm_speeds

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

