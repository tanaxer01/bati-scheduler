from typing import Any, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
import numpy as np
import batsim_py
import os

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.utils import seeding

INF = float('inf')

class SkipTime(gym.Wrapper):
    def __init__(self, env):
        """Return only every `skip`-th frame"""
        super().__init__(env)

    def _advance_time(self):
        envv = self.env.unwrapped

        start_t = envv.simulator.current_time
        while envv.simulator.is_running:
            available_hosts = envv.simulator.platform.get_not_allocated_hosts()
            posible_jobs = [ j for j in envv.simulator.queue if j.res <= len(available_hosts) ]

            if len(posible_jobs) > 0:
                break
            print(".",end="")

            envv.simulator.proceed_time(envv.t_action)
        end_t = envv.simulator.current_time

        if start_t != end_t:
            print("\n<< Start assignation >>")

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._advance_time()

        return self.env.unwrapped._get_state(), {}

    def step(self, action):
        """Pass time til a job can be chosen"""

        # Select the task
        obs, reward, done, trunc, info  = super().step(action)

        # Advance time til next valid state
        self._advance_time()

        # Update the obs before returning it.
        obs  = self.env.unwrapped._get_state()
        done = not self.env.unwrapped.simulator.is_running
        return obs, reward, done, trunc, info

class SimpleEnv(gym.Env):
    """Basic Environment"""

    metadata = { "render.modes": [] }

###############################################################################
# Base Functions
###############################################################################

    def __init__(self,
                 platform_fn: str,
                 workload_fn: str,
                 t_action: int = 1,
                 t_shutdown: int = 0,
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
        self.verbosity: batsim_py.simulator.BatsimVerbosity = verbosity
        self.observation_space, self.action_space = self._get_spaces()
        self.workload: Optional[str] = None
        ##
        self.t_action = t_action
        self.simulation_time = simulation_time
        self.host_speeds = self._get_host_speeds()

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

        # 1. Get job selected by the agent.
        available_hosts = self.simulator.platform.get_not_allocated_hosts()
        posible_jobs = [ j for j in self.simulator.queue if j.res <= len(available_hosts) ]

        job = posible_jobs[ int(action) ]

        # 2. Get the hosts where it's going to be allocated in.
        res = [ h.id for h in available_hosts[:job.res] ]

        # 3. Allocate and measure reward.
        self.simulator.allocate(job.id, res)
        reward = self._get_reward()

        print(f"{self.simulator.current_time}\t({int(action)}/{len(posible_jobs)}, {self.simulator.current_time - job.subtime}) {reward} - {len(posible_jobs)}")

        obs = self._get_state()
        done = not self.simulator.is_running
        info = { "workload": self.workload }
        return (obs, reward, done, False, info)

    def _get_host_speeds(self):
        """ Reads the compputing seeds of each host, and stores them in Kflops """
        root = ET.parse(self.platform_fn).getroot()

        prefixes = { "G": 10e9, "M": 10e6, "K": 10e3 }
        lower_bound = prefixes["K"]

        str_speeds = { h.attrib["id"]: h.get("speed").split(",")[0][:-1] for h in root.iter("host") }
        int_speeds = { k: float(h[:-1]) * prefixes[h[-1]] for k,h in str_speeds.items() }
        norm_speeds = { k: v/lower_bound for k, v in int_speeds.items() }

        return norm_speeds

    def _get_reward(self) -> float:
        score = 0.

        queue = [ j for j in self.simulator.queue ]

        if len(queue) != 0:
            score -= 10 * sum([ (self.simulator.current_time - j.subtime)/j.walltime for j in queue ])/len(queue)

        nb_hosts = sum( 1 for _ in self.simulator.platform.get_not_allocated_hosts() )

        total = 0
        if len(self.simulator.queue):
            total = sum([ j.walltime * j.res for j in self.simulator.queue ]) / len(self.simulator.queue)

        return -1 * total

    def _get_state(self):
        nb_hosts = sum( 1 for _ in self.simulator.platform.hosts )
        nb_avail = sum( 1 for _ in self.simulator.platform.get_not_allocated_hosts() )

        posible_jobs = [ j for j in self.simulator.queue if j.res <= nb_avail ]
        # Queue status
        jobs = np.zeros( (len(posible_jobs), 4) )
        if len(posible_jobs) != 0:
        ## Subtime
            jobs[:,0] = [ j.subtime for j in posible_jobs ]
        ## Resources
            jobs[:,1] = [ j.res     for j in posible_jobs ]
            #jobs[:,1] /= len(available_hosts)
        ## Walltime
            jobs[:,2] = [ j.walltime if j.walltime else -1 for j in posible_jobs ]
            #jobs[:,2] /= jobs[:,2].max()
        ## Flops
            jobs[:,3] = [ j.profile.cpu for j in posible_jobs ]

        queue = { "size": len(posible_jobs), "jobs": jobs }

        # Platform status
        hosts = np.zeros( (nb_hosts, 2) )

        ## Status (0 - sleep, ..., 3 - computation)
        hosts[:,0] = [ not h.is_allocated for h in self.simulator.platform.hosts ]
        ## Speed (Kflops)
        hosts[:,1] = [ self.host_speeds[h.name] for h in self.simulator.platform.hosts ]


        ## Energy consumption
        #hosts[:,2] = [  for h in self.simulator.platform.hosts ]
        ## Energy left (?)

        ## Sort by status & speed
        #sorted_indices = np.lexsort((-hosts[:, 1], -hosts[:, 0]))
        #hosts = hosts[sorted_indices]

        platform = { "nb_hosts": nb_avail, "hosts": hosts }

        # Full state
        state = {
             "queue": queue,
             "platform": platform,
             "current_time": self.simulator.current_time
        }

        return state

    def _get_spaces(self):
        # TODO - update dis
        nb_avail = nb_jobs = 0

        if self.simulator.is_running:
            nb_avail = len(self.simulator.platform.get_not_allocated_hosts())
            nb_jobs  = len([ j for j in self.simulator.queue if j.res <= nb_avail ])

        obs_shape = (nb_jobs, 7, 1)

        observation_space = spaces.Box(low=0, high=INF, shape=obs_shape, dtype=np.float32)
        action_space = spaces.Discrete(1)

        return observation_space, action_space

