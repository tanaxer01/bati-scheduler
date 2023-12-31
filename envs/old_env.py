from typing import Any, Dict, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np

import batsim_py
from batsim_py.jobs import Job
from batsim_py.resources import HostState

import gymnasium as gym
from gymnasium import error, spaces

from .base_env import SchedulingEnv

INF = float('inf')

class SkipTime(gym.Wrapper):
    """Wrapper that advances the simulation time until there is a valid state to process"""

    def __init__(self, env):
        super().__init__(env)

    def _advance_time(self):
        envv = self.unwrapped

        start_t = envv.simulator.current_time
        while envv.simulator.is_running:
            valid_jobs = envv._get_posible_jobs()

            if len(valid_jobs) > 0:
                break

            print(".",end="")
            # envv.simulator.proceed_time(envv.t_action)
            envv.simulator.proceed_time()

        end_t = envv.simulator.current_time
        done  = not envv.simulator.is_running
        if start_t != end_t and not done:
            idle_hosts = filter(lambda x: x.state == HostState.IDLE, self.unwrapped.simulator.platform.hosts)
            sorted_idle = sorted(list( idle_hosts ), key=lambda x: self.unwrapped.host_speeds[x.name])

            if len(sorted_idle) > 2:
                #print([ i.state.value for i in sorted_idle ])
                print(len(sorted_idle),[ i.state.value for i in self.simulator.platform.hosts ], end=" - ")
                #print( "==>", [ i.is_unavailable for i in sorted_idle ])
                #self.simulator.switch_off([ i.id for i in sorted_idle[-1:] ])
                print(len(sorted_idle),[ i.state.value for i in self.simulator.platform.hosts ])




            print(f"\n<< Start assignation {self.unwrapped.simulator.current_time}>>")


    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._advance_time()

        return self.unwrapped._get_state(), {}

    def step(self, action):
        """Pass time til a job can be chosen"""

        # Select the task
        obs, reward, done, trunc, info  = super().step(action)

        # Advance time til next valid state
        self._advance_time()

        # Update the obs before returning it.
        obs  = self.unwrapped._get_state()
        done = not self.unwrapped.simulator.is_running
        return obs, reward, done, trunc, info

class SimpleEnv(SchedulingEnv):
    """Simple scheduling environment that can enforce dependencies between tasks"""

    def __init__(self,
                 platform_fn: str,
                 workload_fn: str,
                 t_action: int = 1,
                 t_shutdown: int = 1,
                 seed: Optional[int] = None,
                 simulation_time: Optional[float] = None,
                 verbosity: batsim_py.simulator.BatsimVerbosity = 'quiet',
                 shutdown_policy = None,
                 track_dependencies: bool = False) -> None:

        super().__init__(
                platform_fn,
                workload_fn,
                t_action,
                seed,
                simulation_time,
                verbosity)

        self.track_dependencies = track_dependencies
        self.shutdown_policy    = shutdown_policy
        self.host_speeds        = self._get_host_speeds()
        self.completed_jobs     = set()

        self.simulator.subscribe(batsim_py.events.JobEvent.COMPLETED, self._on_job_completed)

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        if not self.simulator.is_running or not self.simulator.platform:
            raise error.ResetNeeded("Simulation not running.")

        available_hosts = self.simulator.platform.get_not_allocated_hosts()
        posible_jobs = self._get_posible_jobs()

        # 1. Get job selected by the agent.
        job = posible_jobs[ int(action) ]

        # 2. Get the hosts where it's going to be allocated in.
        res = [ h.id for h in available_hosts[:job.res] ]

        # 3. Allocate and measure reward.
        self.simulator.allocate(job.id, res)
        reward = self._get_reward()

        print(f"\t{self.simulator.current_time} ({int(action)}/{len(posible_jobs)})",
              f" - {self.simulator.current_time - job.subtime} {reward}")

        obs = self._get_state()
        done = not self.simulator.is_running
        info = { "workload": self.workload_fn }
        return (obs, reward, done, False, info)

    def _on_job_completed(self, job) -> None:
        self.completed_jobs.add( int(job.name) )

    def _get_posible_jobs(self) -> Sequence[Job]:
        """ Returns all jobs that fit in the remaining Hosts and that their requisite jobs have been completed. """

        # `get_not_allocated_hosts()` gives us the hosts which are available for allocating jobs.
        nb_avail = sum( 1 for _ in self.simulator.platform.get_not_allocated_hosts() )

        resources_met    = lambda x: x.res <= nb_avail
        # If dependencies have finished, they should be in completed_jobs.
        dependencies_met = lambda x: all( i in self.completed_jobs for i in x.metadata["dependencies"]) if "dependencies" in x.metadata else True

        valid_jobs = filter(lambda x: resources_met(x) and (dependencies_met(x) or not self.track_dependencies), self.simulator.queue)
        return list(valid_jobs)

    def _get_host_speeds(self) -> Dict[str, float]:
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
        current_time = self.simulator.current_time
        nb_avail = sum( 1 for _ in self.simulator.platform.get_not_allocated_hosts() )

        # `simulator.jobs` only contains unfinished jobs, so we are only considering jobs allocated on current time.
        allocated_jobs = filter(lambda x: x.start_time == current_time, self.simulator.jobs)
        allocated_weights = np.fromiter((np.log(j.profile.cpu / j.walltime) if j.walltime else 0.for j in allocated_jobs), float)

        score += allocated_weights.mean()

        # If there are jobs that still can enter the knapsack, the reward should be better.
        valid_jobs   = self._get_posible_jobs()


        # Invalid jobs should generate a discount depening the reason ( resources needed or dependencies ).
        invalid_jobs = [ i for i in self.simulator.queue if i.name not in map(lambda x: x.name, valid_jobs) ]

        res_limited  = [ i for i in self.simulator.queue if i.res > nb_avail ]
        dep_limited  = [ i for i in invalid_jobs if i.name not in map(lambda x: x.name, res_limited) ]

        res_discount  = sum([ current_time - i.subtime for i in res_limited ]) / len(res_limited) if len(res_limited) > 0 else 0.
        dep_discount  = sum([ 1 for _ in dep_limited ]) / len(dep_limited) if len(dep_limited) > 0 else 0.

        score -= res_discount + dep_discount

        allocated_jobs = [ j for j in self.simulator.jobs if j.start_time == current_time ]
        return score

    def _get_state(self):
        nb_hosts = sum( 1 for _ in self.simulator.platform.hosts )
        nb_avail = sum( 1 for _ in self.simulator.platform.get_not_allocated_hosts() )

        valid_jobs = self._get_posible_jobs()

        # Queue status
        jobs = np.zeros( (len(valid_jobs), 5) )
        if len(valid_jobs) != 0:
        ## Subtime
            jobs[:,0] = [ j.subtime for j in valid_jobs ]
        ## Resources
            jobs[:,1] = [ j.res     for j in valid_jobs ]
        ## Walltime
            jobs[:,2] = [ j.walltime if j.walltime else -1 for j in valid_jobs ]
        ## Flops
            jobs[:,3] = [ j.profile.cpu or 0 for j in valid_jobs ]
        ## Dependencies
            queue_deps = sum([ i.metadata["dependencies"] for i in self.simulator.queue if "dependencies" in i.metadata ], [])
            jobs[:,4] = [ queue_deps.count(int(i.name)) for i in valid_jobs ]
            # jobs[:,4] = [ len(j.metadata["dependencies"]) if "dependencies" in j.metadata else 0 for j in valid_jobs ]

        queue = { "size": len(valid_jobs), "jobs": jobs }

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
        nb_avail = nb_jobs = 0

        if self.simulator.is_running:
            nb_avail = len(self.simulator.platform.get_not_allocated_hosts())
            nb_jobs  = len([ j for j in self.simulator.queue if j.res <= nb_avail ])

        obs_shape = (nb_jobs, 7, 1)

        observation_space = spaces.Box(low=0, high=INF, shape=obs_shape, dtype=np.float32)
        action_space = spaces.Discrete(1)

        return observation_space, action_space

