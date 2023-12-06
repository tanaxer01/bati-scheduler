from batsim_py.jobs import Job
from batsim_py.resources import HostState, PowerStateType
from gymnasium import error, spaces
from typing import Any, List, Tuple
import numpy as np

from .base_env import SchedulingEnv

INF = float('inf')

class SimpleEnv(SchedulingEnv):

    @property
    def valid_jobs(self) -> List[Job]:
        """ On top of checking dependencies, it ensures that each job can be allocated. """

         # `get_not_allocated_hosts()` gives us the hosts which are available for allocating jobs.
        nb_avail = sum( 1 for _ in self.simulator.platform.get_not_allocated_hosts() )
        resources_met = lambda x: x.res <= nb_avail

        valid_jobs = filter(lambda x: resources_met(x), super().valid_jobs)
        sorted_jobs = sorted(valid_jobs, key=lambda x: x.metadata["real_subtime"] if "dependencies" in x.metadata else x.subtime)
        return sorted_jobs

    def reset(self, seed=None, options=None) -> Any:
        self._close_simulator()
        self._start_simulator()

        self._advance_time()

        self.observation_space, self.action_space = self._get_spaces()
        return self._get_state(), {}

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        if not self.simulator.is_running or not self.simulator.platform:
            raise error.ResetNeeded("Simulation not running.")

        available_hosts = sorted(self.simulator.platform.get_not_allocated_hosts(), key=lambda h: self.host_speeds[h.name], reverse=True)
        posible_jobs = self.valid_jobs

        # 1. Get job selected by the agent.
        job = posible_jobs[ int(action) ]

        # 2. Get the hosts where it's going to be allocated in.
        res = [ h.id for h in available_hosts[:job.res] ]

        # 3. Allocate and measure reward.
        self.simulator.allocate(job.id, res)
        reward = self._get_reward()

        #print(f"\t{self.simulator.current_time} ({int(action)}/{len(posible_jobs)})",
        #      f" - {self.simulator.current_time - job.subtime} {reward}")

        # 4. Advance time til next valid state
        truncated = self._advance_time()
        #if truncated:
        #    print(f"\n<< Start assignation {self.simulator.current_time}>>")

        obs  = self._get_state()
        info = { "workload": self.workload_fn }
        terminated = not self.simulator.is_running
        return (obs, reward, terminated, truncated, info)

    def _advance_time(self):
        start_t = self.simulator.current_time

        while self.simulator.is_running:
            if len(self.valid_jobs) > 0:
                break

            # Shutdown some devices
            if self.shutdown_policy != None or 1==1:
                available = sorted(self.simulator.platform.get_not_allocated_hosts(),
                                   key=lambda h: self.host_speeds[h.name], reverse=True)
                free_res = len(available) - sum(i.res for i in self.valid_jobs)

                if free_res  > 0:
                    die_you = filter(lambda h: h.state == HostState.IDLE,[ h for h in available[-1*(free_res//2):] ])
                    die_you_ids = [ i.id for i in die_you ]
                    #self.simulator.switch_off(die_you_ids)


            self.simulator.proceed_time()

        end_t = self.simulator.current_time
        return end_t != start_t

    def _get_reward(self) -> float:
        score = 0.
        current_time = self.simulator.current_time
        nb_avail = sum( 1 for _ in self.simulator.platform.get_not_allocated_hosts() )

        # `simulator.jobs` only contains unfinished jobs, so we are only considering jobs allocated on current time.
        allocated_jobs = filter(lambda x: x.start_time == current_time, self.simulator.jobs)
        allocated_weights = np.fromiter((np.log(j.profile.cpu / j.walltime) if j.walltime else 0.for j in allocated_jobs), float)

        score += allocated_weights.mean()

        # If there are jobs that still can enter the knapsack, the reward should be better.
        valid_jobs   = self.valid_jobs


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
        available_hosts = sorted(self.simulator.platform.get_not_allocated_hosts(),
                                 key=lambda h: self.host_speeds[h.name], reverse=True)
        nb_hosts = sum( 1 for _ in self.simulator.platform.hosts )

        valid_jobs = self.valid_jobs

        # Queue status
        jobs = np.zeros( (len(valid_jobs), 6) )

        if len(valid_jobs) > 0:
            ## Waitting time
            jobs[:,0] = [ self.simulator.current_time - j.subtime for j in valid_jobs ]
            ## Resources needed
            jobs[:,1] = [ j.res for j in valid_jobs ]
            ## Walltime
            jobs[:,2] = [ j.walltime if j.walltime else -1 for j in valid_jobs ]
            ## Flops
            jobs[:,3] = [ j.profile.cpu or 0 for j in valid_jobs ]
            ## Dependencies
            queue_deps = sum([ i.metadata["dependencies"] for i in self.simulator.queue if "dependencies" in i.metadata ], [])
            jobs[:,4] = [ queue_deps.count(int(i.name)) for i in valid_jobs ]
            ## Speed
            #jobs[:,5] = [ min([self.host_speeds[h] for h in a])  for a in backfill_allocs ]
            ## Energy
            allocs = [ available_hosts[:j.res] for j in self.valid_jobs ]
            states = [ [ h.get_pstate_by_type(PowerStateType.COMPUTATION)[0] for h in hosts ] for hosts in allocs ]
            watts  = [ [ state.watt_full for state in hosts ] for hosts in states ]

            jobs[:,5] = [ sum(w) for w in watts ]

        queue = { "size": len(valid_jobs), "jobs": jobs }
        return queue

    def _get_spaces(self):
        nb_avail = nb_jobs = 0

        if self.simulator.is_running:
            nb_avail = len(self.simulator.platform.get_not_allocated_hosts())
            nb_jobs  = len([ j for j in self.simulator.queue if j.res <= nb_avail ])

        obs_shape = (nb_jobs, 6, 1)

        observation_space = spaces.Box(low=0, high=INF, shape=obs_shape, dtype=np.float32)
        action_space = spaces.Discrete(1)

        return observation_space, action_space



