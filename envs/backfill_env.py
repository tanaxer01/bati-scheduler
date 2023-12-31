from batsim_py import simulator
from batsim_py.resources import PowerStateType
from gymnasium import error, spaces
from typing import Any, Tuple
import numpy as np

from .base_env import SchedulingEnv

INF = float('inf')

class BackfillEnv(SchedulingEnv):

    def reset(self, seed=None, options=None) -> Any:
        self._close_simulator()
        self._start_simulator()

        self._fcfs_til_next_step()
        # Simulation ended xd

        self.observation_space, self.action_space = self._get_spaces()
        return self._get_state(), {}

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        if not self.simulator.is_running or not self.simulator.platform:
            raise error.ResetNeeded("Simulation not running.")

        reward, finished, truncated = 0., False, False

        idx, allocation = self._backfill_options()[ int(action) ]
        job = self.valid_jobs[idx+1] # idx 0 is the priority job
        self.simulator.allocate(job.id, allocation)


        reward = self._get_reward()

        if len(self._backfill_options()) == 0:

            # Just allocated the last valid action.
            finished, truncated  =self._fcfs_til_next_step(), True

        obs    = self._get_state()
        info = { "workload": self.workload_fn }
        return (obs, reward, finished, truncated, info)

    def _backfill_options(self):
        options = []

        available = sorted(self.simulator.platform.get_not_allocated_hosts(), key=lambda h: self.host_speeds[ h.name ])

        if len(self.valid_jobs) == 0:
            return []

        # The priority job is the first job in the queue
        p_job = self.valid_jobs[0]

        # The remaining jobs can be scheduled if they do not delay p_job.
        backfill_queue = self.valid_jobs[1:]

        # The last host release time will be the p_job expected start time.
        next_releases = sorted(self.simulator.agenda, key=lambda a: a.release_time)
        last_host = next_releases[p_job.res - 1]
        p_start_t = last_host.release_time

        # Find candidates and reserve resources for p_job.
        candidates = [r.host.id for r in next_releases if r.release_time <= p_start_t]

        # Try to maximize the number of hosts available for the remaining queue.
        reservation  = candidates[-p_job.res:]
        not_reserved = [h for h in available if h.id not in reservation]

        for idx, job in enumerate(backfill_queue):
            if   job.res <= len(not_reserved):
                allocation = [h.id for h in not_reserved[:job.res]]
                options.append( (idx, allocation) )
            elif job.walltime and job.walltime <= p_start_t and job.res <= len(available):
                allocation = [ h.id for h in available[:job.res] ]
                options.append( (idx, allocation) )

        return options

    def _fcfs_til_next_step(self) -> bool:
        """ Keeps advancing the simulation until there are more than 2 valid jobs in the queue, else FCFS them into the sistem. """

        while True:
            for job in self.valid_jobs:
                available = self.simulator.platform.get_not_allocated_hosts()

                if job.res <= len(available):
                    # Schedule if the job can start now.
                    allocation = [ h.id for h in available[:job.res] ]
                    self.simulator.allocate(job.id, allocation)

                else:
                    # Otherwise, wait for resources.
                    break

            if (len(self.valid_jobs) != 0 and len(self._backfill_options()) != 0) or not self.simulator.is_running:
                # Found next valid state or the simulation ended.
                return not self.simulator.is_running

            if self.t_action is None:
                self.simulator.proceed_time()
            else:
                self.simulator.proceed_time(self.t_action)

    def _get_reward(self) -> float:
        total = 0.

        current_time = self.simulator.current_time

        nb_hosts = sum(1 for _ in self.simulator.platform.hosts)
        nb_avail = sum(1 for _ in self.simulator.platform.get_not_allocated_hosts())

        # `simulator.jobs` only contains unfinished jobs, we only want those related to this backfill step.
        allocated = list(filter(lambda x: x.start_time == self.simulator.current_time, self.simulator.jobs))
        backfill  = list([j for j in self.simulator.jobs if j in  map(lambda x: x[0], self._backfill_options())])

        # allocated_weight = sum(j.profile.cpu / j.walltime if j.walltime else 0. for j in allocated)
        p_job = self.valid_jobs[0]

        next_releases = sorted(self.simulator.agenda, key=lambda a: a.release_time)
        last_host = next_releases[p_job.res - 1]
        p_start_t = last_host.release_time

        allocated_weight = sum(j.res * j.walltime if j.walltime else 0. for j in allocated)
        total_rectangle  = (current_time - p_start_t) * nb_hosts

        total += allocated_weight / total_rectangle

        '''
        # `simulator.jobs` only contains unfinished jobs, so we are only considering jobs allocated on current time.
        allocated_jobs = filter(lambda x: x.start_time == current_time, self.simulator.jobs)
        allocated_weights = np.fromiter((np.log(j.profile.cpu / j.walltime) if j.walltime else 0.for j in allocated_jobs), float)

        backfill_jobs  = ( i for i in self.simulator.jobs if int(i.name) in map(lambda x: x[0], self._backfill_options()) )
        backfill_weights = np.fromiter((np.log(j.profile.cpu / j.walltime) if j.walltime else 0.for j in backfill_jobs), float)

        #total += sum(allocated_weights)

        # Invalid jobs should generate a discount depening the reason ( resources needed or dependencies ).
        invalid_jobs = [ i for i in self.simulator.queue if i.name not in map(lambda x: x.name, valid_jobs) ]

        res_limited  = [ i for i in self.simulator.queue if i.res > nb_avail ]
        dep_limited  = [ i for i in invalid_jobs if i.name not in map(lambda x: x.name, res_limited) ]

        res_discount  = sum([ current_time - i.subtime for i in res_limited ]) / len(res_limited) if len(res_limited) > 0 else 0.
        dep_discount  = sum([ 1 for _ in dep_limited ]) / len(dep_limited) if len(dep_limited) > 0 else 0.

        score -= res_discount + dep_discount
        '''


        return total

    def _get_state(self):
        backfill_options = self._backfill_options()
        backfill_jobs = list(map(lambda x: self.simulator.jobs[x[0] + 1], backfill_options))
        backfill_allocs = list(map(lambda x: x[1], backfill_options))

        # Backfill status
        jobs = np.zeros( (len(backfill_jobs), 6) )

        if len(backfill_jobs) > 0:
            ## Waiting time
            jobs[:,0] = [ self.simulator.current_time - j.subtime for j in backfill_jobs ]
            ## Resources needed
            jobs[:,1] = [ j.res for j in backfill_jobs ]
            ## Walltime
            jobs[:,2] = [ j.walltime if j.walltime else -1 for j in backfill_jobs ]
            ## Flops
            jobs[:,3] = [ j.profile.cpu or 0 for j in backfill_jobs ]
            ## Dependencies
            queue_deps = sum([ i.metadata["dependencies"] for i in self.simulator.queue if "dependencies" in i.metadata ], [])
            jobs[:,4] = [ queue_deps.count(int(i.name)) for i in backfill_jobs ]
            ## Speed
            #jobs[:,5] = [ min([self.host_speeds[h] for h in a])  for a in backfill_allocs ]
            ## Energy
            allocs = [ [ self.simulator.platform.get_host(h) for h in hosts ] for hosts in backfill_allocs ]
            states = [ [ h.get_pstate_by_type(PowerStateType.COMPUTATION)[0] for h in hosts ] for hosts in allocs ]
            watts  = [ [ state.watt_full for state in hosts ] for hosts in states ]

            jobs[:,5] = [ sum(w) for w in watts ]

        queue = { "size": len(backfill_jobs), "jobs": jobs }
        return queue

    def _get_spaces(self):
        nb_avail = nb_jobs = 0

        if self.simulator.is_running:
            nb_avail = len(self.simulator.platform.get_not_allocated_hosts())
            nb_jobs  = len([ j for j in self.simulator.queue if j.res <= nb_avail ])

        obs_shape = (nb_jobs, 7, 1)

        observation_space = spaces.Box(low=0, high=INF, shape=obs_shape, dtype=np.float32)
        action_space = spaces.Discrete(1)

        return observation_space, action_space

