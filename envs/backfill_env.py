from typing import Any, Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET
import numpy as np

import batsim_py
from batsim_py.jobs import Job

from gymnasium import error, spaces

from .base_env import SchedulingEnv

INF = float('inf')

class BackfillEnv(SchedulingEnv):
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

        self.host_speeds : Dict[str, float] = self._get_host_speeds()
        self.completed_jobs : Set[int] = {*()}

        self.simulator.subscribe(batsim_py.events.JobEvent.COMPLETED, self._on_job_completed)

    @property
    def valid_jobs(self) -> List[Job]:
        """  Filter input and returns all jobs that can be allocated """

        # `get_not_allocated_hosts()` gives us the hosts which are available for allocating jobs.
        nb_avail = sum( 1 for _ in self.simulator.platform.get_not_allocated_hosts() )

        resources_met    = lambda x: x.res <= nb_avail
        dependencies_met = lambda x: "real_subtime" in x.metadata or "dependencies" not in x.metadata

        # Jobs real subtime needs to take into account when they became available
        #valid_jobs  = filter(lambda x: resources_met(x) and dependencies_met(x), self.simulator.queue)

        valid_jobs  = filter(lambda x: dependencies_met(x), self.simulator.queue)
        sorted_jobs = sorted(valid_jobs, key=lambda x: x.metadata["real_subtime"] if "dependencies" in x.metadata else x.subtime)
        return sorted_jobs

    def reset(self, seed=None, options=None) -> Any:
        self._close_simulator()
        self._start_simulator()

        self._fcfs_til_next_step()

        self.observation_space, self.action_space = self._get_spaces()
        return self._get_state(), {}

    def _get_host_speeds(self) -> Dict[str, float]:
        """ Reads the compputing seeds of each host, and stores them in Kflops """

        root = ET.parse(self.platform_fn).getroot()

        prefixes = { "G": 10e9, "M": 10e6, "K": 10e3 }
        lower_bound = prefixes["K"]

        str_speeds = { h.attrib["id"]: h.get("speed").split(",")[0][:-1] for h in root.iter("host") }
        int_speeds = { k: float(h[:-1]) * prefixes[h[-1]] for k,h in str_speeds.items() }
        norm_speeds = { k: v/lower_bound for k, v in int_speeds.items() }

        return norm_speeds

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

            self.simulator.proceed_time()

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
        nb_hosts = sum( 1 for _ in self.simulator.platform.hosts )
        nb_avail = sum( 1 for _ in self.simulator.platform.get_not_allocated_hosts() )

        backfill_options = self._backfill_options()
        backfill_jobs = list(map(lambda x: self.simulator.jobs[x[0] + 1], backfill_options))
        backfill_allocs = list(map(lambda x: x[1], backfill_options))

        # Backfill status
        jobs = np.zeros( (len(backfill_jobs), 5) )

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

