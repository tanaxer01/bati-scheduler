
from typing import Any, Optional, Tuple
import numpy as np

from batsim_py import SimulatorHandler
from batsim_py.jobs import Job
from batsim_py.events import SimulatorEvent

from gridgym.envs.grid_env import GridEnv
from gym import error, spaces

from schedulers.FreeSpaces import JobAgenda

class PerCoreEnv(GridEnv):
    def __init__(self, platform_fn: str,
                 workloads_dir:     str,
                 t_action:   int = 1,
                 t_shutdown: int = 0,
                 hosts_per_server: int = 1,
                 queue_max_len:    int = 20,
                 seed: Optional[int] = None,
                 max_plan_time: float = 20.,
                 min_plan_step: float = 1.,
                 external_events_fn: Optional[str] = None,
                 simulation_time:  Optional[float] = None) -> None:

        if t_action < 0:
            raise error.Error('Expected `t_action` argument to be greater '
                              f'than zero, got {t_action}.')

        self.queue_max_len = queue_max_len
        self.t_action = t_action

        self.max_plan_time = max_plan_time
        self.min_plan_step = min_plan_step

        self.current_job: Optional[Job] = None
        self.current_alloc = []

        self.waiting_jobs  = []
        self.running_jobs  = []

        self.hosts_speeds = None

        super().__init__(platform_fn, workloads_dir, seed,
                         external_events_fn, simulation_time, True,
                         hosts_per_server=hosts_per_server)

        self.simulator.subscribe(SimulatorEvent.SIMULATION_BEGINS, self._on_simulation_begins)
        #self.simulator.subscribe(JobEvent.COMPLETED, self._on_job_completed)
        #self.shutdown_policy = ShutdownPolicy(t_shutdown, self.simulator)

    def _read_core_speeds(self):
        if self.hosts_speeds is None:
            self.hosts_speeds = "A"

    def _on_simulation_begins(self, sim: SimulatorHandler):
        nb_cores = len(list(sim.platform.hosts))
        self.freeSpaceList = JobAgenda(nb_cores, self.max_plan_time, self.min_plan_step)

        # TODO: Read speeds from
        self._read_core_speeds()

    def _on_job_completed(self, job: Job) -> None:
        pass

    def step(self, action: Tuple[int, float]) -> Tuple[Any, float, bool, dict]:
        if self.simulator.is_running and self.simulator.platform:
            raise error.ResetNeeded("Simulation not running.")

        if self.current_job == None:
            if len(self.simulator.queue) == 0:
                # Can't make any decision.
                obs  = self._get_state()
                done = not self.simulator.is_running
                info = { "workload": self.workload }
                return obs, 0., done, info

            # Assign current_job if needed.
            self.current_job = self._get_next_job()


        scheduled, reward = False, 0.
        if action[0] == int(action[1]) == -1:
            # All res have been allocated.
            allocs = [ i[0] for i in self.current_alloc]

            # TODO: Handle wall == 0 AND str id
            assert self.current_job.walltime != None and self.current_job.walltime != -1
            start = self.current_alloc[0][0]
            stop  = start + self.current_job.walltime

            self.freeSpaceList.add_reservation(allocs, int(self.current_job.id), start, stop)
            reward = self._get_reward()

            # Reset current_alloc
            self.current_alloc = []
            scheduled = True
        else:
            self.current_alloc.append(action[0])
            reward = self._get_reward()

        # TODO: Test Scheduling job
        if scheduled:
            self.simulator.proceed_time(self.t_action)

            # Update running_jobs state
            curr_time = self.simulator.current_time
            self.running_jobs = [ i for i in self.running_jobs if i.is_running ]
            self.freeSpaceList.update(curr_time)

            # TODO: handle dependencies
            # TODO: Refactor
            # Get jobs_ids that should be assigned
            to_be_handled = {}
            for i in self.freeSpaceList.items:
                assert i.start >= curr_time
                if i.job not in self.running_jobs:
                    if i not in to_be_handled:
                        to_be_handled[i.job]  = []

                    to_be_handled[i.job].append(i)
                    assert i.start == to_be_handled[i.job].start and  i.end == to_be_handled[i.job].end

            # Allocate jobs
            to_be_handled = { j:[ i.host for i in k ] for j, k in to_be_handled.items() }
            for i, j in to_be_handled.items():
                self.simulator.allocate(i, j)
                self.running_jobs.append(i)

        obs  = self._get_state()
        done = not self.simulator.is_running
        info = { "workload": self. workload }
        return obs, reward, done, info

    def _get_next_job(self):
        jobs = [ i for i in self.simulator.queue if i.id not in self.running_jobs and i.id not in self.waiting_jobs ]
        return jobs[0]

    def _get_reward(self) -> float:
        # TODO Calc reward
        score = 0.

        job_id = self.current_alloc[0][0]
        job = next(filter(lambda x: x.id == job_id, self.simulator.jobs))

        # 1. Waiting time -- Less is better, only on first alloc.
        old_ws = 1./self.current_alloc[0][0] if len(self.current_alloc) == 1 else 0
        score += old_ws

        # 2. Aprox duration change. -- Less is better, only after first alloc.

        # 3. Energy consumption
        hosts  = [ self.simulator.platform.get_host(i) for i in self.current_alloc ]
        assert np.all([ i is not None for i in hosts ])

        #old_ec = np.mean([ host.power for host in hosts[:-1]])
        #delta_ec = hosts[:-1].power - np.mean(old_ec)

        return score

    def _get_state(self) -> Any:
        # TODO: Fix states, only need posible spaces
        state = {}

        posible_spaces = 0
        current_job =  0


        state["queue"] = queue
        state["current_job"] = current_job
        state["platform"] = platform
        state["current_time"] = self.simulator.current_time
        return state

    def _get_spaces(self) -> Tuple[spaces.Dict, spaces.Tuple]:
        nb_hosts = 0
        if self.simulator.is_running:
            nb_hosts   = len(self.hosts)

        queue = spaces.Dict({
            'size': spaces.Discrete(float('inf'))
        })

        platform = spaces.Dict({
            "nb_hosts": spaces.Discrete(nb_hosts),
            "utilization": spaces.Box(low=0, high=float('inf'), shape=(nb_hosts,))
        })

        current_job = spaces.Dict({
            'job':    spaces.Box(low=-1, high=('inf'), shape=(3,)),
            'allocs': spaces.Box(low=0, high=nb_hosts, shape=(nb_hosts,))
        })

        obs_space = spaces.Dict({
            "queue": queue,
            "platform": platform,
            "current_job": current_job
        })

        act_space = spaces.Tuple((
            spaces.Box(low=-1, high=nb_hosts,           shape=()),
            spaces.Box(low=-1, high=self.max_plan_time / self.min_plan_step, shape=())
        ))

        return obs_space, act_space






