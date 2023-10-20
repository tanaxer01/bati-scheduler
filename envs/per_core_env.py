
from typing import Any, Optional, Tuple
import numpy as np

from batsim_py import SimulatorHandler
from batsim_py.jobs import Job
from batsim_py.events import SimulatorEvent

from gridgym.envs.grid_env import GridEnv
from gym import error, spaces

from schedulers.FreeSpaces import FreeSpace, JobAgenda

class PerCoreEnv(GridEnv):
    def __init__(self, platform_fn: str,
                 workloads_dir:     str,
                 t_action:   int = 1,
                 t_shutdown: int = 0,
                 hosts_per_server: int = 1,
                 queue_max_len:    int = 20,
                 seed: Optional[int] = None,
                 max_plan_time: int = 20,
                 min_plan_step: int = 1,
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

    def _on_simulation_begins(self, _):
        nb_cores = len(list(self.simulator.platform.hosts))
        self.freeSpaceList = JobAgenda(nb_cores, self.max_plan_time, self.min_plan_step)
        # TODO: Read speeds from
        #self._read_core_speeds()

    def _on_job_completed(self, job: Job) -> None:
        pass

    def step(self, action: Tuple[int, float]) -> Tuple[Any, float, bool, dict]:
        if not self.simulator.is_running or not self.simulator.platform:
            raise error.ResetNeeded("Simulation not running.")
        '''
        print("step: time", self.simulator.current_time)

        # 1. No jobs on queue
        if len(self.simulator.queue) == 0:
        # if len(self.simulator.queue) == 0 or (self._get_next_job() == None and self.current_job == None):
            self.simulator.proceed_time(self.t_action)
            self.freeSpaceList.update_time(self.simulator.current_time)

            obs  = self._get_state()
            done = not self.simulator.is_running
            info = { "workload": self.workload }
            return obs, 0., done, info

        # 2. No current_job assigned
        if self.current_job == None:
            self.current_job = self._get_next_job()

            # TODO - What happends if next_job is None
            print(f"step: new current job {self.current_job}", action)

            obs  = self._get_state()
            done = not self.simulator.is_running
            info = { "workload": self.workload }
            return obs, 0., done, info

        # 3. Current job is ready to be assigned
        print(self.simulator.current_time, "...", action, self.current_job, self.current_job.res, self.current_alloc)
        scheduled, reward = False, 0.
        if action[0] == int(action[1]) == -1 and action:
            allocs = {i[0] for i in self.current_alloc}
            print("A", allocs, len(allocs), self.current_job.res)
            assert len(allocs) == self.current_job.res

            # TODO: Handle wall == 0
            assert self.current_job.walltime != None and self.current_job.walltime != -1

            start = self.current_alloc[0][1]
            stop  = start + self.current_job.walltime

            #self.freeSpaceList.add_reservation(allocs, self.current_job.id, start, stop)
            #self.freeSpaceList.update_reservation(action[0], self.current_job.id, start, stop)
            reward = self._get_reward()

            # Reset current_alloc
            self.current_job = None
            self.current_alloc = []
            scheduled = True
        else:
            assert tuple(action) not in self.current_alloc, f"found {action}"
            action = (action[0], action[1] + self.freeSpaceList.curr_time)

            self.current_alloc.append(action)
            if len(self.current_alloc) == 1:
                duration = 1000 if self.current_job.walltime is None else self.current_job.walltime
                self.freeSpaceList.add_reservation({int(action[0])}, self.current_job.id, int(action[1]), int(action[1] + duration))
                print("ADD", self.freeSpaceList.items)
            else:
                duration = 1000 if self.current_job.walltime is None else self.current_job.walltime
                self.freeSpaceList.update_reservation(int(action[0]), self.current_job.id, action[1], action[1] + duration)
                print("UPDT", self.freeSpaceList.items)

            print("--->", self.freeSpaceList.items)
            print("--->", self.current_alloc)


            if len(self.current_alloc) == self.current_job.res:
                for i in self.freeSpaceList.generator():
                    if i.job == self.current_job.id:
                        i.full = True
            reward = self._get_reward()

        print(self.simulator.current_time, "...", action, self.current_job, self.current_alloc)
        # TODO: Test Scheduling job
        #if scheduled:
        #    self._schedule_jobs()
        self._test_schedule()
        '''

        print(self.simulator.current_time, "step:", action)

        if self.current_job == None: # No hay trabajo seleccionado
            print("BBB")
            self.current_job = self._get_next_job()
            if self.current_job:
                print(self.simulator.current_time, "step: current job ==", self.current_job, self.current_job.res)

        if len(self.simulator.queue) == 0 or self.current_job == None: # No hay trabajos que agendar
            print("BBBB")
            self._test_schedule()
            self._time_step()
            print(self.simulator.current_time, "step: queue len == 0")

            return self._get_state(), 0., False, {}

        assert self.current_job != None, "Ver que pasa si _get_next_job retorna null"

        duration = int(1000 if self.current_job.walltime is None else self.current_job.walltime)
        if len(self.freeSpaceList.get_posible_spaces(self.current_job.res, duration)) == 0:
            self._test_schedule()
            self._time_step()

            return self._get_state(), 0., False, {}

        print("AAAA", self.current_job, self.current_job.res, self.current_alloc, len(self.simulator.queue))
        print("AAAA", self.freeSpaceList.get_posible_spaces(self.current_job.res, duration))

        reward = 0
        if action == (-1, -1) and self.current_job.res == len(self.current_alloc):
            # Trabajo esta listo
            print(self.simulator.current_time, "step: all allocs should be reserved")

            allocs = {i[0] for i in self.current_alloc}
            print(self.current_alloc, self.current_job.res, self.current_job.id)
            assert len(allocs) == self.current_job.res

            # TODO: Handle wall == 0
            assert self.current_job.walltime != None and self.current_job.walltime != -1
            start = self.current_alloc[0][1]
            stop  = start + self.current_job.walltime

            print("------")
            self.freeSpaceList.add_reservation(allocs, self.current_job.id, start, stop)
            self.current_job = None
            self.current_alloc = []

            reward = self._get_reward()
        elif action != (-1, -1) :
            # Agregar acción a self.curernt_allocs
            action = (int(action[0]), action[1]+self.freeSpaceList.curr_time)

            for i in self.current_alloc:
                assert i[0] != action[0]
                assert i[1] == action[1]


            self.current_alloc.append(action)
            print(self.simulator.current_time, "step: current allocs ==", self.current_alloc)
            reward = self._get_reward()

        print("DDDD")
        # Try schedule
        self._test_schedule()
        for i in self.freeSpaceList.items:
            if len(i.hosts) == 0:
                print(i.job)

        obs  = self._get_state()
        done = not self.simulator.is_running
        info = { "workload": self. workload }
        return obs, reward, done, info

    def _time_step(self):
        self.simulator.proceed_time(self.t_action)
        self.freeSpaceList.update_time(self.simulator.current_time)

    def _test_schedule(self):
        current_time = self.freeSpaceList.curr_time
        jobs = filter(lambda x: x.start == current_time and not x.started and x.full, self.freeSpaceList.items)

        for i in set(jobs):
            print("scheduled", i.job)
            self.freeSpaceList.start_job(i.job)
            self.simulator.allocate(str(i.job), [int(j) for j in i.hosts])

    def _schedule_jobs(self):
        # TODO: handle dependencies
        # Get jobs_ids that should be assigned
        to_be_handled = { (i.job, i.hosts) for i in self.freeSpaceList.items if i.start == self.freeSpaceList.curr_time and not i.started }
        print(">>>", to_be_handled)
        for i, j in to_be_handled:
            self.simulator.allocate(str(i), [ int(k) for k in j])
            self.freeSpaceList.start_job(i)

        # TODO: si de un time step a otro se nos paso una tarea echarla a andar o tirar error?

        # Update running_jobs state
        self.simulator.proceed_time(self.t_action)
        curr_time = self.simulator.current_time
        print(curr_time)
        self.freeSpaceList.update_time(curr_time)

    def _get_next_job(self):
        jobs = [ i for i in self.simulator.queue if i.id not in self.freeSpaceList.jobs ]
        return jobs[0] if len(jobs) > 0 else None

    def _get_reward(self) -> float:
        # TODO Calc reward
        score = 0.

        # 1. Waiting time -- Less is better, only on first alloc.
        #old_ws = 1./self.current_alloc[0][0] if len(self.current_alloc) == 1 else 0
        # TODO fix
        old_ws = -1 * self.current_alloc[0][0] if len(self.current_alloc) == 1 else 0
        score += old_ws

        # 2. Aprox duration change. -- Less is better, only after first alloc.

        # 3. Energy consumption
        hosts  = [ self.simulator.platform.get_host(int(i)) for i,_ in self.current_alloc ]
        assert np.all([ i is not None for i in hosts ])

        #old_ec = np.mean([ host.power for host in hosts[:-1]])
        #delta_ec = hosts[:-1].power - np.mean(old_ec)

        return score

    def _get_state(self) -> Any:
        # TODO: Fix states, only need posible spaces
        state = {}

        if self.current_job is None:
            print("state: no current job, time --", len(list(self.simulator.queue)), self.simulator.current_time)
            return { "posible_spaces": [], "current_job": [-1, -1, -1] }

        duration = int(1000 if self.current_job.walltime is None else self.current_job.walltime)

        ## Posible spaces

        posible_spaces = self.freeSpaceList.get_posible_spaces(duration, self.current_job.res)
        if len(self.current_alloc) != 0:
            # 2. Si ya hay al menos 1 alloc, damos solo los que estan en el mismo rango horario
            start, end = self.current_alloc[0][1], self.current_alloc[0][1] + duration
            cores = {i[0] for i in self.current_alloc}

            posible_spaces = list(filter(lambda x: x.start == start and x.end == end, posible_spaces))
            for i in posible_spaces:
                i.hosts -= cores

        all_spaces = []
        for i in posible_spaces:
            for j in i.hosts:
                all_spaces.append(FreeSpace({j}, i.start, i.end))

        ## Spaces stats

        space_stats = np.zeros((len(all_spaces), 4))
        # 0 . core_id
        # 1 . wait_time
        # 2 . Energy consum
        # 3 . Estimated_exec_time
        for i, j in enumerate(all_spaces):
            #assert len(j.hosts) == 1
            # TODO: How to choose first host on first assig
            space_stats[i,0] = list(j.hosts)[0]
            space_stats[i,1] = j.start - self.freeSpaceList.curr_time
            space_stats[i,2] = self.simulator.platform.get_host(list(j.hosts)[0]).power
            # TODO usar tiempo estimado con velocidad y no end - start
            space_stats[i,3] = j.end - j.start

        #current_job = [ self.current_job.res, self.current_job.walltime, len(self.current_alloc) ]
        job_stats = np.zeros(3)
        # 0 . wait_time
        # 1 . expected_compute
        # 2 . cant of cores alloced
        # 3 . cant of cores needed
        job_stats[0] = -1 if len(self.current_alloc) == 0 else self.current_alloc[0][0]
        job_stats[1] = len(self.current_alloc)
        job_stats[2] = self.current_job.res

        state["posible_spaces"] = space_stats
        state["current_job"]    = job_stats

        return state

    def _get_spaces(self) -> Tuple[spaces.Dict, spaces.Tuple]:
        nb_hosts = spaces_cant = 0
        if self.simulator.is_running:
            nb_hosts = len(list(self.simulator.platform.hosts))

        all_cant  = 0
        if self.current_job != None:
            cores = self.current_job.res
            duration = 1000 if self.current_job.walltime == None else int(self.current_job.walltime)
            spaces_cant = self.freeSpaceList.get_posible_spaces(cores, duration)

            for i in spaces_cant:
                assert len(i.hosts) != 0
                for j in i.hosts:
                    all_cant += 1

        posible_spaces = spaces.Box(low=0, high=self.max_plan_time, shape=(all_cant, 4))
        current_job    = spaces.Box(low=0, high=self.max_plan_time, shape=(3,))

        obs_space = spaces.Dict({
            'posible_spaces': posible_spaces,
            'current_job': current_job
        })

        act_space = spaces.Tuple((
            spaces.Box(low=1, high=nb_hosts,           shape=()),
            spaces.Box(low=-1, high=self.max_plan_time / self.min_plan_step, shape=())
        ))

        return obs_space, act_space

