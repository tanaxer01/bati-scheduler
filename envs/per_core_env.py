
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

        print("step: time", self.simulator.current_time)
        for i in self.freeSpaceList.items:
            print(i, i.full, i.started)

        self._test_schedule()

        # 1. No jobs arrived
        if len(self.simulator.queue) == 0 or (self._get_next_job() == None and self.current_job == None):
            self.simulator.proceed_time(self.t_action)
            self.freeSpaceList.update_time(self.simulator.current_time)


            obs  = self._get_state()
            done = not self.simulator.is_running
            info = { "workload": self.workload }
            return obs, 0., done, info

        # 2. Asign current_job
        if self.current_job == None:
            self.current_job = self._get_next_job() 

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
                self.freeSpaceList.add_reservation({int(action[0])}, self.current_job.id, action[1], action[1] + duration)
                print("ADD", self.freeSpaceList.items)
            else:
                duration = 1000 if self.current_job.walltime is None else self.current_job.walltime
                self.freeSpaceList.update_reservation(int(action[0]), self.current_job.id, action[1], action[1] + duration)
                print("UPDT", self.freeSpaceList.items)

            if len(self.current_alloc) == self.current_job.res:
                for i in self.freeSpaceList.generator():
                    if i.job == self.current_job.id:
                        i.full = True
            reward = self._get_reward()

        print(self.simulator.current_time, "...", action, self.current_job, self.current_alloc)
        # TODO: Test Scheduling job
        #if scheduled: 
        #    self._schedule_jobs()

        obs  = self._get_state()
        done = not self.simulator.is_running
        info = { "workload": self. workload }
        return obs, reward, done, info

    def _test_schedule(self):
        to_be_handled = [ (i.job, i.hosts) for i in self.freeSpaceList.items if i.start == self.freeSpaceList.curr_time and not i.started and i.full ]
        print(">>>", to_be_handled)
        print(">>>", [ i for i in self.freeSpaceList.items if i.full ])
        print(">>>", [ i for i in self.freeSpaceList.items ])

        for i, j in to_be_handled:
            self.freeSpaceList.start_job(i)
            self.simulator.allocate(str(i), list(j))
            

    def _schedule_jobs(self):
        # TODO: handle dependencies
        # Get jobs_ids that should be assigned
        to_be_handled = { (i.job, i.hosts) for i in self.freeSpaceList.items if i.start == self.freeSpaceList.curr_time and not i.started }
        print(">>>", to_be_handled)
        for i, j in to_be_handled:
            self.simulator.allocate(str(i), list(j))
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
        try:
            hosts  = [ self.simulator.platform.get_host(int(i)) for i,_ in self.current_alloc ]
        except Exception as e:
            print( self.current_alloc )
            return
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

        # 1. Si no hay espacios elegidos, listamos todos
        if len(self.current_alloc) == 0:
            posible_spaces = self.freeSpaceList.get_posible_spaces(duration, self.current_job.res)
        else:
            pass
        # 2. Si ya hay al menos 1, damos solo los que estan en el mismo rango horario
            #cores = {i[0] for i in self.current_alloc}
            posible_spaces = self.freeSpaceList.get_posible_spaces(duration, self.current_job.res)
            #posible_space = filter(lambda x: x.start == self.current_alloc[0][1] and x.end == self.current_alloc[0][1] + duration, posible_spaces)

            #print(list(posible_space))

            #posible_spaces = self.freeSpaceList._check_range(self.current_alloc[0][1], self.current_alloc[0][1]+ duration)
            #print(self.current_alloc, duration)
            #posible_spaces = list(filter(lambda x: len(x.hosts.intersection(cores)) == 0 , posible_spaces))

        #print("state: ", [ i for i in posible_spaces if 0 in i.hosts ] )

        space_stats = np.zeros((len(posible_spaces), 4))
        # 0 . core_id
        # 1 . wait_time
        # 2 . Energy consum
        # 3 . Estimated_exec_time
        for i, j in enumerate(posible_spaces):
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

        if self.current_job != None:
            cores = self.current_job.res
            duration = 1000 if self.current_job.walltime == None else int(self.current_job.walltime)
            spaces_cant = len(list(self.freeSpaceList.get_posible_spaces(cores, duration)))

        posible_spaces = spaces.Box(low=0, high=self.max_plan_time, shape=(spaces_cant, 4))
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

