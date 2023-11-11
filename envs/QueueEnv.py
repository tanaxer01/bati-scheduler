import numpy as np

from gym import error, spaces
import xml.etree.ElementTree as ET
from gridgym.envs.grid_env import GridEnv, batsim_py
from typing import Any, Optional, Tuple, Dict

from batsim_py.jobs import Job
from batsim_py.resources import Host
from batsim_py import SimulatorHandler, SimulatorEvent, HostEvent, JobEvent


INF = float('inf')

class ShutdownPolicy():
    def __init__(self, timeout: int, simulator: SimulatorHandler):
        super().__init__()
        self.timeout = timeout
        self.simulator = simulator
        self.idle_servers: Dict[int, float] = {}

        self.simulator.subscribe(
            HostEvent.STATE_CHANGED, self._on_host_state_changed)
        self.simulator.subscribe(
            SimulatorEvent.SIMULATION_BEGINS, self._on_sim_begins)

    def shutdown_idle_hosts(self, *args, **kwargs):
        hosts_to_turnoff = []
        for h_id, start_t in list(self.idle_servers.items()):
            if self.simulator.current_time - start_t >= self.timeout:
                hosts_to_turnoff.append(h_id)
                del self.idle_servers[h_id]

        if hosts_to_turnoff:
            self.simulator.switch_off(hosts_to_turnoff)

    def _on_host_state_changed(self, host: Host):
        if host.is_idle:
            if host.id not in self.idle_servers:
                self.idle_servers[host.id] = self.simulator.current_time
                t = self.simulator.current_time + self.timeout
                self.simulator.set_callback(t, self.shutdown_idle_hosts)
        else:
            self.idle_servers.pop(host.id, None)

    def _on_sim_begins(self, _):
        self.idle_servers.clear()
        for h in self.simulator.platform.hosts:
            if h.is_idle:
                self.idle_servers[h.id] = self.simulator.current_time
                t = self.simulator.current_time + self.timeout
                self.simulator.set_callback(t, self.shutdown_idle_hosts)

class QueueEnv(GridEnv):
    def __init__(self,
                 platform_fn: str,
                 workloads_dir: str,
                 t_action: int = 1,
                 t_shutdown: int = 0,
                 hosts_per_server: int = 1,
                 queue_max_len: int = 20,
                 seed: Optional[int] = None,
                 external_events_fn: Optional[str] = None,
                 simulation_time: Optional[float] = None) -> None:

        if t_action < 0:
            raise error.Error("Expecter `t_action` argument to be greater "
                              f"than zero, got {t_action}.")


        self.queue_max_len = queue_max_len
        self.t_action = t_action

        super().__init__(platform_fn, workloads_dir, seed,
                         external_events_fn, simulation_time, True,
                         hosts_per_server=hosts_per_server)

        #self.simulator.subscribe(
        #        JobEvent.SUBMITTED, self._on_job_submitted)

        self.simulator.subscribe(JobEvent.COMPLETED, self._on_job_completed)
        self.shutdown_policy = ShutdownPolicy(t_shutdown, self.simulator)

        root = ET.parse(platform_fn).getroot()

        prefixes = { "G": 10e9, "M": 10e6, "K": 10e3 }
        to_int = lambda x: float(x[:-1]) * prefixes[x[-1]]

        self.host_speeds = { h.attrib["id"]: to_int(h.get("speed").split(",")[0][:-1])
                                for h in root.iter("host") }

        self.running_jobs  = dict()
        self.completed_jobs = set()

    def _on_job_completed(self, job):
        self.completed_jobs.add(job.id)
        self.running_jobs.pop(job.id)

    def step(self, action) -> Tuple[Any, float, bool, dict]:
        assert self.simulator.is_running and self.simulator.platform

        if 0 > action > self.queue_max_len:
            raise error.InvalidAction(f"Invalid Action: {action}/{self.queue_max_len}")


        # action > 0 -> place in list
        scheduled, reward = False, 0.
        if action > 0:
            job = self.simulator.queue[int(action)-1]

            available = self.simulator.platform.get_not_allocated_hosts()
            if job.res <= len(available):
                res = [h.id for h in available[:job.res]]
                self.simulator.allocate(job.id, res)
                self.running_jobs[job.id] = [h.name for h in available[:job.res]]
                scheduled = True

        if not scheduled:
            reward = self._get_reward()
            self.simulator.proceed_time(self.t_action)

        obs = self._get_state()
        done = not self.simulator.is_running
        info = {"workload": self.workload}
        return (obs, reward, done, info)

    '''
    def _get_job_obj(self, id) -> Job:
        jobs = [ i for i in self.simulator.jobs if i.id == id ]
        #assert len(jobs) == 1, "If job is in self.running_jobs it can't be None"

        if len(jobs):
            return jobs[0]
    '''

    def _get_reward(self) -> float:

        # 1. Get running jobs
        jobs = filter(lambda x: x.id in self.running_jobs, self.simulator.jobs)

        # TODO - Why did I filter by dependencies xd.
        '''
        # Computing jobs
        jobs = list(filter(lambda x: hasattr(x.profile, "cpu"), jobs))

        # Check dependencies
        dependencies = map(lambda x: x.metadata["dependencies"]
                                if "dependencies" in x.metadata else [], jobs)
        deps_status  = [ sum(j not in self.completed_jobs for j in i) == 0
                                                        for i in dependencies ]

        jobs = [ i for i, j in zip(jobs, deps_status) if j == True ]
        '''

        # 2. Calculating reward
        job_slowest_res = [ min(map(lambda x: self.host_speeds[x], i))
                                        for i in self.running_jobs.values() ]

        expected_turnaround = [ i.profile.cpu / j for i,j in zip(jobs, job_slowest_res) ]
        waiting_time = [ i.waiting_time if i.waiting_time != None else 0 for i in jobs ]

        r = [ 1 / np.log(i + j) for i, j in zip(expected_turnaround, waiting_time) ]


        return sum(r)

    def _get_state(self) -> Any:
        nb_hosts = sum( 1 for _ in self.simulator.platform.hosts)

        # 1. Get ready to schedule jobs.
        ## Jobs that fit in the platform.
        valid_jobs = filter(lambda j: j.res <= nb_hosts, self.simulator.queue)

        '''
        ## Jobs that have all their dependencies fulfilled
        dependencies = map(lambda x: x.metadata["dependencies"]
                            if "dependencies" in x.metadata else [], valid_jobs)
        dep_status = [ sum(j not in self.completed_jobs for j in i) == 0
                                                        for i in dependencies ]
        valid_jobs = [ i for i, j in zip(valid_jobs, dep_status) if j == True ]

        if len(valid_jobs) > self.queue_max_len:
            valid_jobs = valid_jobs[:self.queue_max_len]
        '''

        valid_jobs = list(valid_jobs)
        jobs = np.zeros((len(valid_jobs), 4))

        if jobs.shape[0] != 0:
            # Job subtime
            jobs[:,0] = [ j.subtime for j in valid_jobs ]
            # Job resources
            jobs[:,1] = [ j.res     for j in valid_jobs ]
            # Job wall
            jobs[:,2] = [ -1 if j.walltime is None else j.walltime
                                    for j in valid_jobs ]
            # Job flops
            jobs[:,3] = [ j.profile.cpu if hasattr(j.profile, "cpu") else -1
                                    for j in valid_jobs ]


        queue = { "size": len(self.simulator.queue), "jobs": jobs }

        # 3. Build Platform State
        hosts = np.zeros((nb_hosts, 3))

        if nb_hosts != 0:
            # TODO - Revisar forma en que se estan mandando estos valores
            #        (ej. Mandar speed relativa a la más lenta o algo asi)

            # Host status
            hosts[:,0] = [ h.state.value for h in self.simulator.platform.hosts ]
            # Host speed
            hosts[:,1] = [ self.host_speeds[h.name]
                                         for h in self.simulator.platform.hosts ]
            # Host energy
            hosts[:,2] = [ h.get_pstate_by_type(batsim_py.resources.PowerStateType.COMPUTATION)[0].watt_full
                                         for h in self.simulator.platform.hosts ]

        platform = { "nb_hosts": nb_hosts, "hosts": hosts }

        state = {
            "queue": queue,
            "platform": platform,
            "current_time": self.simulator.current_time
        }

        return state

    def _get_spaces(self):
        # TODO - update dis
        nb_hosts, agenda_shape, status_shape = 0, (), ()
        if self.simulator.is_running:
            nb_hosts = sum(1 for _ in self.simulator.platform.hosts)
            status_shape = (nb_hosts,  )
            agenda_shape = (nb_hosts, 3)

        # Queue
        queue = spaces.Dict({
            "size": spaces.Discrete(INF),
            "jobs": spaces.Box(low=-1, high=INF, shape=(self.queue_max_len, 5))
        })

        # Platform
        platform = spaces.Dict({
            "nb_hosts": spaces.Discrete(nb_hosts),
            "agenda": spaces.Box(low=-1, high=INF, shape=agenda_shape),
            "status": spaces.Box(low= 0, high= 7,  shape=status_shape)
        })

        obs_space = spaces.Dict({
            "queue": queue,
            "platform": platform,
            "current_time": spaces.Box(low=0, high=INF, shape=())
        })

        action_space = spaces.Discrete(self.queue_max_len+1)
        return obs_space, action_space



