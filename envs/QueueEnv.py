import numpy as np

from gym import error, spaces
import xml.etree.ElementTree as ET
from batsim_py import SimulatorHandler, SimulatorEvent, HostEvent, JobEvent
from batsim_py.resources import Host
from gridgym.envs.grid_env import GridEnv
from typing import Any, Optional, Tuple, Dict


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

        self.simulator.subscribe(
                JobEvent.SUBMITTED, self._on_job_submitted)

        #self.simulator.subscribe(
        #    batsim_py.JobEvent.COMPLETED, self._on_job_completed)
        self.shutdown_policy = ShutdownPolicy(t_shutdown, self.simulator)

        root = ET.parse(platform_fn).getroot()

        prefixes = { "G": 10e9, "M": 10e6, "K": 10e3 }
        to_int = lambda x: float(x[:-1]) * prefixes[x[-1]]

        self.host_speeds = { h.attrib["id"]: to_int(h.get("speed").split(",")[0][:-1])
                                for h in root.iter("host") }

    def _on_job_submitted(self, job):
        pass



    def step(self, action) -> Tuple[Any, float, bool, dict]:
        assert self.simulator.is_running and self.simulator.platform
        assert 0 <= action <= self.queue_max_len , f"Invalid aciton {action}."

        # action > 0 -> place in list
        scheduled, reward = False, 0.
        if action > 0:
            job = self.simulator.queue[int(action)-1]
            available = self.simulator.platform.get_not_allocated_hosts()
            if job.res <= len(available):
                res = [h.id for h in available[:job.res]]
                self.simulator.allocate(job.id, res)
                scheduled = True

        if not scheduled:
            reward = self._get_reward()
            self.simulator.proceed_time(self.t_action)

        obs = self._get_state()
        done = not self.simulator.is_running
        info = {"workload": self.workload}
        return (obs, reward, done, info)

    def _get_reward(self) -> float:
        print(">", [ i for i in self.simulator.agenda ])



        nb_hosts = sum( 1 for _ in self.simulator.platform.hosts )
        # QoS
        wait_t = sum(
                     1./j.walltime if j.walltime else 1 for j in self.simulator.queue[:self.queue_max_len] ) / nb_hosts

        # Energy waste
        energy_score = sum( 1. for h in self.simulator.platform.hosts if h.is_idle )
        energy_score /= nb_hosts

        # Utilization
        u = sum(1. for h in self.simulator.platform.hosts if h.is_computing)
        u /= nb_hosts

        return u - energy_score - wait_t

    def _get_state(self) -> Any:
        # Queue
        queue = {
            "size": len(self.simulator.queue),
            "jobs": np.zeros( (self.queue_max_len, 3) )
        }

        # TODO - Add estimated length
        for i, job in enumerate(self.simulator.queue[:self.queue_max_len]):
            wall = -1 if job.walltime is None else job.walltime
            queue["jobs"][i] = [
                job.subtime,
                job.res,
                wall
            ]

        # Platform
        nb_hosts = sum( 1 for _ in self.simulator.platform.hosts)
        platform = {
            "nb_hosts": nb_hosts,
            "status": np.array(
                [h.state.value for h in self.simulator.platform.hosts ]),
            "agenda": np.zeros( (nb_hosts, 2) )
        }

        for i in self.simulator.jobs:
            if not i.is_running:
                continue

            if i.allocation == None:
                continue

            for h_id in i.allocation:
                platform["agenda"][h_id] = [
                    i.start_time,
                    i.walltime or -1
                ]

        state = {
            "queue": queue,
            "platform": platform,
            "current_time": self.simulator.current_time
        }

        return state

    def _get_spaces(self):
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



