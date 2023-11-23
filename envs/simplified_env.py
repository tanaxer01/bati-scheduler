
import xml.etree.ElementTree as ET
from typing import Any, Optional, Tuple
from gridgym.envs.grid_env import GridEnv
from gym import error, spaces
import numpy as np
from torch import wait
from torch._dynamo import run

INF = float('inf')

class SimpleEnv(GridEnv):
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
            raise error.Error("Expected `t_action` acrgument to be greater "
                              f"than zero, got {t_action}.")

        self.queue_max_len = queue_max_len
        self.t_action = t_action

        self.host_speeds = self._get_host_speeds(platform_fn)
        self.prev_action = 0

        super().__init__(platform_fn, workloads_dir, seed,
                         external_events_fn, simulation_time, True,
                         hosts_per_server=hosts_per_server)

    def _get_host_speeds(self, platform_fn: str):
        root = ET.parse(platform_fn).getroot()

        prefixes = { "G": 10e9, "M": 10e6, "K": 10e3 }

        all_speeds = { h.attrib["id"]: h.get("speed") for h in root.iter("host") }
        cmp_speeds = { i: j.split(",")[0][:-1] for i, j in all_speeds.items() if j is not None }
        parsed_speeds = { i: float(j[:-1]) * prefixes[j[-1]] for i, j in cmp_speeds.items() }

        return parsed_speeds

    def step(self, action) -> Tuple[Any, float, bool, dict]:
        assert self.simulator.is_running and self.simulator.platform, f"Simulation not running."

        print("!!", self.simulator.current_time,  action)
        available_hosts = self.simulator.platform.get_not_allocated_hosts()
        posible_jobs = [ j for j in self.simulator.queue if j.res <= len(available_hosts) ]

        job = posible_jobs[int(action)]

        res = [h.id for h in available_hosts[:job.res]]
        self.simulator.allocate(job.id, res)

        reward = self._get_reward()
        """
        if action.item() == 0:
            if self.prev_action != 0:
                reward = self._get_reward()

            # logs {
                print(f"--> {self.simulator.current_time} END OF ASIGNATION ({reward}) {len(self.simulator.queue)} <--")
            else:
                print(".", end="")
            self.prev_action = 0
            # } logs

            self.simulator.proceed_time(self.t_action)
        else:
            available_hosts = self.simulator.platform.get_not_allocated_hosts()
            posible_jobs = [ j for j in self.simulator.queue if j.res <= len(available_hosts) ]

            job = posible_jobs[int(action) - 1]

            ''' logs '''
            if self.prev_action == 0:
                print(f"\n--> {self.simulator.current_time} START OF ASIGNATION <--")
            print(f"\t({int(action.item())}, {self.simulator.current_time - job.subtime}) , {len(self.simulator.queue)}, {job.res}/{len(available_hosts)}")
            self.prev_action = action.item()
            ''' logs '''

            res = [h.id for h in available_hosts[:job.res]]
            self.simulator.allocate(job.id, res)
        """

        obs = self._get_state()
        done = not self.simulator.is_running
        info = { "workload": self.workload, "prev_action": self.prev_action }

        return (obs, reward, done, info)

    def _get_reward(self) -> float:
        running = [j for j in self.simulator.jobs if j.is_running]
        queue = [j for j in self.simulator.queue ]

        ## Waiting time
        wait_queue = [ self.simulator.current_time - j.subtime for j in queue ]
        score_queue = [ 100 * int(i > 1) for i in wait_queue ]

        #wait_running = np.array([ -1 * (j.waiting_time >= 1.) if j.waiting_time else 0 for j in running])
        #wait_queue = sum([ j.waiting_time if j.waiting_time else 0 for j in queue ])

        #return wait_running.mean() if wait_running.shape[0] != 0 else 0.
        return -1 * sum(score_queue)

    def _get_state(self) -> Any:
        nb_hosts = sum( 1 for _ in self.simulator.platform.hosts )

        available_hosts = self.simulator.platform.get_not_allocated_hosts()
        posible_jobs = [ j for j in self.simulator.queue if j.res <= len(available_hosts) ]

        while self.simulator.is_running:
            self.simulator.proceed_time(self.t_action)
            available_hosts = self.simulator.platform.get_not_allocated_hosts()
            posible_jobs = [ j for j in self.simulator.queue if j.res <= len(available_hosts) ]

            if len(posible_jobs) != 0:
                break


        # Queue status
        jobs = np.zeros( (len(posible_jobs), 3) )
        if len(posible_jobs) != 0:
        ## Subtime
            jobs[:,0] = [ j.subtime for j in posible_jobs ]
        ## Resources
            jobs[:,1] = [ j.res     for j in posible_jobs ]
        ## Walltime
            jobs[:,2] = [ j.walltime if j.walltime else -1 for j in posible_jobs ]
        ## Flops
            # jobs[:,3]

        queue = { "size": len(self.simulator.queue), "jobs": jobs }

        # Platform status
        hosts = np.zeros( (nb_hosts, 2) )
        for h in self.simulator.platform.hosts:
        ## Status
            hosts[:,0] = h.state.value
        ## Speed
            hosts[:,1] = self.host_speeds[h.name]

        platform = { "nb_hosts": len(available_hosts), "hosts": hosts }

        # Full state
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
