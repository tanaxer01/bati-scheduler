
from gym import error, spaces
from typing import Any, Optional, Tuple
from gridgym.envs.grid_env import GridEnv

import numpy as np

INF = float('inf')

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
        #    batsim_py.JobEvent.COMPLETED, self._on_job_completed)
        #self.shutdown_policy = ShutdownPolicy(t_shutdown, self.simulator)

    def step(self, action) -> Tuple[Any, float, bool, dict]:
        assert self.simulator.is_running and self.simulator.platform
        assert 0 <= action <= self.queue_max_len , f"Invalid aciton {action}."

        # action > 0 -> place in list 
        scheduled, reward = False, 0.
        if action > 0:
            # TODO - sort queue.
            sorted_queue = []
            scheduled = True

        if not scheduled:
            reward = self._get_reward()
            self.simulator.proceed_time(self.t_action)

        obs = self._get_state()
        done = not self.simulator.is_running
        info = {"workload": self.workload}
        return (obs, reward, done, info)

    def _get_reward(self) -> float:
        return 0.

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



