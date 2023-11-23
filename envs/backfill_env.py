import numpy as np
from gym import error, spaces
from gridgym.envs.grid_env import GridEnv
from typing import Any, Optional, Tuple

from batsim_py import JobEvent


INF = float('inf')

class BackfillEnv(GridEnv):
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

        self.prev_action = 0

        super().__init__(platform_fn, workloads_dir, seed,
                         external_events_fn, simulation_time, True,
                         hosts_per_server=hosts_per_server)

    def step(self, action) -> Tuple[Any, float, bool, dict]:
        assert self.simulator.is_running and self.simulator.platform, f"Simulation not running."

        reward = 0.
        if action.item() == 0:
            reward = self._get_reward()

            ''' logs '''
            if self.prev_action != 0:
                print(f"--> {self.simulator.current_time} END OF ASIGNATION ({reward}) {len(self.simulator.queue)} <--")
            else:
                print(".", end="")
            self.prev_action = 0
            ''' logs '''

            self.simulator.proceed_time(self.t_action)
        else:
            job = self.simulator.queue[int(action) - 1]

            ''' logs '''
            if self.prev_action == 0:
                print(f"\n--> {self.simulator.current_time} START OF ASIGNATION <--")
            print(f"\t({int(action.item())}, {self.simulator.current_time - job.subtime}) , {len(self.simulator.queue)}")
            self.prev_action = action.item()
            ''' logs '''

            # Currently available hosts.
            available = self.simulator.platform.get_not_allocated_hosts()

            # Currently reserved hosts.
            next_releases = sorted(self.simulator.agenda, key=lambda a: a.release_time)
            candidates = [r.host.id for r in next_releases ]

            if job.res <= len(available):
                allocation = [h.id for h in available[:job.res]]
                #print(f"| 1: {job.res}/{len(available)}")

                self.simulator.allocate(job.id, allocation)
            else:

                #allocation =  [h.id for h in available]
                #allocation += candidates[:job.res - len(allocation)]
                allocation = candidates[:job.res]
                #print(f"| 2: {job.res}")

                print("!!!!!!!!!!!!!!!!!!!!!!!!!", next_releases[job.res-1].release_time)
                self.simulator.allocate(job.id, allocation)


        obs = self._get_state()
        done = not self.simulator.is_running
        info = {"workload": self.workload}
        return (obs, reward, done, info)

    def _get_reward(self) -> float:
        wait_running = np.array([ j.waiting_time if j.waiting_time else 0 for j in self.simulator.jobs if j.is_running ])

        return sum([ -100 * i//10 for i in wait_running ])

    def _get_state(self) -> Any:
        nb_hosts = sum( 1 for _ in self.simulator.platform.hosts )

        posible_jobs = [ j for j in self.simulator.queue ]

        # Queue status
        jobs = np.zeros( (len(posible_jobs), 3) )
        if len(posible_jobs) != 0:
        ## Subtime
            jobs[:,0] = [ j.subtime for j in posible_jobs ]
        ## Resources
            jobs[:,1] = [ j.res     for j in posible_jobs ]
        ## Walltime
            jobs[:,2] = [ j.walltime if j.walltime else -1 for j in posible_jobs ]

        queue = { "size": len(self.simulator.queue), "jobs": jobs }

        # Platform status
        hosts = np.full((nb_hosts, 2), -1)

        ## Host status, if host is reserved check when it is going to be free.
        for res in self.simulator.agenda:
            hosts[res.host.id,0] = res.release_time

        for host in self.simulator.platform.get_not_allocated_hosts():
            hosts[host.id,0] = 0

        platform = { "nb_hosts": nb_hosts, "hosts": hosts }

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

