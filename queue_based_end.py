from typing import Any, Optional, Tuple

from gridgym.envs.grid_env import GridEnv

from gym import error, spaces

class QueueBasedEnv(GridEnv):
    def __init__(self, platform_fn: str, workloads_dir: str, t_action: int= 1, t_shutdown: int = 0,
                 hosts_per_server: int = 1, queue_max_len: int = 20, seed: Optional[int] = None,
                 external_events_fn: Optional[str] = None, simulation_time: Optional[float] = None) -> None:

        if t_action < 0:
            raise error.Error(f'Expected `t_action` argument to be greater than zero, got {t_action}.')

        self.queue_max_len = queue_max_len
        self.t_action = t_action

        self.waiting_tasks     = []
        self.not_runable_tasks = []

        super().__init__(platform_fn, workloads_dir, seed, external_events_fn, simulation_time)

        #self.shutdown_policy =

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        if not self.simulator.is_running or not self.simulator.platform:
            raise error.ResetNeeded("Simulation not running.")

        if self.queue_max_len < action < 0:
            raise error.InvalidAction(f'Invalid action {action}.')

        scheduled, reward = False, 0.
        while len(self.simulator.queue) != 0:
            break

        return {}, reward, scheduled, {}

    def _get_reward(self)-> float:
        return 0

    def _get_state(self) -> Any:
        queue : dict = {}
        queue["size"] = len(self.simulator.queue) # Queue Size


        platform : dict = {}



        return {}

    def _get_space(self) -> Tuple[spaces.Dict, spaces.Discrete]:
        return spaces.Dict(), spaces.Discrete()

