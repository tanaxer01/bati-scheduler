from batsim_py import SimulatorHandler, Job

from gym import spaces

from typing import Any, Optional, Tuple
from gridgym.envs.grid_env import GridEnv

class TestEnv(GridEnv):
    def __init__(self) -> None:
        pass

    def _on_job_completed(self, job: Job) -> None:
        pass

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        """
        Perform the specified action and updates the state of the simulation.
        """
        if not self.simulation.is_running or not self.simulator.platform:
            raise error.ResetNeeded("Simulation not running.")

        if 0 > action > len(self.servers):
            raise error.InvalidAction(f"Invalid Action: {action}.")


        # obs, reward, done, info
        return (0, 0, False, {})

    def _get_reward(self) -> float:
        """
        Calculates the reward obtained by the changes in the environment.
        """
        return 0.

    def _get_state(self) -> Any:
        """
        Returns the state of the environment.
        """

        # Resource State
        ## 1. Processing power per Machine
        ## 2. Remaining time per Machine

        ## 3. Energy consumption

        # Task Queue
        ## 1. Resources needed
        ## 2. Estimated work time

        ## 3. Has it any dependencies ?

        return []

    def _get_spaces(self) -> Tuple[spaces.Dict, spaces.Discrete]:
        return (spaces.Dict(), spaces.Discrete())


