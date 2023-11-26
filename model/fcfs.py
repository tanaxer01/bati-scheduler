import json
import numpy as np
from typing import Optional
from .metrics import MonitorsInterface


class FCFSAgent():
    def __init__(self, monitors: Optional[MonitorsInterface] = None):
        self.monitors = monitors

    def act(self, state) -> int:
        return 0

    def test(self, env):
        if self.monitors:
            self.monitors.init_episode(env.simulator, True)

        history = { "score": 0, "steps": 0, "info": None }
        _, done, info = env.reset(), False, {}

        while not done:
            _, reward, done, _, info = env.step( 0  )

            history['score'] += reward
            history['steps'] += 1
            history['info']   = info

        if self.monitors:
            print(f"SAVING IN {self.monitors.save_dir}")
            self.monitors.record()


        print(f"\n[DONE] Score: {history['score']} - Steps: {history['steps']} {len(env.simulator.queue)}")


