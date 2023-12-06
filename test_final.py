from logging import shutdown
from typing import Optional
import batsim_py

from envs.simple_env        import SimpleEnv
from envs.backfill_env      import BackfillEnv
from envs.shutdown_policies import TimeoutPolicy

from model.metrics  import MonitorsInterface

from model.agent    import Agent
from model.fcfs     import FCFSAgent

# Meanwhile
STATE_SIZE = 6

def init_monitors(name, dir="/data/expe-out"):
    return MonitorsInterface(
            name = name,
            save_dir =dir,
            monitors_fns = [
                batsim_py.monitors.JobMonitor,
                batsim_py.monitors.SimulationMonitor,
                batsim_py.monitors.SchedulerMonitor,
                batsim_py.monitors.ConsumedEnergyMonitor,
                batsim_py.monitors.HostStateSwitchMonitor
            ])

def train_model(env_fn: type, platform_fn: str, workload_fn: str):
    print("[TRAIN]")
    env = env_fn(platform_fn=platform_fn, workload_fn=workload_fn, shutdown_policy = lambda s: TimeoutPolicy(5,s))
    agent = Agent(STATE_SIZE)

    agent.play(env, 10, True)

def test_model(name:str, env_fn: type, platform_fn: str, test_fn: str, train_fn: Optional[str]= None):
    if train_fn != None:
        train_model(env_fn, platform_fn, train_fn)

    print("[TEST]")
    env = env_fn(platform_fn=platform_fn, workload_fn=test_fn, shutdown_policy = lambda s: TimeoutPolicy(5,s))

    monitors = init_monitors(name)
    agent = Agent(STATE_SIZE, monitors=monitors)
    agent.load("/data/expe-out/network.chkpt/netwrk")

    agent.test(env)

plat  = "/data/platforms/FatTree/fat_tree_heterogeneous.xml"
plat2  = "/data/platforms/FatTree/generated.xml"
train = "/data/workloads/training"

plat4  = "/data/platforms/FatTree/fat_tree_4.xml"
train4 = "/data/workloads/training4"

test  = "/data/workloads/test/w.json"

### DQN4fat_tree_heterogeneous
test_model("DQN3", SimpleEnv, plat2, test, train)
#test_model("DQN3", SimpleEnv, plat2, test)

### BACK+DQN4
#test_model("EASY+DQN", BackfillEnv, plat4, test)
#test_model("EASY+DQN", BackfillEnv, plat2, test, train)

