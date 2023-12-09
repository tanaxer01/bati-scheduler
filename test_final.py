from logging import shutdown
from typing import Callable, Optional
import batsim_py

from envs.simple_env        import SimpleEnv
from envs.backfill_env      import BackfillEnv
from envs.shutdown_policies import ShutdownPolicy, TimeoutPolicy

from model.metrics  import MonitorsInterface

from model.agent    import Agent

# Meanwhile
STATE_SIZE = 7

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

def train_model(env_fn: type,
                platform_fn: str,
                workload_fn: str,
                shutdown_policy: Optional[Callable[[int], TimeoutPolicy]] = None):

    print("[TRAIN]")
    env = env_fn(platform_fn=platform_fn, workload_fn=workload_fn, shutdown_policy=shutdown_policy)
    obs_size = env.observation_space.shape[1]

    agent = Agent(obs_size)
    agent.play(env, 50, True)

def test_model(name:str,
               env_fn: type,
               platform_fn: str,
               test_fn: str,
               train_fn: Optional[str] = None,
               shutdown_policy: Optional[Callable[[int], TimeoutPolicy]] = None):

    if train_fn != None:
        train_model(env_fn, platform_fn, train_fn)

    print("[TEST]")
    env = env_fn(platform_fn=platform_fn, workload_fn=test_fn, shutdown_policy = shutdown_policy)
    obs_size = env.observation_space.shape[1]

    monitors = init_monitors(name)



    agent = Agent(obs_size, monitors=monitors)
    agent.load("/data/expe-out/network.chkpt/netwrk")

    agent.test(env)

## Files
plat  = "/data/platforms/generator/fat_tree_heterogeneous.xml"
plat2  = "/data/platforms/generator/generated.xml"
train = "/data/workloads/training/16nodes"

plat4  = "/data/platforms/generator/fat_tree_4.xml"
train4 = "/data/workloads/training/4nodes"

test  = "/data/workloads/test/w.json"

A  = "/data/workloads/generator/"
B  = "/data/workloads/generator/no_hills_workload.json"

##
T = 10
policy = lambda s: TimeoutPolicy(T, s)

### DQN4fat_tree_heterogeneous
#test_model("10Peaks", SimpleEnv, plat2, test, train, policy)

test_model("DQN+AV",   SimpleEnv, plat2, A + "caso_variado_A.json", None, policy)
test_model("DQN+BV",   SimpleEnv, plat2, A + "caso_variado_B.json", None, policy)
test_model("DQN+CV",   SimpleEnv, plat2, A + "caso_variado_C.json", None, policy)
#test_model("plain+A+DQN", SimpleEnv, plat2, A + "caso_plano_A.json", None, policy)
#test_model("plain+B+DQN", SimpleEnv, plat2, A + "caso_plano_B.json", None, policy)
#test_model("plain+C+DQN", SimpleEnv, plat2, A + "caso_plano_C.json", None, policy)

### BACK+DQN4
#test_model("EASY+DQN", BackfillEnv, plat4, test, train, policy)
#test_model("EASY+DQN", BackfillEnv, plat2, test, train)

