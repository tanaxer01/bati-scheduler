import batsim_py

from envs.simple_env import SimpleEnv, SkipTime
from envs.backfill_env import BackfillEnv
from envs.shutdown_policies import TimeoutPolicy

from model.metrics import MonitorsInterface

from model.agent import Agent
from model.bf_agent import BFAgent
from model.fcfs  import FCFSAgent


print("[TRAIN]")

#state_size = 5
state_size = 8


'''
#env = BackfillEnv(
env = SimpleEnv(
    platform_fn = "/data/platforms/FatTree/fat_tree_4.xml",
    workload_fn = "/data/workloads/training4",
    track_dependencies=True,
    t_action = 5,
    t_shutdown = 1,
    #shutdown_policy = TimeoutPolicy
    )
env = SkipTime(env)

ini_state = env.reset()

agent = Agent(state_size)
#agent = BFAgent(state_size)

#agent.play(env, True)

'''

print("[TEST]")

#env = BackfillEnv(
env = SimpleEnv(
        platform_fn = "/data/platforms/FatTree/fat_tree_4.xml",
        workload_fn = "/data/workloads/test/w.json",
        track_dependencies=True,
        t_action = 5,
        t_shutdown = 4)
env = SkipTime(env)

batsim_monitors = MonitorsInterface(
        name = "FCFS",
        save_dir ="/data/expe-out",
        monitors_fns = [
            batsim_py.monitors.JobMonitor,
            batsim_py.monitors.SimulationMonitor,
            batsim_py.monitors.SchedulerMonitor,
            batsim_py.monitors.ConsumedEnergyMonitor,
            batsim_py.monitors.HostStateSwitchMonitor
        ])

agent = FCFSAgent(batsim_monitors)

#agent = Agent(state_size, monitors=batsim_monitors)
#agent = BFAgent(state_size, monitors=batsim_monitors)
#agent.load("/data/expe-out/network.chkpt/netwrk")

agent.test(env)

