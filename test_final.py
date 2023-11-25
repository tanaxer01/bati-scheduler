import batsim_py

from envs.basic_env import BasicEnv
from envs.backfill_env import BackfillEnv
#from envs.simplified_env import SimpleEnv
from envs.simple_env import SimpleEnv, SkipTime

from dqn.cnn_agent import CnnAgent
from dqn.simple_agent import SimpleAgent
from dqn.easy_agent import EasyAgent

from model.agent import Agent
from model.fcfs  import FCFSAgent


print("[TRAIN]")

state_size = 9
env = SimpleEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
            #workload_fn = "/data/workloads/test/w.json",
            workload_fn = "/data/workloads/training",
            track_dependencies=True,
            t_action = 5,
            t_shutdown = 60)
env = SkipTime(env)

ini_state = env.reset()
#state_size = ini_state["platform"]["nb_hosts"] * 3 + 4


agent = Agent(state_size)
agent.play(env, True)


print("[TEST]")

env = SimpleEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
            workload_fn = "/data/workloads/test/w.json",
            #workload_fn = "/data/workloads/training",
            track_dependencies=True,
            t_action = 5,
            t_shutdown = 60)
env = SkipTime(env)

jobs_df = batsim_py.monitors.JobMonitor(env.simulator)
sim_df  = batsim_py.monitors.SimulationMonitor(env.simulator)
schedule_df  = batsim_py.monitors.SchedulerMonitor(env.simulator)

#agent = FCFSAgent()
agent = Agent(state_size)
agent.load("/data/expe-out/network.chkpt/netwrk")
agent.test(env)


jobs_df.to_csv("/data/expe-out/jobs-DQN.out")
sim_df.to_csv("/data/expe-out/sim-DQN.out")
schedule_df.to_csv("/data/expe-out/schedule-DQN.out")
