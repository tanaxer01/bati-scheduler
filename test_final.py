import batsim_py

from envs.QueueEnv import QueueEnv
from dqn.agent import Agent

env = QueueEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
            workloads_dir = "/data/workloads/test",
            t_action = 10,
            queue_max_len = 20,
            t_shutdown = 500,
            hosts_per_server = 1)

print("[START]")

jobs_df = batsim_py.monitors.JobMonitor(env.simulator)
sim_df  = batsim_py.monitors.SimulationMonitor(env.simulator)
schedule_df  = batsim_py.monitors.SchedulerMonitor(env.simulator)

ini_state = env.reset()
state_size = ini_state["platform"]["nb_hosts"] * 3 + 4

agent = Agent(state_size)
agent.train(env)

jobs_df.to_csv("/data/expe-out/jobs-DQN.out")
sim_df.to_csv("/data/expe-out/sim-DQN.out")
schedule_df.to_csv("/data/expe-out/schedule-DQN.out")

