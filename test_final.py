
from envs.QueueEnv import QueueEnv
from dqn.agent import Agent

env = QueueEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
            workloads_dir = "/data/workloads/test",
            t_action = 10,
            queue_max_len = 20,
            t_shutdown = 500,
            hosts_per_server = 1)

print("[START]")
agent = Agent(9)
agent.train(env)

