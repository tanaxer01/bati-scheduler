
from envs.QueueEnv import QueueEnv
from dqn.agent import Agent

env = QueueEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
            workloads_dir = "/data/workloads/test",
            t_action = 10,
            queue_max_len = 20,
            t_shutdown = 500,
            hosts_per_server = 1)

print("[START]")

ini_state = env.reset()
state_size = ini_state["platform"]["nb_hosts"] * 3 + 4

agent = Agent(state_size)
agent.train(env)

