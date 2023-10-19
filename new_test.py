
from scheduling_env    import SchedulingEnv
from envs.per_core_env import PerCoreEnv

from schedulers.NewAgent  import AgentWrapper as QueueAgent
from schedulers.DQNAgent2 import AgentWrapper as PerCoreAgent

def run_per_core():
    print("[RUNNING]")

    env = PerCoreEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
                    workloads_dir = "/data/workloads/test",
                    t_action = 1,
                    queue_max_len = 20,
                    max_plan_time=100,
                    t_shutdown = 500,
                    hosts_per_server = 1)
    ini_state = env.reset()

    #state_size = len(ini_state["posible_spaces"]) * 3 + 3
    state_size  = 7

    action_size = 1
    for i in env.action_space:
        action_size *= int(i.high - i.low)

    agent = PerCoreAgent(state_size, action_size, 0)
    #agent.train(env)
    agent.play(env, False)

def run_gym():
    print("[RUNNING]")

    env = SchedulingEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
                    workloads_dir = "/data/workloads/test",
                    t_action = 10,
                    queue_max_len = 20,
                    t_shutdown = 500,
                    hosts_per_server = 1)

    ini_state = env.reset()
    #state_size  = ini_state["queue"]["jobs"].shape[0] * 3 + 6 + ini_state["platform"]["agenda"].shape[0]
    state_size  = 6
    action_size = env.action_space.n
    print(">>", action_size)

    agent = QueueAgent(state_size, action_size, 0)
    agent.play(env, False)


if __name__ == "__main__":
    #run_gym()
    run_per_core()
