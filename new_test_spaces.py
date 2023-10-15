from space_bases_env import SpaceBasedEnv
from schedulers.DQNAgent2 import AgentWrapper

def run_gym():
    print("[RUNNING]")

    env = SpaceBasedEnv(platform_fn = "/data/platforms/platform.xml",
                    workloads_dir = "/data/workloads/test",
                    t_action = 10,
                    queue_max_len = 20,
                    t_shutdown = 500,
                    hosts_per_server = 1)

    ini_state = env.reset()
    state_size  = ini_state["queue"]["jobs"].shape[0] * 3 + 6 + ini_state["platform"]["agenda"].shape[0]
    action_size = env.action_space.n

    agent = AgentWrapper(state_size, action_size, 0)
    #agent.train(env)
    #agent.play(env, False)

if __name__ == "__main__":
    run_gym()