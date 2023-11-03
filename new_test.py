import batsim_py

from envs.final_env    import DspEnv
from envs.QueueEnv     import QueueEnv
from envs.per_core_env import PerCoreEnv

from schedulers.QueueSched import AgentWrapper as QueueAgent
from schedulers.DQNAgent2  import AgentWrapper as PerCoreAgent

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

    env = QueueEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
                    workloads_dir = "/data/workloads/test",
                    t_action = 10,
                    queue_max_len = 20,
                    t_shutdown = 500,
                    hosts_per_server = 1)

    jobs_mon = batsim_py.monitors.JobMonitor(env.simulator)
    sim_mon  = batsim_py.monitors.SimulationMonitor(env.simulator)
    schedule_mon  = batsim_py.monitors.SchedulerMonitor(env.simulator)

    ini_state = env.reset()
    state_size = 9
    action_size = env.action_space.n

    agent = QueueAgent(state_size, action_size, 0)

    #agent.train()


    hist = {}
    hist = agent.play(env, True)

    print("[DONE]")
    return jobs_mon, sim_mon, schedule_mon, hist

def run_final():
    print("[RUNNING]")

    env = DspEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
                    workloads_dir = "/data/workloads/test",
                    t_action = 10,
                    queue_max_len = 20,
                    t_shutdown = 500,
                    hosts_per_server = 1)

    agent = QueueAgent(9, 1, 0)
    agent.play(env, True)




if __name__ == "__main__":
    #run_per_core()
    #run_final()
    run_gym()

    '''
    jobs_df, sim_df, schedule_df, _ = run_gym()

    jobs_df.to_csv("/data/expe-out/jobs-DQN.out")
    sim_df.to_csv("/data/expe-out/sim-DQN.out")
    schedule_df.to_csv("/data/expe-out/schedule-DQN.out")
    '''
