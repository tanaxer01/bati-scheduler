import batsim_py

from schedulers.DQNAgent3 import DQNAgent

#import gridgym.envs.scheduling_env as e
import scheduling_env as e
#import test_env as e


def run_gym():
#    env = e.SchedulingEnv(platform_fn = "/data/platforms/platform.xml",
    env = e.SchedulingEnv(platform_fn = "/data/platforms/platform.xml",
                    workloads_dir = "/data/workloads/test",
                    t_action = 3,
                    queue_max_len = 20,
                    t_shutdown = 1,
                    hosts_per_server = 1)

    jobs_mon = batsim_py.monitors.JobMonitor(env.simulator)
    sim_mon  = batsim_py.monitors.SimulationMonitor(env.simulator)
    schedule_mon  = batsim_py.monitors.SchedulerMonitor(env.simulator)

    agent = DQNAgent()
    hist = agent.play(env, True)

    print("[DONE]")
    return jobs_mon, sim_mon, schedule_mon, hist

if __name__ == "__main__":
    jobs_df, sim_df, schedule_df, _ = run_gym()

    jobs_df.to_csv("/data/expe-out/jobs-DQN.out")
    sim_df.to_csv("/data/expe-out/sim-DQN.out")
    schedule_df.to_csv("/data/expe-out/schedule-DQN.out")
