import batsim_py
import gym

import numpy as np


from schedulers.DQNAgent2 import DQNAgent
import gridgym.envs.off_reservation_env as e
#import gridgym.envs.scheduling_env as e
import test_env as e


class FirstFitScheduler():
    def act(self, obs):
        agenda = obs['platform']['agenda']
        queue = obs['queue']['jobs']
        nb_available = len(agenda) - sum(1 for j in agenda if j[1] != 0)
        job_pos = next((i for i, j in enumerate(queue) if 0 < j[1] <= nb_available), -1)

        curr = obs["current_time"]
        if not np.all(queue == 0):
            res = [ curr - i if i != 0 else 0 for i in queue[: , 0]  ]

        return job_pos + 1

    def play(self, env, verbose=True):
        history = {"score": 0, 'steps': 0, 'info': None}
        obs, done, info = env.reset(), False, {}
        while not done:
            obs, reward, done, info = env.step(self.act(obs))
            history['score'] += reward
            history['steps'] += 1
            history['info'] = info

            if history["score"] < -3000:
                done = True

        if verbose:
            print("[DONE] Score: {} - Steps: {}".format(history['score'], history['steps']))
        env.close()
        return history


def run_gym():
    print("[RUNNING]")

    agent  = FirstFitScheduler()

    '''
    env = gym.make("gridgym:Scheduling-v0",
                   #platform_fn="/data/platforms/FatTree/generated.xml",
                   platform_fn="/data/platforms/platform.xml",
                   workloads_dir="/data/workloads/test",
                   t_action=1,
                   queue_max_len=20,
                   t_shutdown=5,
                   hosts_per_server=2)
    '''
    env = e.TestEnv(platform_fn = "/data/platforms/FatTree/generated.xml",
                    workloads_dir = "/data/workloads/test",
                    t_action = 1,
                    queue_max_len = 10,
                    t_shutdown = 5,
                    hosts_per_server = 1)



    jobs_mon = batsim_py.monitors.JobMonitor(env.simulator)
    sim_mon  = batsim_py.monitors.SimulationMonitor(env.simulator)
    schedule_mon  = batsim_py.monitors.SchedulerMonitor(env.simulator)

    agent = DQNAgent()
    #agent = FirstFitScheduler()
    hist = agent.play(env, True)


    print("[DONE]")
    return jobs_mon, sim_mon, schedule_mon, hist


if __name__ == "__main__":
    jobs_df, sim_df, schedule_df, _ = run_gym()

    jobs_df.to_csv("/data/expe-out/jobs-DQN.out")
    sim_df.to_csv("/data/expe-out/sim-DQN.out")
    schedule_df.to_csv("/data/expe-out/schedule-DQN.out")
