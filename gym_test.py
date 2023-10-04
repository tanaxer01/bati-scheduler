import argparse

import gym
import batsim_py

from schedulers.FCFSScheduler import FCFSScheduler
import gridgym.envs.off_reservation_env as e


class FirstFitScheduler():
    def act(self, obs):
        agenda = obs['platform']['agenda']
        queue = obs['queue']['jobs']
        nb_available = len(agenda) - sum(1 for j in agenda if j[1] != 0)
        job_pos = next((i for i, j in enumerate(queue)
                        if 0 < j[1] <= nb_available), -1)
        return job_pos + 1

    def play(self, env, verbose=True):
        history = {"score": 0, 'steps': 0, 'info': None}
        obs, done, info = env.reset(), False, {}
        while not done:
            obs, reward, done, info = env.step(self.act(obs))
            history['score'] += reward
            history['steps'] += 1
            history['info'] = info

        if verbose:
            print(
                "[DONE] Score: {} - Steps: {}".format(history['score'], history['steps']))
        env.close()
        return history


def run_gym():
    print("[RUNNING]")

    agent = FirstFitScheduler()

    env = gym.make("gridgym:Scheduling-v0",
                   platform_fn="/data/platforms/energy_platform_mod.xml",
                   workloads_dir="/data/workloads/test",
                   t_action=1,
                   queue_max_len=20,
                   t_shutdown=5,
                   hosts_per_server=12)

    jobs_mon = batsim_py.monitors.JobMonitor(env.simulator)
    sim_mon  = batsim_py.monitors.SimulationMonitor(env.simulator)

    hist = agent.play(env, True)

    print("[DONE]")
    return jobs_mon, sim_mon, hist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="gridgym:Scheduling-v0", type=str)
    # Agent specific args
    parser.add_argument("--queue_sz", default=20, type=int)
    parser.add_argument("--t_action", default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run_gym()
