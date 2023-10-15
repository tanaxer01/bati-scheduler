import gym
from batsim_py import monitors 
import test_env as e

class BatiScheduler():
    def act(self, obs):
        pass

    def play(self, env, verbose=True):
        history = { "score": 0, "steps": 0, "info": None }
        obs, done, info, = env.reset(), False, {}

        while not done:
            obs, reward, done, info = env.step(self.act(obs))
            history["score"] += reward
            history["steps"] += 1
            history["info"]   = info

        if verbose:
            print(f"[DONE] Score: {history['score']} - Steps: {history['steps']}")

        env.close()
        return history

def run():
    agent = BatiScheduler()
    env   = gym.make()

    jobs_mon = monitors.JobMonitor(env.simulator)
    sim_mon  = monitors.SimulationMonitor(env.simulator)

    hist  = agent.play(env, True)
    print("[DONE]")

    return jobs_mon, sim_mon

if __name__ == "__main__":
    run()

