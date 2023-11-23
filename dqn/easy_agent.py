import numpy as np
import json

class EasyAgent():
    def play(self, env, verbose=True):
        history = { "score": 0, "steps": 0, "info": None }
        (obs, _), done, info = env.reset(), False, {}

        logs = { "scores": [], "queue_len": [], "wait": []}

        while not done:
            posible_jobs = any( True for j in obs["queue"]["jobs"] if j[1] <= obs["platform"]["nb_hosts"] )
            obs, reward, done, _, info = env.step( int(posible_jobs) )

            waits = np.array([ i.waiting_time for i in env.simulator.jobs if i.is_running ])
            logs["scores"].append(float(reward))
            logs["queue_len"].append( sum(1 for _ in env.simulator.queue) )
            logs["wait"].append( waits.mean() if waits.shape[0] else 0. )


            history['score'] += reward
            history['steps'] += 1
            history['info']   = info


        with open('/data/expe-out/play_scores.json', 'w') as fp:
            json.dump(logs, fp)

        if verbose:
            print(f"\n[DONE] Score: {history['score']} - Steps: {history['steps']} {len(env.simulator.queue)}")

