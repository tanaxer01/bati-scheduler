import random
import numpy as np
import torch

from collections import deque
from itertools   import combinations

from .dqn import Agent
from .FreeSpaces import FreeSpaceContainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentWrapper:
    def __init__(self, state_size, action_size, seed):
        self.seed = random.seed(seed)
        self.agent = Agent(state_size, action_size, seed)
        self.listFreeSpaces = FreeSpaceContainer(action_size, 20.0)

    def act(self, obs, eps) -> int:
        platform = obs["platform"]
        queue    = obs["queue"]

        # 1. Check if queue is empty, in that case do nothing.
        if queue["size"] == 0:
            # TODO - check if shuffling the freeSpaceContainer could improve the schedule
            return 0

        # 2. Calculate posible free spaces 
        job = queue["jobs"][0]
        posible_spaces = self.listFreeSpaces.get_posible_spaces(job.wall, job.res)

        # 3. Score each space 



        # Prep obs for self.agent.act

        if queue["size"] == 0:
            return 0

        ## Queue Data
        queue_len  = queue["size"]
        queue_wait = obs["current_time"] - queue["jobs"][:,0]
        queue_res  = queue["jobs"][:,1]
        queue_wall = queue["jobs"][:,2]

        ## Plaform Data
        host_status = np.zeros(5)
        for i in platform["status"]:
            host_status[i-1] += 1
        host_status /= platform["agenda"].shape[0]
        host_remaining_time = obs["current_time"] - (platform["agenda"][:,0] + platform["agenda"][:,1])

        all_data = np.concatenate([
            [queue_len],
            queue_wait,
            queue_res,
            queue_wall,
            host_status,
            host_remaining_time
        ])

        obs = torch.from_numpy(all_data).float().unsqueeze(0).to(device)
        self.agent.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.agent.qnetwork_local(obs)
        self.agent.qnetwork_local.train()
        valid_actions = np.array([ j if i < queue_len else -1 * float('inf') for i,j in enumerate(action_values.tolist()[0])])

        # Epsilon -greedy action selection
        if random.random() > eps:
            return np.argmax(valid_actions) - 1
            #return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(queue_len)) - 1
            #return random.choice(np.arange(self.action_size))
        #return self.agent.act(all_data, eps) - 1

    def _get_free_spaces(self):
        pass

    def _score_spaces(self, spaces):
        pass

    def train(self, env, n_episodes = 200, max_t = 1000, eps_start = 1.0, eps_end = 0.1, eps_decay = 0.996, verbose=True) -> None:
        scores = [] # list containing score from each episode 
        scores_window = deque(maxlen = 100) # last 100 scores
        eps = eps_start

        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0

            for t in range(max_t):
                action = self.act(state, eps)
                next_state, reward, done, _ = env.step(action)

                ## above step decides whether we will train(learn) the network
                ## actor (local_qnetwork) or we will fill the replay buffer
                ## if len replay buffer is equal to the batch size then we will
                ## train the network or otherwise we will add experience tuple in our 
                ## replay buffer.
                state = next_state
                score += reward
                if done: 
                    break

                scores_window.append(score)
                scores.append(score)

                eps = max(eps*eps_decay, eps_end)
                print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
                if i_episode %100==0:
                    print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))

                if np.mean(scores_window)>=200.0:
                    print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                           np.mean(scores_window)))
                    torch.save(self.agent.qnetwork_local.state_dict(),'checkpoint.pth')
                    break
        return scores

    def play(self, env, load_data=True, verbose=True) -> None:
        if load_data:
            self.agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

        eps = 1.0 
        eps_end = 0.1
        eps_decay = 0.996


        history = { 'score': 0, 'steps': 0, 'info': None }
        obs, done, info = env.reset(), False, {}

        while not done:
            obs, reward, done, info = env.step(self.act(obs, eps))
            history['score'] += reward
            history['steps'] += 1
            history['info'] = info
            print(history["steps"], history["score"], eps)

            if history["score"] < -100:
                break

            eps = max(eps*eps_decay, eps_end)


        if verbose:
            print(f"[DONE] Score: {history['score']} - Steps: {history['steps']}")

'''
def action_to_num(machines: list[int], action: list[int]) -> int:
    all_actions = sum([ list(combinations(machines, i))  for i in range(1, len(machines)+1) ], [])
    action_dict = { j: i for i, j in enumerate([()] + all_actions) }

    if action in all_actions:
        return action_dict[action]
    return -1

class DQNAgent(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps_done = 0
        self.listFreeSpace : Optional[FreeSpaceContainer] = None

    def act(self, obs) -> int:
        queue = obs['queue']
        platform = obs['platform']
        nb_available = len(platform["agenda"]) - sum(1 for j in platform["agenda"] if j[1] != 0)

        # Add fst job to listFreeSpace
        if queue["size"] == 0:
            return 0

        job = queue["jobs"][0]
        posible_actions = self.listFreeSpace.get_spaces(job[3], job[2])
        #posible_actions = [ action_to_num(obs["platform"]["ids"], i) for i in posible_actions ]

        # 
        if len(posible_actions) == 0:
            pass 

        self.steps_done += 1
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        if sample > eps_threshold:
            # EXPLOTATION

            ## Reservations
            utilization = np.full(nb_available, 1)
            for i in self.listFreeSpace.agenda:
                utilization[i-1] 

            ## Queue Data
            queue_len  = queue["size"]
            queue_wait = obs["current_time"] - queue["jobs"][:,0]
            queue_res  = queue["jobs"][:,1]
            queue_wall = queue["jobs"][:,2]
            queue_data = np.concatenate([ [queue_len],queue_wait,queue_res,queue_wall ])

            ## Platform Data
            host_status = np.zeros(5)
            for i in platform["status"]:
                host_status[i-1] += 1
            host_status /= platform["agenda"].shape[0]
            host_remaining_time = obs["current_time"] - (platform["agenda"][:,0] + platform["agenda"][:,1])
            host_data  = np.concatenate([ host_status, host_remaining_time ]) 

            all_data   = np.concatenate([queue_data, host_data])
            all_tensor = torch.tensor(all_data, device=self.device, dtype=torch.float)

            with torch.no_grad():
                # Build Q-Net input
                # - Util. x Host en el FreeSpace
                # - Cant Espacios
                # -  

                future_reservations = np.array(sum([[ i.host, i.start, i.end ] for i in self.listFreeSpace.agenda], []))
                #queue_data    = np.concatenate([obs["queue"]["jobs"].ravel(), [obs["queue"]["size"]]])
                #platform_data = np.concatenate([obs["platform"]["status"].ravel(),obs["platform"]["agenda"].ravel()])
                #state_data = np.concatenate([queue_data, platform_data])
                #size = obs["queue"]["size"]
                state_data = np.concatenate()

                obs_arr = torch.tensor(state_data, device=self.device, dtype=torch.float)
                results = self.policy_net(obs_arr).tolist()
                masked_results = [ i if j in posible_actions else -1*float('inf') for i, j in enumerate(results) ]

                return masked_results.index(max(masked_results))
        else:
            # EXPLORATION
            # return random.choice(posible_actions)
            pass

    def play(self, env, verbose=True) -> None:
        history = { 'score': 0, 'steps': 0, 'info': None }
        obs, done, info = env.reset(), False, {}

        nb_core = len(obs["platform"]["status"])
        # queue
        queue_s = obs["queue"]["jobs"].ravel().shape[0] + 1
        print(obs["queue"].keys(), queue_s)

        platform_s = obs["platform"]["status"].ravel().shape[0] + obs["platform"]["agenda"].ravel().shape[0]
        print(obs["platform"].keys(), platform_s)

        state_s = queue_s + platform_s + 1
        obs_s   = 2**nb_core
        print(">>", obs_s)
        print(">>", state_s, env.action_space.n)

        self.listFreeSpace = FreeSpaceContainer(nb_core, 20.)

        self.policy_net = DQN(state_s, obs_s).to(self.device)
        #self.target_net = DQN(state_s, env.action_space.n).to(self.device)
        #self.target_net.load_state_dict(self.policy_net.state_dict())

        while not done:
            obs, reward, done, info = env.step(self.act(obs))
            history['score'] += reward
            history['steps'] += 1
            history['info'] = info

            if history["score"] < -3000:
                print(f"[ERROR] Simulation stopped.")

        if verbose:
            print(f"[DONE] Score: {history['score']} - Steps: {history['steps']}")
'''

