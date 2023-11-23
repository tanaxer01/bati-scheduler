import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x).T



if __name__ == "__main__":
    import gym
    env = gym.make("CartPole-v1")

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state = env.reset()
    n_observations = len(state)

    A = DQN(n_observations, n_actions)
    B = DQN(n_observations+1, 1)

    inp  = torch.randn(1, 4)
    inp2 = inp.expand(2, -1)
    idxs = torch.arange(inp2.size(0)).unsqueeze(1)
    inp2 = torch.cat((inp2, idxs), dim=1)

    print(inp.shape, inp)
    print(inp2.shape, inp2)
    print("==================================================")
    out  = A(inp).T
    out2 = B(inp2)

    print(out.shape, out)
    print(out2.shape, out2)
    print("==================================================")
    maxi  = out.max(1).indices.view(1,1)
    maxi2 = out2.max(1).indices.view(1,1)

    print(maxi.shape, maxi)
    print(maxi2.shape, maxi2)
    print("==================================================")
    print("==================================================")
    batch = torch.stack([ inp for i in range(3) ], dim=0)

    print(batch.shape, batch)
    print("==================================================")
    batch_out = A(batch)

    print(batch_out.shape, batch_out)
    print("==================================================")
    batch_max = batch_out.max(2).indices

    print(batch_max.shape, batch_max)

