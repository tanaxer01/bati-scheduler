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

class DDQN(nn.Module):
    def __init__(self, n_observations:  int, n_actions: int) -> None:
        super().__init__()

        self.online = self._build_fcn(n_observations, n_actions)
        self.target = self._build_fcn(n_observations, n_actions)
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        match model:
            case "online":
                return self.online(input)
            case "target":
                return self.target(input)

    def _build_fcn(self, obs, acts):
        return nn.Sequential(
            # Layer 1
            nn.Linear(obs, 64),
            nn.ReLU(),
            # Layer 2
            nn.Linear(64, 64),
            nn.ReLU(),
            # Layer 3
            nn.Linear(64, 32),
            nn.ReLU(),
            # Layer 4
            nn.Linear(32, acts)
        )



if __name__ == "__main__":
    n_observations = 8
    n_actions = 1

    print("Using cuda:", torch.cuda.is_available())
    print("==================================================")

    A = DQN(n_observations, n_actions)
    B = DQN(n_observations+1, 1)

    inp  = torch.randn(1, n_observations)
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
    '''
    batch = torch.stack([ inp for i in range(3) ], dim=0)

    print(batch.shape, batch)
    print("==================================================")
    batch_out = A(batch)

    print(batch_out.shape, batch_out)
    print("==================================================")
    batch_max = batch_out.max(2).indices

    print(batch_max.shape, batch_max)
    '''

