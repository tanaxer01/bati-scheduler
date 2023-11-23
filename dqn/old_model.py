import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.quantized import Conv2d

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions=1):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class TestConvDQN(nn.Module):

    def __init__(self, n_observations, n_actions=20):
        super(TestConvDQN, self).__init__()
        self.layer1 = nn.Conv2d(2, 3, kernel_size=1)
        self.layer2 = nn.Conv2d(3, 1, kernel_size=1)

    def  forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x.view(x.size(0), -1)
        return x

class ConvDQN(nn.Module):
    def __init__(self, n_observations, n_actions=20):
        super(ConvDQN, self).__init__()
        self.layer1 = nn.Conv2d(n_observations, n_actions, kernel_size=1)
        self.layer2 = nn.Conv2d(n_actions, 64, kernel_size=1)
        self.layer3 = nn.Conv2d(64, 32, kernel_size=1)
        self.layer4 = nn.Conv2d(32, 1, kernel_size=1)

    def  forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.view(x.size(0), -1)

if __name__ == "__main__":

    with torch.no_grad():
        x = torch.randn((1,4))
        print(f'{x.shape=}')
        linear = torch.nn.Linear(4,2,bias=False)
        print(f'{linear.weight.shape=}')
        conv = torch.nn.Conv2d(4,2,kernel_size=(1,1),bias=False)
        print(f'{conv.weight.shape=}')
        conv.weight.set_(linear.weight[:,:,None,None])
        print(f'{(linear.weight == conv.weight[:,:,0,0]).all()=}')
        print(f'{x @ linear.weight.T=}')
        print(linear(x))
        print(conv(x[:,:,None,None])[:,:,0,0])
        print("==============================")



