'''
contains 

'''
from collections import namedtuple
import torchvision.models as models
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torch.autograd as autograd
import torch.nn.functional as F

class MNISTNet(nn.Module):
    '''
    Simple classification (Dense) network for MNIST
    '''
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
    
    @property
    def __name__(self):
        return "MNIST Dense Network"

class CIFARNet(nn.Module):
    '''
    Simple classification (Conv) network for CIFAR-10
    '''
    def __init__(self):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(1250, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    @property
    def __name__(self):
        return "CIFAR-10 CNN"

if __name__ == "__main__":
    net = MNISTNet()
    net1 = CIFARNet()
    print(net.__name__)
    print(net1.__name__)