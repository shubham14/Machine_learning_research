'''
Python script to train/prune networks based on MNIST and CIFAR-10
'''

# import torch libraries
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
# import utility libraries
import sys
import argparse
import os

# toggle use of GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(dataset, root, transform_set, download=True, batch_size=100):
    '''
    Takes in root data of the MNIST/CIFAR-10 file
    dataset : str, MNIST and CIFAR-10
    '''
    if dataset.lower() == 'mnist':
        train_set = dset.MNIST(root=root, train=True, transform=transform_set, download=download)
        test_set = dset.MNIST(root=root, train=False, transform=transform_set, download=download)
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                        batch_size=batch_size,
                        shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                        dataset=test_set,
                        batch_size=1,
                        shuffle=False)

        print ('==>>> total training batches: {}'.format(len(train_loader)))
        print ('==>>> total testing samples: {}'.format(len(test_loader)))
        return train_loader, test_loader
    else:
        train_set = dset.CIFAR10(root=root, train=True, transform=transform_set, download=download)
        test_set = dset.CIFAR10(root=root, train=False, transform=transform_set, download=download)
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                        batch_size=batch_size,
                        shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                        dataset=test_set,
                        batch_size=1,
                        shuffle=False)

        print('==>>> total training batches: {}'.format(len(train_loader)))
        print('==>>> total testing samples: {}'.format(len(test_loader)))
        return train_loader, test_loader

class Trainer():
    '''
    drop_rate : float, pruning percentage
    train_loader, test_loader obtained from data_load
    '''
    def __init__(self, drop_rate, dataset):
        self.drop_rate = drop_rate
        self.dataset = dataset 
    
    def setOptimzer(self, net):
        pass

    def train(self, trainloader, net, criterion, optimizer, device, scheduler):
        for epoch in range(100):  # loop over the dataset multiple times
            scheduler.step()
            start = time.time()
            running_loss = 0.0
            for i, (images, labels) in enumerate(trainloader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                out = net(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 2000 mini-batches
                    end = time.time()
                    print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                        (epoch + 1, i + 1, running_loss / 100, end-start))
                    start = time.time()
                    running_loss = 0.0
        print('Finished Training')


    def test(self, testloader, net, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

if __name__ == "__main__":
    root = r'C:\Users\Shubham\Desktop\Lottery Ticket Hypothesis'
    transform_set = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_loader, test_loader = load_data('cifar', root, transform_set)
