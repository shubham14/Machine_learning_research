'''
Class blind pruning in RNN/LSTM and CNN arhitectures
'''

import sys
from time import time
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from RNN import *

sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01
threshold = 0.5

# MNIST Dataset
train_dataset = dsets.MNIST(root='../data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='../data/',
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

rnn = LSTM(input_size, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.RMSprop(rnn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    t0 = time()
    for i, (images, labels) in enumerate(train_loader):
        images = to_var(images.view(-1, sequence_length, input_size))
        labels = to_var(labels)

        optim.zero_grad()
        out = rnn(images)
        loss = criterion(out, labels)
        loss.backward()
        optim.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Time: %.2f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data, \
                     time()-t0))

acc = compute_accuracy(rnn, sequence_length, input_size, test_loader)
print(acc)

pruned_inds_by_layer = []
weight_tensors0 = []
weight_tensors1 = []
# count = 0
for p in rnn.parameters():
    pruned_inds = 'None'
    if len(p.data.size()) == 2:
    # if count == 0:
        pruned_inds = p.data.abs() < threshold
        weight_tensors0.append(p.clone())
        p.data[pruned_inds] = 0.
        weight_tensors1.append(p.clone())
    pruned_inds_by_layer.append(pruned_inds)
    # count += 1

acc = compute_accuracy(rnn, sequence_length, input_size, test_loader)
print(acc)

# Re-train the network but don't update zero-weights (by setting the corresponding gradients to zero)
for epoch in range(num_epochs):
    t0 = time()
    for i, (images, labels) in enumerate(train_loader):
        images = to_var(images.view(-1, sequence_length, input_size))
        labels = to_var(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = rnn(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # zero-out all the gradients corresponding to the pruned connections
        for l,p in enumerate(rnn.parameters()):
            pruned_inds = pruned_inds_by_layer[l]
            if type(pruned_inds) is not str:
                p.grad.data[pruned_inds] = 0.

        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Time: %.2f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0], \
                     time()-t0))

acc = compute_accuracy(rnn, sequence_length, input_size, test_loader)
print(acc)


weight_tensors2 = []
for p in rnn.parameters():
    pruned_inds = 'None'
    if len(p.data.size()) == 2:
        pruned_inds = p.data.abs() < threshold
        weight_tensors2.append(p.clone())
    pruned_inds_by_layer.append(pruned_inds)

param = list(rnn.parameters())
