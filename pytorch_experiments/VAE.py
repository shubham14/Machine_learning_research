# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 17:50:45 2019

@author: Shubham
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm

class Normal:
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r
    
class Encoder(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        out = F.relu(self.linear2(x))
        return out
    
class Decoder(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        out = F.relu(self.linear2(x))
        return out
    
class VAE(nn.Module):
    latent_dim = 8
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = nn.Linear(100, 8)
        self._enc_log_sigma = nn.Linear(100, 8)
        
    def _sample_latent(self, h_enc):
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        self.z_mean = mu
        self.z_sigma = sigma
        
        return mu + sigma * Variable(std_z, requires_grad=False)
    
    def forward(self, state):
        h_enc = self.encoder(state)
        z =  self._sample_latent(h_enc)
        return self.decoder(z)
    
def latent_loss(z_mean, z_std):
    return 0.5 * torch.mean(z_mean * z_mean + z_std * z_std - torch.log(z_std *z_std) - 1)

if __name__ == "__main__":
    inp_dim = 28 * 28
    batch_size = 32
    transform = transforms.Compose(
            [transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
    encoder = Encoder(inp_dim, 100, 100)
    decoder = Decoder(8, 100, inp_dim)
    vae = VAE(encoder, decoder)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    for epoch in range(100):
        for i, data in tqdm(enumerate(dataloader, 0), total=len(mnist)/batch_size):
            inputs, classes = data
            inputs, classes = Variable(inputs.resize_(batch_size, inp_dim)), Variable(classes)
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            l = loss.item()
        print(epoch, l)

    plt.imshow(vae(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
    plt.show(block=True)
        