import torch
import torch.nn as nn
import torch.functional as F
import torchtext

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        