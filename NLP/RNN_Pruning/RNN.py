import torch
import torch.nn as nn
import torch.autograd as Variable

def to_var(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return torch.tensor(x)

def compute_accuracy(rnn, sequence_length, input_size, data_loader):
    correct = 0; total = 0
    for images, labels in data_loader:
        images = to_var(images.view(-1, sequence_length, input_size))
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    acc = (100. * correct)/total
    return acc



class RNN(nn.Module):
    def __init__(self, inp_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(inp_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, batch_first=True))
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class LSTM(nn.Module):
    def __init__(self, inp_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(inp_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = to_var(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = to_var(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        out, (h0, c0) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out