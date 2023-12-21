import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class lateral_QNetwork(nn.Module):
  """Q-network"""
  def __init__(self, n_inputs, n_outputs, n_hidden, learning_rate):
    super(lateral_QNetwork, self).__init__()
    # network

    if n_hidden>0:
      # Define hidden layer
      self.hidden = nn.Linear(n_inputs, n_hidden)

      # Define output layer
      self.out = nn.Linear(n_hidden, n_outputs, bias = False)

    else:
      self.hidden = 0
      # Define output layer
      self.out = nn.Linear(n_inputs, n_outputs, bias = False)
      torch.nn.init.uniform_(self.out.weight, 0, 0.01)


    # training
    self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)  # Change optimizer to Adam

  def forward(self, x):
    if self.hidden == 0:
      return self.out(x)
    else:
      x = self.hidden(x)
      return self.out(x)

  def loss(self, q_outputs, q_targets):
    return torch.sum(torch.pow(q_targets - q_outputs, 2))

class vertical_QNetwork(nn.Module):
  """Q-network"""
  def __init__(self, n_inputs, n_outputs, learning_rate):
    super(vertical_QNetwork, self).__init__()
    # network
    self.out = nn.Linear(n_inputs, n_outputs)
    torch.nn.init.uniform_(self.out.weight, 0, 0.01)
    # training
    self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)  # Change optimizer to Adam

  def forward(self, x):
    return self.out(x)

  def loss(self, q_outputs, q_targets):
    return torch.sum(torch.pow(q_targets - q_outputs, 2))