import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNetwork(nn.Module):
  """Q-network"""
  def __init__(self, n_inputs, n_outputs, learning_rate):
    super(QNetwork, self).__init__()
    # network
    self.out = nn.Linear(n_inputs, n_outputs)
    torch.nn.init.uniform_(self.out.weight, 0, 0.01)
    # training
    self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)  # Change optimizer to Adam

  def forward(self, x):
    return self.out(x)

  def loss(self, q_outputs, q_targets):
    return torch.sum(torch.pow(q_targets - q_outputs, 2))

