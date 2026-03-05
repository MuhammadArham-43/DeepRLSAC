import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.nn_utils import weights_init_

class V(nn.Module):
    def __init__(self, num_inputs, hidden_dim, init, activation):
        super(V, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(lambda m: weights_init_(m, init=init, activation=activation))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
             raise ValueError(f"unknown activation {activation}")
    
    def forward(self, state):
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x


class Q(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        init,
        activation,
        bias: bool = True
    ):
        super(Q, self).__init__()

        self.network_input = num_inputs + num_actions

        self.linear1 = nn.Linear(self.network_input, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear3 = nn.Linear(hidden_dim, 1, bias=bias)

        self.apply(lambda m: weights_init_(m, init=init, activation=activation))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
             raise ValueError(f"unknown activation {activation}")

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x

class DoubleQ(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        init,
        activation,
        bias: bool = True
    ):
        super(DoubleQ, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear3 = nn.Linear(hidden_dim, 1, bias=bias)

        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim, bias=bias)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear6 = nn.Linear(hidden_dim, 1, bias=bias)

        self.apply(lambda m: weights_init_(m, init=init, activation=activation))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")
    
    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)

        x1 = self.activation(self.linear1(xu))
        x1 = self.activation(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = self.activation(self.linear4(xu))
        x2 = self.activation(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2