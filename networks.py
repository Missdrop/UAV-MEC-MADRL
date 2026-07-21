import torch
import torch.nn as nn


class HiddenLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layer_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *([HiddenLayer(hidden_dim) for _ in range(hidden_layer_count)]),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.network(state)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layer_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *([HiddenLayer(hidden_dim) for _ in range(hidden_layer_count)]),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)
