import torch
from torch import nn


class HiddenLayer(nn.Module):
    def __init__(self, dim, layer_norm=False, dropout=0.0):
        super().__init__()
        # Hidden layer = linear + layer norm + ReLU + dropout
        layers = [nn.Linear(dim, dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ActorNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_layer_count,
        layer_norm=False,
        dropout=0.0,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
        )

        # hidden layers
        if isinstance(hidden_dim, int):
            hidden_layers = [
                HiddenLayer(hidden_dim, layer_norm, dropout)
                for _ in range(hidden_layer_count)
            ]
        else:
            hidden_layers = [
                HiddenLayer(in_dim, layer_norm, dropout) for in_dim in hidden_dim
            ]
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.input(x)
        x = self.hidden_layers(x)
        return self.output(x)


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim: int | tuple[int],
        hidden_layer_count=0,
        layer_norm=False,
        dropout=0.0,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
        )

        # hidden layers
        if isinstance(hidden_dim, int):
            hidden_layers = [
                HiddenLayer(hidden_dim, layer_norm, dropout)
                for _ in range(hidden_layer_count)
            ]
        else:
            hidden_layers = [
                HiddenLayer(in_dim, layer_norm, dropout) for in_dim in hidden_dim
            ]

        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.input(x)
        x = self.hidden_layers(x)
        return self.output(x)
