import os

import torch
from torch import nn, optim


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class HiddenLayer(nn.Module):
    def __init__(self, dim, layer_norm=False, dropout_rate=0.0):
        super().__init__()
        # Hidden layer = linear + layer norm + ReLU + dropout
        layers: list[nn.Module] = [nn.Linear(dim, dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(nn.ReLU())
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))

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
        dropout_rate=0.0,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.hidden_layers = nn.Sequential(
            *[
                HiddenLayer(hidden_dim, layer_norm, dropout_rate)
                for _ in range(hidden_layer_count)
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        x = self.input(state)
        x = self.hidden_layers(x)
        return self.output(x)


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim: int,
        hidden_layer_count,
        layer_norm=False,
        dropout_rate=0.0,
        # attention
        head_count: int = 0,
        encoder_layer_count: int = 0,
    ):
        super().__init__()
        # attention switch
        self.attention = encoder_layer_count > 0

        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
        )
        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=head_count,
                dim_feedforward=hidden_dim,
                dropout=dropout_rate,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=encoder_layer_count
            )
            self.pool = nn.Linear(hidden_dim, 1)
        else:
            self.encoder = nn.Identity()
            self.pool = nn.Identity()

        self.hidden_layers = nn.Sequential(
            *[
                HiddenLayer(hidden_dim, layer_norm, dropout_rate)
                for _ in range(hidden_layer_count)
            ]
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # -> [batch_size, agent_count, state_dim + action_dim]
        x = torch.cat([state, action], dim=-1)
        # if using attention
        if self.attention:
            x = self.input(x)
            x = self.encoder(x)

            # use softmax to normalize weights
            weights = torch.softmax(self.pool(x), dim=1)
            x = torch.sum(x * weights, dim=1)
        else:
            x = x.view(state.size(0), -1)
            x = self.input(x)

        x = self.hidden_layers(x)
        return self.output(x)


class Actor:
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        hidden_layer_count: int = 2,
        learning_rate: float = 1e-4,
        learning_rate_decay: float = 0.9999,
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        self.device = device

        # initialize networks
        self.online_actor = ActorNetwork(
            state_dim,
            hidden_dim,
            action_dim,
            hidden_layer_count,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
        )
        self.shadow_actor = ActorNetwork(
            state_dim,
            hidden_dim,
            action_dim,
            hidden_layer_count,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
        )
        self.exploratory_actor = ActorNetwork(
            state_dim,
            hidden_dim,
            action_dim,
            hidden_layer_count,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
        )

        # move networks to device
        self.online_actor.to(device)
        self.shadow_actor.to(device)
        self.exploratory_actor.to(device)

        # copy parameters to target networks
        self.shadow_actor.load_state_dict(self.online_actor.state_dict())
        # exploratory_actor need not be synced since it will be synced on forwarding

        # set eval
        self.shadow_actor.eval()
        self.exploratory_actor.eval()

        # initialize optimizer
        self.optimizer = optim.Adam(self.online_actor.parameters(), lr=learning_rate)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=learning_rate_decay
        )

    def forward(
        self,
        observation: torch.Tensor,
        use_target: bool = False,
        explore: bool = False,  # use exploratory actor
        action_noise: float = 0.0,
        action_noise_limit: float | None = None,
        param_noise: float = 0.0,  # parameter noise for exploratory actor
        with_gradient: bool = False,
    ) -> torch.Tensor:
        # choose actor
        if explore:
            # update exploratory parameters
            self.exploratory_actor.load_state_dict(self.online_actor.state_dict())
            with torch.no_grad():
                for param in self.exploratory_actor.parameters():
                    # add noise
                    param += torch.randn_like(param) * param_noise
            actor_net = self.exploratory_actor
        elif use_target:
            actor_net = self.shadow_actor
        else:
            actor_net = self.online_actor

        # forward & add noise
        with torch.set_grad_enabled(with_gradient):
            action = actor_net(observation)
            # add gaussian noise with (mean = 0, standard deviation = noise_mu)
            if action_noise > 0:
                noise = torch.randn_like(action) * action_noise
                if action_noise_limit is not None:
                    noise = torch.clamp(noise, -action_noise_limit, action_noise_limit)
                action = action + noise

            action = torch.clamp(action, -1.0, 1.0)

        return action

    def step(self, loss):
        # do optimizer.step and decay learning rate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def eval(self):
        self.online_actor.eval()

    def train(self):
        self.online_actor.train()

    def update(self, tau: float):
        soft_update(self.shadow_actor, self.online_actor, tau)

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        torch.save(
            self.online_actor.state_dict(),
            os.path.join(directory, "actor.pth"),
        )
        torch.save(
            self.shadow_actor.state_dict(),
            os.path.join(directory, "shadow_actor.pth"),
        )
        torch.save(
            self.exploratory_actor.state_dict(),
            os.path.join(directory, "exploratory_actor.pth"),
        )
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(directory, "actor_optimizer.pth"),
        )

    def load(self, directory: str):
        self.online_actor.load_state_dict(
            torch.load(
                os.path.join(directory, "actor.pth"),
                map_location=self.device,
            )
        )
        self.shadow_actor.load_state_dict(
            torch.load(
                os.path.join(directory, "shadow_actor.pth"),
                map_location=self.device,
            )
        )
        self.exploratory_actor.load_state_dict(
            torch.load(
                os.path.join(directory, "exploratory_actor.pth"),
                map_location=self.device,
            )
        )
        self.optimizer.load_state_dict(
            torch.load(
                os.path.join(directory, "actor_optimizer.pth"),
                map_location=self.device,
            )
        )


class Critic:
    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        hidden_dim: int = 64,
        hidden_layer_count: int = 2,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.9999,
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
        # attention
        head_count: int = 0,
        encoder_layer_count: int = 0,
    ):
        self.device = device

        # init critic networks
        self.online_critic = CriticNetwork(
            input_dim,
            hidden_dim,
            hidden_layer_count,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
            head_count=head_count,
            encoder_layer_count=encoder_layer_count,
        )
        self.target_critic = CriticNetwork(
            input_dim,
            hidden_dim,
            hidden_layer_count,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
            head_count=head_count,
            encoder_layer_count=encoder_layer_count,
        )

        # move networks to device
        self.online_critic.to(device)
        self.target_critic.to(device)

        # copy parameters to target networks
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # initialize optimizer
        self.optimizer = optim.Adam(self.online_critic.parameters(), lr=learning_rate)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=learning_rate_decay
        )

        # set eval
        self.target_critic.eval()

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        use_target: bool,
        with_gradient: bool = False,
    ) -> torch.Tensor:
        with torch.set_grad_enabled(with_gradient):
            critic_net = self.target_critic if use_target else self.online_critic
            q_value = critic_net(state, action)
        return q_value

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def update(self, tau):
        soft_update(self.target_critic, self.online_critic, tau)

    def eval(self):
        self.online_critic.eval()

    def train(self):
        self.online_critic.train()

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        torch.save(
            self.online_critic.state_dict(),
            os.path.join(directory, "shared_critic.pth"),
        )
        torch.save(
            self.target_critic.state_dict(),
            os.path.join(directory, "shared_target_critic.pth"),
        )
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(directory, "shared_critic_optimizer.pth"),
        )

    def load(self, directory: str):
        self.online_critic.load_state_dict(
            torch.load(
                os.path.join(directory, "shared_critic.pth"),
                map_location=self.device,
            )
        )
        self.target_critic.load_state_dict(
            torch.load(
                os.path.join(directory, "shared_target_critic.pth"),
                map_location=self.device,
            )
        )
        self.optimizer.load_state_dict(
            torch.load(
                os.path.join(directory, "shared_critic_optimizer.pth"),
                map_location=self.device,
            )
        )
