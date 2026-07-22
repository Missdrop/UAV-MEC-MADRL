import os
import torch
import torch.optim as optim
import torch.nn as nn

from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(
        self,
        id: int,
        agent_count: int,
        state_dim: int,
        action_dim: int,
        # network parameters
        actor_hidden_dim: int = 64,
        actor_hidden_layer_count: int = 2,
        critic_hidden_dim: int = 64,
        critic_hidden_layer_count: int = 2,
        critic_count: int = 1,  # the count of critic network, e.g. 1 for DDPG, 2 for TD3
        # training parameters
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,  # discount factor
        tau: float = 0.005,  # soft update parameter
        noise_mu: float = 0.1,  # exploration noise (standard deviation of the gaussian distribution)
    ):
        self.gamma = gamma
        self.tau = tau
        self.id = id
        self.noise_mu = noise_mu

        # initialize networks
        self.actor = ActorNetwork(
            state_dim,
            action_dim,
            actor_hidden_dim,
            actor_hidden_layer_count,
        )
        self.target_actor = ActorNetwork(
            state_dim,
            action_dim,
            actor_hidden_dim,
            actor_hidden_layer_count,
        )
        # init critic networks
        critic_input_dim = agent_count * (state_dim + action_dim)
        self.critics = nn.ModuleList(
            [
                CriticNetwork(
                    critic_input_dim,
                    critic_hidden_dim,
                    critic_hidden_layer_count,
                )
                for _ in range(critic_count)
            ]
        )

        self.target_critics = nn.ModuleList(
            [
                CriticNetwork(
                    critic_input_dim,
                    critic_hidden_dim,
                    critic_hidden_layer_count,
                )
                for _ in range(critic_count)
            ]
        )

        # copy parameters to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critics.load_state_dict(self.critics.state_dict())

        # initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=critic_lr)

    def to(self, device: torch.device):
        self.actor.to(device)
        self.critics.to(device)
        self.target_actor.to(device)
        self.target_critics.to(device)

    def eval(self):
        self.actor.eval()
        self.critics.eval()
        self.target_actor.eval()
        self.target_critics.eval()
        self.noise = 0.0  # Disable noise during evaluation

    def calculate_action(
        self,
        observation: torch.Tensor,
        use_target: bool,
        noise_mu: float = 0.0,
        with_gradiant: bool = False,
    ) -> torch.Tensor:
        actor_net = self.target_actor if use_target else self.actor
        with torch.set_grad_enabled(with_gradiant):
            action = actor_net(observation)[0]

            # add gaussian noise with (mean = 0, standard deviation = noise_mu)
            if noise_mu > 0:
                noise = torch.randn_like(action) * noise_mu
                action = action + noise

        action = torch.clamp(action, -1.0, 1.0)

        return action

    def calculate_q_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        use_target: bool,
        with_gradiant: bool = False,
    ) -> torch.Tensor:
        with torch.set_grad_enabled(with_gradiant):
            critic_nets = self.target_critics if use_target else self.critics
            q_values = [critic_net(state, action) for critic_net in critic_nets]
            # choose the minimum Q value
            q_value = (
                torch.min(torch.stack(q_values), dim=0).values
                if len(q_values) > 1  # if DDPG, return Q value directly
                else q_values[0]
            )
        return q_value

    def update_network_parameters(self):
        with torch.no_grad():
            # Update target actor
            for target_param, param in zip(
                self.target_actor.parameters(), self.actor.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

            # Update target critic
            for target_param, param in zip(
                self.target_critics.parameters(), self.critics.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

    def save_models(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        torch.save(
            self.actor.state_dict(),
            os.path.join(directory, f"agent{self.id}_actor.pth"),
        )
        torch.save(
            self.target_actor.state_dict(),
            os.path.join(directory, f"agent{self.id}_target_actor.pth"),
        )
        torch.save(
            self.critics.state_dict(),
            os.path.join(directory, f"agent{self.id}_critic.pth"),
        )
        torch.save(
            self.target_critics.state_dict(),
            os.path.join(directory, f"agent{self.id}_target_critic.pth"),
        )

    def load_models(
        self, directory: str, map_location: torch.device = torch.device("cpu")
    ):
        self.actor.load_state_dict(
            torch.load(
                os.path.join(directory, f"agent{self.id}_actor.pth"),
                map_location=map_location,
            )
        )
        self.target_actor.load_state_dict(
            torch.load(
                os.path.join(directory, f"agent{self.id}_target_actor.pth"),
                map_location=map_location,
            )
        )
        self.critics.load_state_dict(
            torch.load(
                os.path.join(directory, f"agent{self.id}_critic.pth"),
                map_location=map_location,
            )
        )
        self.target_critics.load_state_dict(
            torch.load(
                os.path.join(directory, f"agent{self.id}_target_critic.pth"),
                map_location=map_location,
            )
        )
