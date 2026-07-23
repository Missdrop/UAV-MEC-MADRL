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
        # device
        device: torch.device = torch.device("cpu"),
    ):
        self.gamma = gamma
        self.tau = tau
        self.id = id
        self.device = device

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
        # move networks to device
        self.actor.to(device)
        self.critics.to(device)
        self.target_actor.to(device)
        self.target_critics.to(device)

        # copy parameters to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critics.load_state_dict(self.critics.state_dict())

        # initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=critic_lr)

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

    def calculate_critic_loss(
        self, states, actions, rewards, next_states, next_actions, dones
    ):
        with torch.no_grad():
            # squeeze(-1) to remove the last 1 dimension
            # shape: [batch_size, 1] -> [batch_size]
            target_q = self.calculate_q_value(
                next_states, next_actions, use_target=True, with_gradiant=False
            ).squeeze(-1)
            td_target = rewards + self.gamma * target_q * (1.0 - dones)

        q = self.calculate_q_value(
            states, actions, use_target=False, with_gradiant=True
        ).squeeze(-1)
        critic_loss = torch.nn.functional.mse_loss(q, td_target)

        return critic_loss

    def optimize_parameters(self, loss, network: str):
        """Optimize the parameters of the specified network. ("actor" or "critic")"""
        if network == "actor":
            optimizer = self.actor_optimizer
        elif network == "critic":
            optimizer = self.critic_optimizer
        else:
            raise ValueError("Invalid network type")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    def load_models(self, directory: str):
        self.actor.load_state_dict(
            torch.load(
                os.path.join(directory, f"agent{self.id}_actor.pth"),
                map_location=self.device,
            )
        )
        self.target_actor.load_state_dict(
            torch.load(
                os.path.join(directory, f"agent{self.id}_target_actor.pth"),
                map_location=self.device,
            )
        )
        self.critics.load_state_dict(
            torch.load(
                os.path.join(directory, f"agent{self.id}_critic.pth"),
                map_location=self.device,
            )
        )
        self.target_critics.load_state_dict(
            torch.load(
                os.path.join(directory, f"agent{self.id}_target_critic.pth"),
                map_location=self.device,
            )
        )
