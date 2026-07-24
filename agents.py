import os
import torch
import torch.optim as optim
import torch.nn as nn


from networks import ActorNetwork, CriticNetwork


class SharedCritic:
    def __init__(
        self,
        agent_count: int,
        state_dim: int,
        action_dim: int,
        critic_hidden_dim: int = 64,
        critic_hidden_layer_count: int = 2,
        critic_count: int = 1,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: torch.device = torch.device("cpu"),
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = device

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
        self.critics.to(device)
        self.target_critics.to(device)

        # copy parameters to target networks
        self.target_critics.load_state_dict(self.critics.state_dict())

        # initialize optimizer
        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=critic_lr)

    def calculate_q_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        use_target: bool,
        with_gradient: bool = False,
    ) -> torch.Tensor:
        with torch.set_grad_enabled(with_gradient):
            critic_nets = self.target_critics if use_target else self.critics
            q_values = [critic_net(state, action) for critic_net in critic_nets]
            # TD3 uses min(Q1, Q2) only for the bootstrapped target.
            q_value = (
                torch.min(torch.stack(q_values), dim=0).values
                if use_target and len(q_values) > 1
                else q_values[0]
            )
        return q_value  # shape: [batch_size, 1]

    def calculate_critic_loss(
        self, states, actions, rewards, next_states, next_actions, dones
    ):
        with torch.no_grad():
            # squeeze(-1) to remove the last 1 dimension
            # shape: [batch_size, 1] -> [batch_size]
            target_q = self.calculate_q_value(
                next_states, next_actions, use_target=True, with_gradient=False
            ).squeeze(-1)
            rewards = rewards.squeeze(-1)
            dones = dones.squeeze(-1)
            td_target = rewards + self.gamma * target_q * (1.0 - dones)

        # Each TD3 critic must be fitted independently.
        critic_losses = [
            torch.nn.functional.mse_loss(critic(states, actions).squeeze(-1), td_target)
            for critic in self.critics
        ]
        critic_loss = torch.stack(critic_losses).mean()

        return critic_loss

    def optimize_parameters(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def set_gradient_enabled(self, enabled: bool):
        for parameter in self.critics.parameters():
            parameter.requires_grad_(enabled)

    def update_network_parameters(self):
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critics.parameters(), self.critics.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

    def save_models(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        torch.save(
            self.critics.state_dict(),
            os.path.join(directory, "shared_critic.pth"),
        )
        torch.save(
            self.target_critics.state_dict(),
            os.path.join(directory, "shared_target_critic.pth"),
        )
        torch.save(
            self.critic_optimizer.state_dict(),
            os.path.join(directory, "shared_critic_optimizer.pth"),
        )

    def load_models(self, directory: str):
        self.critics.load_state_dict(
            torch.load(
                os.path.join(directory, "shared_critic.pth"),
                map_location=self.device,
                weights_only=True,
            )
        )
        self.target_critics.load_state_dict(
            torch.load(
                os.path.join(directory, "shared_target_critic.pth"),
                map_location=self.device,
                weights_only=True,
            )
        )
        optimizer_path = os.path.join(directory, "shared_critic_optimizer.pth")
        if os.path.exists(optimizer_path):
            self.critic_optimizer.load_state_dict(
                torch.load(
                    optimizer_path,
                    map_location=self.device,
                    weights_only=True,
                )
            )


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
        shared_critic: SharedCritic | None = None,
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
        self.shared_critic = shared_critic or SharedCritic(
            agent_count=agent_count,
            state_dim=state_dim,
            action_dim=action_dim,
            critic_hidden_dim=critic_hidden_dim,
            critic_hidden_layer_count=critic_hidden_layer_count,
            critic_count=critic_count,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            device=device,
        )
        self.critics = self.shared_critic.critics
        self.target_critics = self.shared_critic.target_critics
        # move networks to device
        self.actor.to(device)
        self.target_actor.to(device)

        # copy parameters to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())

        # initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = self.shared_critic.critic_optimizer

    def calculate_action(
        self,
        observation: torch.Tensor,
        use_target: bool,
        noise_mu: float = 0.0,
        noise_clip: float | None = None,
        with_gradient: bool = False,
    ) -> torch.Tensor:
        actor_net = self.target_actor if use_target else self.actor
        with torch.set_grad_enabled(with_gradient):
            action = actor_net(observation)

            # add gaussian noise with (mean = 0, standard deviation = noise_mu)
            if noise_mu > 0:
                noise = torch.randn_like(action) * noise_mu
                if noise_clip is not None:
                    noise = torch.clamp(noise, -noise_clip, noise_clip)
                action = action + noise

        action = torch.clamp(action, -1.0, 1.0)

        return action

    def calculate_q_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        use_target: bool,
        with_gradient: bool = False,
    ) -> torch.Tensor:
        return self.shared_critic.calculate_q_value(
            state, action, use_target, with_gradient
        )

    def calculate_critic_loss(
        self, states, actions, rewards, next_states, next_actions, dones
    ):
        return self.shared_critic.calculate_critic_loss(
            states, actions, rewards, next_states, next_actions, dones
        )

    def optimize_parameters(self, loss, network: str):
        """Optimize the parameters of the specified network. ("actor" or "critic")"""
        if network == "actor":
            optimizer = self.actor_optimizer
        elif network == "critic":
            self.shared_critic.optimize_parameters(loss)
            return
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
            self.actor_optimizer.state_dict(),
            os.path.join(directory, f"agent{self.id}_actor_optimizer.pth"),
        )

    def load_models(self, directory: str):
        self.actor.load_state_dict(
            torch.load(
                os.path.join(directory, f"agent{self.id}_actor.pth"),
                map_location=self.device,
                weights_only=True,
            )
        )
        self.target_actor.load_state_dict(
            torch.load(
                os.path.join(directory, f"agent{self.id}_target_actor.pth"),
                map_location=self.device,
                weights_only=True,
            )
        )
        optimizer_path = os.path.join(directory, f"agent{self.id}_actor_optimizer.pth")
        if os.path.exists(optimizer_path):
            self.actor_optimizer.load_state_dict(
                torch.load(
                    optimizer_path,
                    map_location=self.device,
                    weights_only=True,
                )
            )
