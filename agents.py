import os
import torch
import torch.optim as optim
import numpy as np
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(
        self,
        id: int,
        critic_input_dim: int,
        state_dim: int,
        action_dim: int,
        # network parameters
        actor_hidden_dim: int = 64,
        actor_hidden_layer_count: int = 2,
        critic_hidden_dim: int = 64,
        critic_hidden_layer_count: int = 2,
        # training parameters
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,  # discount factor
        tau: float = 0.005,  # soft update parameter
        noise: float = 0.1,  # exploration noise
    ):
        self.gamma = gamma
        self.tau = tau
        self.id = id
        self.noise = noise

        # initialize networks
        self.actor = ActorNetwork(
            state_dim,
            action_dim,
            actor_hidden_dim,
            actor_hidden_layer_count,
        )
        self.critic = CriticNetwork(
            critic_input_dim,
            critic_hidden_dim,
            critic_hidden_layer_count,
        )
        self.target_actor = ActorNetwork(
            state_dim,
            action_dim,
            actor_hidden_dim,
            actor_hidden_layer_count,
        )
        self.target_critic = CriticNetwork(
            critic_input_dim,
            critic_hidden_dim,
            critic_hidden_layer_count,
        )

        # copy parameters to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def to_device(self, device: torch.device):
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()
        self.noise = 0.0  # Disable noise during evaluation

    def choose_action(
        self, observation: np.ndarray, evaluate: bool = False
    ) -> np.ndarray:
        device = next(self.actor.parameters()).device

        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            actions = self.actor(state)[0]

        if not evaluate:
            noise = torch.randn_like(actions) * self.noise
            actions = actions + noise

        clipped_action = torch.clamp(actions, -1.0, 1.0)
        return clipped_action.cpu().numpy()

    def update_network_parameters(self):
        # Update target actor
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        # Update target critic
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
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
            self.critic.state_dict(),
            os.path.join(directory, f"agent{self.id}_critic.pth"),
        )
        torch.save(
            self.target_critic.state_dict(),
            os.path.join(directory, f"agent{self.id}_target_critic.pth"),
        )

    def load_models(self, directory: str):
        self.actor.load_state_dict(
            torch.load(os.path.join(directory, f"agent{self.id}_actor.pth"))
        )
        self.target_actor.load_state_dict(
            torch.load(os.path.join(directory, f"agent{self.id}_target_actor.pth"))
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(directory, f"agent{self.id}_critic.pth"))
        )
        self.target_critic.load_state_dict(
            torch.load(os.path.join(directory, f"agent{self.id}_target_critic.pth"))
        )
