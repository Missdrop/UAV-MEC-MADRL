import os

import torch

from networks import Actor, Critic


class Agent:
    def __init__(
        self,
        # device
        device: torch.device,
        id: int,
        # network I/O dim
        state_dim: int,
        action_dim: int,
        critic_input_dim: int = 0,
        # network parameters
        actor_hidden_dim: int = 64,
        actor_hidden_layer_count: int = 2,
        critic_hidden_dim: int = 64,
        critic_hidden_layer_count: int = 2,
        critic_count: int = 1,  # the count of critic network, e.g. 1 for DDPG, 2 for TD3
        # training parameters
        actor_lr: float = 1e-4,
        actor_lr_decay: float = 0.9999,
        critic_lr: float = 1e-3,
        critic_lr_decay: float = 0.9999,
        gamma: float = 0.99,  # discount factor
        tau: float = 0.005,  # soft update parameter
        # use shared critic
        shared_critics: list[Critic] | None = None,
        # optional
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        self.gamma = gamma
        self.tau = tau
        self.id = id
        self.device = device

        self.shared_critic = bool(shared_critics)

        if not shared_critics and critic_input_dim == 0:
            raise ValueError("please set critic_input_dim")

        # initialize modules
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            hidden_layer_count=actor_hidden_layer_count,
            learning_rate=actor_lr,
            learning_rate_decay=actor_lr_decay,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
            device=device,
        )
        self.critics: list[Critic] = (
            shared_critics
            if shared_critics
            else [
                Critic(
                    input_dim=critic_input_dim,
                    hidden_dim=critic_hidden_dim,
                    hidden_layer_count=critic_hidden_layer_count,
                    learning_rate=critic_lr,
                    learning_rate_decay=critic_lr_decay,
                    layer_norm=layer_norm,
                    dropout_rate=dropout_rate,
                    device=device,
                )
                for _ in range(critic_count)
            ]
        )

        # directly use actor's method
        self.act = self.actor.forward

    # def act(
    #     self,
    #     observation: torch.Tensor,
    #     use_target: bool = False,
    #     explore: bool = False,  # use exploratory actor
    #     action_noise: float = 0.0,
    #     action_noise_limit: float | None = None,
    #     param_noise: float = 0.0,  # parameter noise for exploratory actor
    #     with_gradient: bool = False,
    # ) -> torch.Tensor:
    #     return self.actor.forward(
    #         observation,
    #         use_target=use_target,
    #         explore=explore,
    #         action_noise=action_noise,
    #         action_noise_limit=action_noise_limit,
    #         param_noise=param_noise,
    #         with_gradient=with_gradient
    #     )

    def action_loss(self, states, joint_actions) -> torch.Tensor:
        q_values = self.criticise(
            states, joint_actions, use_target=False, with_gradient=True
        )
        q_values = torch.stack([q_value.mean() for q_value in q_values])
        return -q_values.mean()

    def criticise(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        use_target: bool,
        with_gradient: bool = False,
    ) -> list[torch.Tensor]:
        score = []
        for critic in self.critics:
            score.append(
                critic.forward(
                    state, action, use_target=use_target, with_gradient=with_gradient
                ).squeeze(-1)  # convert [batch_size, 1] to [batch_size]
            )

        return score  # shape: [critic_count, batch_size]

    def td_error(
        self, states, actions, rewards, next_states, next_actions, dones
    ) -> list[torch.Tensor]:
        with torch.no_grad():
            target_q = self.criticise(
                next_states, next_actions, use_target=True, with_gradient=False
            )
            # choose the minimum from multiple outputs
            target_q = torch.min(torch.stack(target_q, dim=0), dim=0).values
            # squeeze(-1) to remove the last 1 dimension
            # shape: [batch_size, 1] -> [batch_size]
            rewards = rewards.squeeze(-1)
            dones = dones.squeeze(-1)
            td_target = rewards + self.gamma * target_q * (1.0 - dones)

        q_value = self.criticise(states, actions, use_target=False, with_gradient=True)

        # calculate mse loss for each critic
        losses = [
            torch.nn.functional.mse_loss(q_value[i], td_target)
            for i in range(len(self.critics))
        ]

        return losses

    def step_critic(
        self, states, actions, rewards, next_states, next_actions, dones
    ) -> float:
        losses = self.td_error(states, actions, rewards, next_states, next_actions, dones)
        loss = 0.0
        for id, critic in enumerate(self.critics):
            critic.step(losses[id])
            loss += float(losses[id].detach().cpu())
        return loss / len(losses)

    def step_actor(self, states, joint_actions) -> float:
        loss = self.action_loss(states, joint_actions)
        self.actor.step(loss)
        return float(loss.detach().cpu())

    def eval(self):
        self.actor.eval()
        for critic in self.critics:
            critic.eval()

    def train(self):
        self.actor.train()
        for critic in self.critics:
            critic.train()

    def update(self):
        """
        If shared critic used, this method will update actor ONLY
        """
        if not self.shared_critic:
            for critic in self.critics:
                critic.update(self.tau)
        self.actor.update(self.tau)

    def save(self, directory: str):
        self.actor.save(os.path.join(directory, "actor"))
        if not self.shared_critic:  # actor ONLY when using shared critic
            for id, critic in enumerate(self.critics):
                critic.save(os.path.join(directory, "critic" + str(id)))

    def load(self, directory: str):
        self.actor.load(os.path.join(directory, "actor"))
        if not self.shared_critic:  # actor ONLY when using shared critic
            for id, critic in enumerate(self.critics):
                critic.load(os.path.join(directory, "critic" + str(id)))
