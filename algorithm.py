import torch
import numpy as np
from agents import Agent
from memory_replay import MultiAgentBuffer


class Algorithm:
    def __init__(
        self,
        # algorithm
        algorithm: str,  # "MADDPG" or "MATD3"
        # agent parameters
        agent_count: int,
        state_dim: int,
        action_dim: int,
        actor_hidden_dim: int = 64,
        actor_hidden_layer_count: int = 2,
        critic_hidden_dim: int = 64,
        critic_hidden_layer_count: int = 2,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        # MADDPG & MATD3
        exploration_noise: float = 0.1,
        # MATD3
        target_policy_noise: float = 0.2,
        policy_delay_step: int = 2,  # update actor and target networks every n steps
        # memory replay parameters
        buffer_size: int = 100,
        batch_size: int = 1000000,
        # device
        device: torch.device = torch.device("cpu"),
    ):
        self.agent_count = agent_count
        self.action_dim = action_dim
        self.device = device
        self.exploration_noise = exploration_noise
        # MATD3 only
        self.target_policy_noise = target_policy_noise if algorithm == "MATD3" else 0.0
        self.policy_delay_step = policy_delay_step if algorithm == "MATD3" else 1

        # init agents
        self.agents = [
            Agent(
                id=id,
                state_dim=state_dim,
                action_dim=action_dim,
                actor_hidden_dim=actor_hidden_dim,
                actor_hidden_layer_count=actor_hidden_layer_count,
                agent_count=agent_count,
                critic_hidden_dim=critic_hidden_dim,
                critic_hidden_layer_count=critic_hidden_layer_count,
                gamma=gamma,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                tau=tau,
                device=device,
                critic_count=1 if algorithm == "DDPG" else 2,
            )
            for id in range(agent_count)
        ]

        # init buffer
        self.buffer = MultiAgentBuffer(
            capacity=buffer_size,
            agent_count=agent_count,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        self.batch_size = batch_size

        # count the training step
        self.step = 0

    def choose_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        actions = [
            agent.calculate_action(
                torch.Tensor(state), use_target=True, with_gradiant=False
            )
            for _, agent in enumerate(self.agents)
        ]
        return np.array(actions)  # shape: [agent_count, action_dim]

    def train(self):
        # if not enough samples in buffer, don't train
        if len(self.buffer) < self.batch_size:
            return

        # update training step
        self.step += 1

        # --- 1. Get training data ---

        # raw_states, raw_actions, raw_next_states shape: [batch_size, n_agents, other_dim]
        # reward, dones shape: [batch_size, 1]
        raw_states, raw_actions, rewards, raw_next_states, dones = self.buffer.sample(
            self.batch_size
        )

        # flatten the tensors
        # shape: [batch_size, other_dim]
        states = raw_states.view(self.batch_size, -1)
        next_states = raw_next_states.view(self.batch_size, -1)
        actions = raw_actions.view(self.batch_size, -1)

        # calculate next actions
        # shape: [agent_count, batch_size, action_dim]
        next_actions = [
            agent.calculate_action(
                raw_next_states[:, i, :], use_target=True, with_gradiant=False
            )
            for i, agent in enumerate(self.agents)
        ]
        # concat by action dimension
        # shape: [batch_size, agent_count * action_dim]
        next_actions = torch.cat(next_actions, dim=1)

        # --- 2. Training process ---

        # Train critic
        for _, agent in enumerate(self.agents):
            critic_loss = agent.calculate_critic_loss(
                states, actions, rewards, next_states, next_actions, dones
            )
            agent.optimize_parameters(critic_loss, "critic")

        if self.step % self.policy_delay_step == 0:
            # Train actor
            for id, agent in enumerate(self.agents):
                # build a joint action list first
                # the current agent's action is calculated with gradient,
                # others with no gradient
                # shape: [agent_count, batch_size, action_dim]
                joint_actions_list = []
                for id_, agent_ in enumerate(self.agents):
                    if id == id_:
                        action = agent.calculate_action(
                            raw_states[:, id_, :],
                            noise_mu=self.exploration_noise,
                            use_target=False,
                            with_gradiant=True,
                        )
                    else:
                        action = agent_.calculate_action(
                            raw_states[:, id_, :],
                            noise_mu=self.exploration_noise,
                            use_target=False,
                            with_gradiant=False,
                        )
                    joint_actions_list.append(action)
                # shape: [batch_size, agent_count * action_dim]
                joint_actions = torch.cat(joint_actions_list, dim=1)

                # actor loss = -Q
                actor_loss = -agent.calculate_q_value(
                    states, joint_actions, use_target=False, with_gradiant=False
                ).mean()

                agent.optimize_parameters(actor_loss, "actor")

                # update parameters (actor and critic) to target networks
                agent.update_network_parameters()

    def save_checkpoint(self, directory: str):
        for agent in self.agents:
            agent.save_models(directory)

    def load_checkpoint(self, directory: str):
        for agent in self.agents:
            agent.load_models(directory)
