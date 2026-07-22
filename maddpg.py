import torch
import numpy as np
from agents import Agent
from memory_replay import MultiAgentBuffer


class MADDPG:
    def __init__(
        self,
        agent_count: int,
        # agent parameters
        actor_input_dim: int,
        action_dim: int,
        actor_hidden_dim: int = 64,
        actor_hidden_layer_count: int = 2,
        critic_hidden_dim: int = 64,
        critic_hidden_layer_count: int = 2,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise: float = 0.1,
        # model save location
    ):
        self.agent_count = agent_count
        self.action_dim = action_dim

        critic_input_dim = agent_count * actor_input_dim + agent_count * action_dim

        self.agents: list[Agent] = []
        for id in range(self.agent_count):
            agent = Agent(
                id=id,
                actor_input_dim=actor_input_dim,
                action_dim=action_dim,
                actor_hidden_dim=actor_hidden_dim,
                actor_hidden_layer_count=actor_hidden_layer_count,
                critic_input_dim=critic_input_dim,
                critic_hidden_dim=critic_hidden_dim,
                critic_hidden_layer_count=critic_hidden_layer_count,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                tau=tau,
                noise=noise,
            )
            self.agents.append(agent)

    def to_device(self, device: torch.device):
        for agent in self.agents:
            agent.to_device(device)

    def eval(self):
        for agent in self.agents:
            agent.eval()

    def choose_action(self, raw_obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        actions = []
        for id, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[id], evaluate=evaluate)
            actions.append(action)
        return np.array(actions)  # shape: [agent_count, action_dim]

    @staticmethod
    def _train_critic(
        agent,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        next_actions: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            target_q = agent.target_critic(next_states, next_actions).squeeze(-1)

            target_q = (
                rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * agent.gamma * target_q
            )

        q = agent.critic(states, actions).squeeze(-1)
        critic_loss = torch.nn.functional.mse_loss(q, target_q)

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        return critic_loss.item()

    def _train_actor(
        self,
        agent_id: int,
        agent: Agent,
        raw_states: torch.Tensor,
        states: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> float:
        joint_actions = []

        for i, curr_agent in enumerate(self.agents):
            agent_obs = raw_states[:, i, :]
            if i == agent_id:
                # current agent, calculate action
                joint_actions.append(agent.actor(agent_obs))
            else:
                # other agent, sample action
                joint_actions.append(raw_actions[:, i, :])

        joint_actions_flat = torch.cat(joint_actions, dim=1)
        actor_loss = -agent.critic(states, joint_actions_flat).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        return actor_loss.item()

    def learn(self, memory: MultiAgentBuffer, batch_size: int = 100) -> None:
        if len(memory) < batch_size:
            return

        # states: [batch_size, n_agents, obs_dim]
        # actions: [batch_size, n_agents, action_dim]
        raw_states, raw_actions, rewards, raw_next_states, dones = memory.sample(
            batch_size
        )

        # shape: [batch_size, total_dim]
        states = raw_states.view(batch_size, -1)
        next_states = raw_next_states.view(batch_size, -1)
        actions = raw_actions.view(batch_size, -1)

        # calculate next actions
        next_actions = []
        for i, agent in enumerate(self.agents):
            next_state = raw_next_states[:, i, :]
            next_action = agent.target_actor(next_state)
            next_actions.append(next_action)
        next_actions = torch.cat(next_actions, dim=1)

        # Train critic
        for id, agent in enumerate(self.agents):
            self._train_critic(
                agent=agent,
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                next_actions=next_actions,
                dones=dones,
            )

        # Train actor
        for id, agent in enumerate(self.agents):
            self._train_actor(
                agent_id=id,
                agent=agent,
                raw_states=raw_states,
                states=states,
                raw_actions=raw_actions,
            )

        # update parameters
        for agent in self.agents:
            agent.update_network_parameters()

    def save_checkpoint(self, directory: str):
        for agent in self.agents:
            agent.save_models(directory)

    def load_checkpoint(self, directory: str):
        for agent in self.agents:
            agent.load_models(directory)
