import numpy as np
import torch

from agents import Agent, SharedCritic
from memory_replay import MultiAgentBuffer


class Utils:
    def __init__(self, dtype=torch.float32, device=torch.device("cpu")):
        self.dtype = dtype
        self.device = device

    @staticmethod
    def tensor_to_np(tensor: torch.Tensor):
        return tensor.detach().cpu().numpy()

    def np_to_tensor(self, np_array: np.ndarray):
        return torch.as_tensor(np_array, dtype=self.dtype, device=self.device)

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
        target_noise_bound: float = 0.5,
        policy_delay_step: int = 2,  # update actor and target networks every n steps
        # memory replay parameters
        buffer_size: int = 1000000,
        batch_size: int = 100,
        # device
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        if algorithm not in ("MADDPG", "MATD3"):
            raise ValueError('algorithm must be "MADDPG" or "MATD3"')

        self.utils = Utils(dtype=dtype, device=device)
        self.agent_count = agent_count
        self.action_dim = action_dim
        self.device = device
        self.exploration_noise = exploration_noise
        # MATD3 only
        self.target_policy_noise = target_policy_noise if algorithm == "MATD3" else 0.0
        self.target_noise_bound = target_noise_bound if algorithm == "MATD3" else None
        self.policy_delay_step = policy_delay_step if algorithm == "MATD3" else 1

        # init shared critic
        self.shared_critic = SharedCritic(
            agent_count=agent_count,
            state_dim=state_dim,
            action_dim=action_dim,
            critic_hidden_dim=critic_hidden_dim,
            critic_hidden_layer_count=critic_hidden_layer_count,
            critic_count=1 if algorithm == "MADDPG" else 2,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            device=device,
        )

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
                critic_count=1 if algorithm == "MADDPG" else 2,
                shared_critic=self.shared_critic,
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
        self.done = True
        self.state: np.ndarray = np.zeros(0)
        self.evaluate_done = True
        self.evaluate_state: np.ndarray = np.zeros(0)

    def explore(
        self,
        env,
        render: bool = False,
        evaluate: bool = False,
    ):
        if evaluate:
            if self.evaluate_done:
                self.evaluate_state, _ = env.reset()
                self.evaluate_done = False
            current_state = self.evaluate_state
        else:
            if self.done:
                self.state, _ = env.reset()
                self.done = False
            current_state = self.state

        noise_mu = 0.0 if evaluate else self.exploration_noise

        # calculate action
        # state_tensor shape: [agent_count, state_dim]
        state_tensor = self.utils.np_to_tensor(current_state)
        # each action shape: [1, action_dim]
        actions = [
            agent.calculate_action(
                state_tensor[i : i + 1],
                noise_mu=noise_mu,
                use_target=False,
            )
            for i, agent in enumerate(self.agents)
        ]
        # actions shape: [agent_count, action_dim]
        actions = self.utils.tensor_to_np(torch.cat(actions, dim=0))

        # take action
        next_state, reward, terminated, truncated, info = env.step(actions)
        current_done = terminated or truncated

        # push transition to buffer
        if not evaluate:
            self.buffer.push(current_state, actions, reward, next_state, current_done)

        # update current state
        if evaluate:
            self.evaluate_state = next_state
            self.evaluate_done = current_done
        else:
            self.state = next_state
            self.done = current_done

        if render:
            return reward, current_done, env.render()

        return reward, current_done, None

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

        # convert into tensors
        raw_states = self.utils.np_to_tensor(raw_states)
        raw_actions = self.utils.np_to_tensor(raw_actions)
        rewards = self.utils.np_to_tensor(rewards)
        raw_next_states = self.utils.np_to_tensor(raw_next_states)
        dones = self.utils.np_to_tensor(dones)

        # flatten the tensors
        # shape: [batch_size, other_dim]
        states = raw_states.view(self.batch_size, -1)
        next_states = raw_next_states.view(self.batch_size, -1)
        actions = raw_actions.view(self.batch_size, -1)

        # calculate next actions
        # shape: [agent_count, batch_size, action_dim]
        next_actions = [
            agent.calculate_action(
                raw_next_states[:, i, :],
                noise_mu=self.target_policy_noise,
                noise_clip=self.target_noise_bound,
                use_target=True,
                with_gradient=False,
            )
            for i, agent in enumerate(self.agents)
        ]
        # concat by action dimension
        # shape: [batch_size, agent_count * action_dim]
        next_actions = torch.cat(next_actions, dim=1)

        # --- 2. Training process ---

        # 1. Train critic
        critic_loss = self.shared_critic.calculate_critic_loss(
            states, actions, rewards, next_states, next_actions, dones
        )
        self.shared_critic.optimize_parameters(critic_loss)

        if self.step % self.policy_delay_step == 0:
            # 2. Train actor
            self.shared_critic.set_gradient_enabled(False)
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
                            use_target=False,
                            with_gradient=True,
                        )
                    else:
                        action = agent_.calculate_action(
                            raw_states[:, id_, :],
                            use_target=False,
                            with_gradient=False,
                        )
                    joint_actions_list.append(action)

                # shape: [batch_size, agent_count * action_dim]
                joint_actions = torch.cat(joint_actions_list, dim=1)

                # actor loss = -Q
                actor_loss = -agent.calculate_q_value(
                    states, joint_actions, use_target=False, with_gradient=True
                ).mean()

                agent.optimize_parameters(actor_loss, "actor")
            self.shared_critic.set_gradient_enabled(True)

            # 3. Update target networks
            for agent in self.agents:
                agent.update_network_parameters()
            self.shared_critic.update_network_parameters()

    def save_checkpoint(self, directory: str):
        for agent in self.agents:
            agent.save_models(directory)
        self.shared_critic.save_models(directory)

    def load_checkpoint(self, directory: str):
        for agent in self.agents:
            agent.load_models(directory)
        self.shared_critic.load_models(directory)
