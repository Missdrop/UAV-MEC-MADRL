import json
import os

import numpy as np
import torch

from agents import Agent
from memory_replay import MultiAgentBuffer
from networks import Critic


class Utils:
    def __init__(self, device: torch.device, dtype=torch.float32):
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
        # device
        device: torch.device,
        # agent parameters
        agent_count: int,
        observation_dim: int,
        action_dim: int,
        actor_hidden_dim: int = 64,
        actor_hidden_layer_count: int = 2,
        critic_hidden_dim: int = 64,
        critic_hidden_layer_count: int = 2,
        actor_lr: float = 1e-4,
        actor_lr_decay: float = 0.9999,
        critic_lr: float = 1e-3,
        critic_lr_decay: float = 0.9999,
        critic_count: int = 2,
        gamma: float = 0.99,  # discount factor
        tau: float = 1e-4,  # soft update factor
        # MADDPG & MATD3
        action_noise: float = 0.1,
        # MATD3
        target_policy_noise: float = 0.2,
        target_noise_bound: float | None = 0.5,
        policy_delay_step: int = 2,  # update actor and target networks every n steps
        # memory replay parameters
        buffer_size: int = 100000,
        batch_size: int = 256,
        dtype: torch.dtype = torch.float32,
        # optional
        use_shared_critics: bool = True,
        parameter_noise: float = 0.1,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.2,
        head_count: int = 0,
        encoder_layer_count: int = 0,
        seed: int | None = None,
    ):

        # record parameters
        init_config = locals().copy()
        init_config.pop("self")
        init_config.pop("__class__", None)
        init_config["dtype"] = str(dtype).removeprefix("torch.")
        init_config.pop("device")
        self.config = init_config

        self.utils = Utils(dtype=dtype, device=device)
        self.agent_count = agent_count
        self.device = device
        self.action_noise = action_noise
        self.gamma = gamma
        self.tau = tau
        # MATD3 only
        self.target_policy_noise = target_policy_noise
        self.target_noise_bound = target_noise_bound
        self.policy_delay_step = policy_delay_step
        # exploratory actor
        self.parameter_noise = parameter_noise

        critic_input_dim = observation_dim + action_dim
        if encoder_layer_count <= 0:
            critic_input_dim *= agent_count

        # init shared critic
        self.shared_critics = (
            [
                Critic(
                    input_dim=critic_input_dim,
                    hidden_dim=critic_hidden_dim,
                    hidden_layer_count=critic_hidden_layer_count,
                    learning_rate=critic_lr,
                    learning_rate_decay=critic_lr_decay,
                    dropout_rate=dropout_rate,
                    layer_norm=use_layer_norm,
                    head_count=head_count,
                    encoder_layer_count=encoder_layer_count,
                    device=device,
                )
                for _ in range(critic_count)
            ]
            if use_shared_critics
            else None
        )

        # init agents
        self.agents = [
            Agent(
                id=id,
                critic_input_dim=critic_input_dim,
                state_dim=observation_dim,
                action_dim=action_dim,
                actor_hidden_dim=actor_hidden_dim,
                actor_hidden_layer_count=actor_hidden_layer_count,
                critic_hidden_dim=critic_hidden_dim,
                critic_hidden_layer_count=critic_hidden_layer_count,
                critic_count=critic_count,
                actor_lr=actor_lr,
                actor_lr_decay=actor_lr_decay,
                critic_lr=critic_lr,
                critic_lr_decay=critic_lr_decay,
                gamma=gamma,
                tau=tau,
                dropout_rate=dropout_rate,
                layer_norm=use_layer_norm,
                head_count=head_count,
                encoder_layer_count=encoder_layer_count,
                device=device,
                shared_critics=self.shared_critics,
            )
            for id in range(agent_count)
        ]

        # init buffer
        self.buffer = MultiAgentBuffer(
            capacity=buffer_size,
            agent_count=agent_count,
            state_dim=observation_dim,
            action_dim=action_dim,
            seed=seed,
        )
        self.batch_size = batch_size

        # count the training step
        self.train_step = 0
        self.done = True
        self.state: np.ndarray = np.zeros(0)
        self.evaluate_done = True
        self.evaluate_state: np.ndarray = np.zeros(0)

    def explore(
        self,
        env,
        render: bool = False,
        evaluate: bool = False,
        use_param_noise: bool = False,
    ):
        if evaluate:
            if self.evaluate_done:
                self.evaluate_state, _ = env.reset()
                self.evaluate_done = False
            state = self.evaluate_state
        else:
            if self.done:
                self.state, _ = env.reset()
                self.done = False
            state = self.state

        # calculate action
        # state_tensor shape: [agent_count, state_dim]
        state_tensor = self.utils.np_to_tensor(state)
        # each action shape: [1, action_dim]
        action_noise = 0.0 if evaluate else self.action_noise
        param_noise = 0.0 if evaluate or not use_param_noise else self.parameter_noise
        actions = [
            agent.act(
                state_tensor[i : i + 1],
                action_noise=action_noise,
                param_noise=param_noise,
                explore=not evaluate,
                use_target=False,
            )
            for i, agent in enumerate(self.agents)
        ]
        # actions shape: [agent_count, action_dim]
        actions = self.utils.tensor_to_np(torch.cat(actions, dim=0))

        # take action
        next_state, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # push transition to buffer
        if not evaluate:
            self.buffer.push(state, actions, reward, next_state, done)

        # update current state
        if evaluate:
            self.evaluate_state = next_state
            self.evaluate_done = done
        else:
            self.state = next_state
            self.done = done

        image = env.render() if render else None

        return reward, done, image, info

    def step(self):
        # if not enough samples in buffer, don't train
        if len(self.buffer) < self.batch_size:
            return None

        # update training step
        self.train_step += 1

        # --- 1. Get training data ---

        # raw_states, raw_actions, raw_next_states shape: [batch_size, n_agents, other_dim]
        # reward, dones shape: [batch_size, 1]
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # convert into tensors
        states = self.utils.np_to_tensor(states)
        actions = self.utils.np_to_tensor(actions)
        rewards = self.utils.np_to_tensor(rewards)
        next_states = self.utils.np_to_tensor(next_states)
        dones = self.utils.np_to_tensor(dones)

        # calculate next actions
        # shape: [agent_count, [batch_size, action_dim]]
        next_actions_list = [
            agent.act(
                next_states[:, i, :],
                action_noise=self.target_policy_noise,
                action_noise_limit=self.target_noise_bound,
                use_target=True,
                with_gradient=False,
            )
            for i, agent in enumerate(self.agents)
        ]
        # -> [batch_size, n_agents, action_dim]
        next_actions = torch.stack(next_actions_list, dim=1)

        # --- 2. Training process ---

        # 1. Train critic
        critic_losses: list[float] = []
        if self.shared_critics:
            # since using shared critic, use agents[0].critic
            critic_loss = self.agents[0].step_critic(
                states, actions, rewards, next_states, next_actions, dones
            )
            critic_losses.append(critic_loss)
        else:
            for agent in self.agents:
                critic_loss = agent.step_critic(
                    states, actions, rewards, next_states, next_actions, dones
                )
                critic_losses.append(critic_loss)

        # 2. Train actor
        actor_losses: list[float] = []
        if self.train_step % self.policy_delay_step == 0:
            for id, agent in enumerate(self.agents):
                # build a joint action list first
                # the current agent's action is calculated with gradient,
                # others with no gradient
                # shape: [agent_count, batch_size, action_dim]
                joint_actions_list: list[torch.Tensor] = []
                for chosen_id, chosen_agent in enumerate(self.agents):
                    joint_actions_list.append(
                        chosen_agent.act(
                            states[:, chosen_id, :],
                            with_gradient=id == chosen_id,
                        )
                    )

                # shape: [batch, agent_count, action_dim]
                joint_actions = torch.stack(joint_actions_list, dim=1)

                # actor loss = -Q
                actor_loss = self.agents[id].step_actor(states, joint_actions)

                actor_losses.append(actor_loss)

            # 3. Update shadow networks
            for agent in self.agents:
                agent.update()

            # if using shared critics
            if self.shared_critics:
                for critic in self.shared_critics:
                    critic.update(self.tau)

        return {
            "critic_loss": critic_losses,
            "actor_loss": actor_losses,
        }

    def eval(self):
        for agent in self.agents:
            agent.eval()

    def train(self):
        for agent in self.agents:
            agent.train()

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
        for id, agent in enumerate(self.agents):
            agent.save(os.path.join(directory, "agent" + str(id)))
        if self.shared_critics:
            for id, critic in enumerate(self.shared_critics):
                critic.save(os.path.join(directory, "shared_critic" + str(id)))

    def load(self, directory: str):
        for id, agent in enumerate(self.agents):
            agent.load(os.path.join(directory, "agent" + str(id)))
        if self.shared_critics:
            for id, critic in enumerate(self.shared_critics):
                critic.load(os.path.join(directory, "shared_critic" + str(id)))
