import gymnasium
import numpy as np
import torch
from agents import Agent
from memory_replay import MultiAgentBuffer
from utils import Utils


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
        buffer_size: int = 1000000,
        batch_size: int = 100,
        # device
        device: torch.device = torch.device("cpu"),
        utils: Utils = Utils(),
    ):
        self.utils = utils
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
                critic_count=1 if algorithm == "MADDPG" else 2,
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
        self.state = np.ndarray
        self.reward = 0.0

    def calculate_actions(
        self,
        states: torch.Tensor,  # shape: [batch_size, agent_count, state_dim]
        agent_id: int = 0,
        evaluate: bool = False,
        joint: bool = False,
        sample: bool = False,
    ) -> list[torch.Tensor]:
        """
        1. evaluate = True -> Current Actor, no grad, noise = 0.0
        2. joint = True: train Actor -> Current Actor, gradient ONLY for agent_id, noise = 0.0
        3. sample = True -> Current Actor, no grad, noise = exploration_noise
        4. all False: train Critic (Target Y) -> Target Actor, no grad, noise = target_policy_noise
        """
        # avoid undefined cases
        if evaluate and (joint or sample):
            raise ValueError("evaluate cannot be True when joint or sample is True")
        if sample and joint:
            raise ValueError("sample cannot be True when joint is True")

        # choose noise
        if evaluate or joint:
            noise_mu = 0.0
        elif sample:
            noise_mu = self.exploration_noise
        else:
            noise_mu = self.target_policy_noise

        use_target = False if (evaluate or joint or sample) else True

        actions_list = []
        for id, agent in enumerate(self.agents):
            with_gradient = True if (joint and id == agent_id) else False
            actions_list.append(
                agent.calculate_action(
                    states[:, id, :],
                    noise_mu=noise_mu,
                    use_target=use_target,
                    with_gradient=with_gradient,
                )
            )
        return actions_list  # shape: [agent_count, batch_size, action_dim]

    def explore(self, env: gymnasium.Env) -> tuple[float, bool]:
        """:return accumulation_reward, done"""
        state, _ = env.reset()
        done = False
        reward_sum = 0.0
        while not done:
            # calculate action
            # shape: [1, num_uavs, 3]
            state_tensor = self.utils.np_to_tensor(state).unsqueeze(0)
            # shape: [num_uavs, 1, action_dim]
            actions = self.calculate_actions(state_tensor, sample=True)
            actions = self.utils.tensor_to_np(torch.stack(actions, dim=0).squeeze(1))

            # take action
            next_state, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # push transition to buffer
            self.buffer.push(state, actions, reward, next_state, done)

            # update current state
            state = next_state
            reward_sum += reward

        return reward_sum

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
        next_actions = self.calculate_actions(
            states=raw_next_states,
        )
        # concat by action dimension
        # shape: [batch_size, agent_count * action_dim]
        next_actions = torch.cat(next_actions, dim=1)

        # --- 2. Training process ---

        # 1. Train critic
        for _, agent in enumerate(self.agents):
            critic_loss = agent.calculate_critic_loss(
                states, actions, rewards, next_states, next_actions, dones
            )
            agent.optimize_parameters(critic_loss, "critic")

        if self.step % self.policy_delay_step == 0:
            # 2. Train actor
            for id, agent in enumerate(self.agents):
                # build a joint action list first
                # the current agent's action is calculated with gradient,
                # others with no gradient
                # shape: [agent_count, batch_size, action_dim]
                joint_actions_list = self.calculate_actions(
                    states=raw_states,
                    agent_id=id,
                    joint=True,
                )
                # shape: [batch_size, agent_count * action_dim]
                joint_actions = torch.cat(joint_actions_list, dim=1)

                # actor loss = -Q
                actor_loss = -agent.calculate_q_value(
                    states, joint_actions, use_target=False, with_gradient=True
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
