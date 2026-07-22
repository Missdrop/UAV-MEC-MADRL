import numpy as np
import torch


class MultiAgentBuffer:
    def __init__(
        self,
        capacity: int,
        agent_count: int,
        state_dim: int,
        action_dim: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.device = device

        #  pre-allocate memory to avoid dynamic resizing
        self.states = np.zeros((capacity, agent_count, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, agent_count, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, agent_count, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.index = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        # sample and convert to torch tensors
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size