import numpy as np


class MultiAgentBuffer:
    def __init__(
        self,
        capacity: int,
        agent_count: int,
        state_dim: int,
        action_dim: int,
    ):
        self.capacity = capacity

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
        indices = np.random.randint(0, self.size, size=batch_size)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size