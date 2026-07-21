import numpy as np
import torch
from transitions import Transition


class Buffer:
    def __init__(
        self,
        capacity: int,
        state_dim: tuple[int, int],
        action_dim: tuple[int, int],
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.device = torch.device(device)

        #  pre-allocate memory to avoid dynamic resizing
        self.states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.index = 0
        self.size = 0

    def push(self, transition: Transition):
        self.states[self.index] = transition.state
        self.actions[self.index] = transition.action
        self.rewards[self.index] = transition.reward
        self.next_states[self.index] = transition.next_state
        self.dones[self.index] = transition.done

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