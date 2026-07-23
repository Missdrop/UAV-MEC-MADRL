import torch
import numpy as np


class Utils:
    def __init__(self, dtype=torch.float32, device=torch.device("cpu")):
        self.dtype = dtype
        self.device = device

    @staticmethod
    def tensor_to_np(tensor: torch.Tensor):
        return tensor.detach().cpu().numpy()

    def np_to_tensor(self, np_array: np.ndarray):
        return torch.as_tensor(np_array, dtype=self.dtype, device=self.device)
