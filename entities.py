import numpy as np
import math


class UAV:
    def __init__(
        self,
        id: int,
        position: tuple[float, float, float],  # meters[x, y, z]
        transmit_power: float = 5.0,  # Watts
        CPU_speed: float = 3.0,  # GHz
        CPU_power: float = 0.3,  # Watts
        signal_radius: float = 300.0,  # meters
    ):
        self.id = id
        self.position = np.array(position, dtype=np.float32)
        self.transmit_power = transmit_power
        self.CPU_speed = CPU_speed
        self.CPU_power = CPU_power
        self.signal_radius = signal_radius

    def move(
        self,
        delta_x: float,
        delta_y: float,
        area_size: tuple[float, float],
    ) -> float:
        """
        Move by a Cartesian displacement and return the clipped distance.

        Returning the clipped component lets the environment penalize commands
        that repeatedly push into a map boundary.
        """
        requested = np.array([delta_x, delta_y], dtype=np.float32)
        old_xy = self.position[:2].copy()
        new_xy = old_xy + requested
        clipped_xy = np.clip(new_xy, [0.0, 0.0], area_size)
        self.position[:2] = clipped_xy
        return float(np.linalg.norm(new_xy - clipped_xy))


class UE:
    def __init__(
        self,
        id: int,
        position: tuple[float, float, float],
        transmit_power: float = 0.1,  # Watts
    ):
        self.id = id
        self.position = np.array(position, dtype=np.float32)
        self.transmit_power = transmit_power

        self.task: tuple[float, float] | None = None  # (CPU cycles/bit, data size Mbits)
        self.connected_uav_id: int = -1

    def generate_task(
        self,
        cpu_cycles_range: tuple[int, int] = (100, 200),  # CPU cycles per bit
        data_size_range: tuple[int, int] = (1, 5),  # Mbits
        rng: np.random.Generator | None = None,
    ):
        """
        UE generates a random task,
        with cpu cycles per bit and data size range
        """
        rng = rng if rng is not None else np.random.default_rng()
        self.task = (
            float(rng.integers(cpu_cycles_range[0], cpu_cycles_range[1] + 1)),
            float(rng.integers(data_size_range[0], data_size_range[1] + 1)),
        )


class FogNode:
    def __init__(
        self,
        id: int,
        position: tuple[float, float, float],
        CPU_speed: float = 10.0,  # GHz
    ):
        self.id = id
        self.position = np.array(position, dtype=np.float32)
        self.CPU_speed = CPU_speed
