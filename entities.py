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
        distance: float,
        angle: float,  # degree
        area_size: tuple[float, float],
    ):
        """
        Move the UAV with distance, angle.
        Clip the new position to the map boundary.
        """
        new_x = self.position[0] + distance * math.cos(math.radians(angle))
        new_y = self.position[1] + distance * math.sin(math.radians(angle))

        # Rejecting the whole move at a boundary makes many consecutive
        # actions produce exactly the same frame and gives the policy a flat transition.
        # Clipping preserves the feasible component of the move.
        self.position[0] = np.clip(new_x, 0.0, area_size[0])
        self.position[1] = np.clip(new_y, 0.0, area_size[1])


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
    ):
        """
        UE generates a random task,
        with cpu cycles per bit and data size range
        """
        self.task = (
            float(np.random.randint(cpu_cycles_range[0], cpu_cycles_range[1] + 1)),
            float(np.random.randint(data_size_range[0], data_size_range[1] + 1)),
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
