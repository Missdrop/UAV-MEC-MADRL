import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from entities import UAV, UE, FogNode


class Environment(gym.Env):
    """
    This is a multi-agent environment on UAV offloading task, which contains:
        - multiple UE clusters
        - multiple UAVs
        - multiple fog devices
    Custom positions can be provided.
    But in this implemention, they are all in a fixed altitude (z-axis),
    and the UAVs can only move in the x-y plane.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        # render parameters
        render_mode: str | None = None,
        figsize: tuple[int, int] = (6, 6),
        # environment parameters
        area_size: tuple[float, float] = (600.0, 600.0),
        max_steps: int = 200,
        terminate_unconnection_persentage: float = 0,  # terminate if more than 30% UEs are unconnected, set 0 to disable
        bandwidth: float = 1.0,  # MHz
        noise_power: float = 1e-13,  # Watts
        unconnected_penalty_factor: float = 10.0,  # punish weight for unconnection (each pair of UE and UAV)
        coverage_threshold: int = 6,  # punish if more than n UEs are unconnected
        coverage_penalty_weight: float = 10.0,  # punish weight for unconnected UEs (with coverage_threshold)
        # UAV parameters
        uav_count: int = 2,
        uav_custom_position: list[tuple[float, float, float]] | None = None,
        uav_transmit_power: float = 5.0,  # Watts
        uav_cpu_speed: float = 3.0,  # GHz
        uav_cpu_power: float = 0.3,  # Watts
        uav_signal_radius: float = 300.0,  # meters
        uav_altitude: float = 60.0,  # meters
        # UAV movement parameters
        max_move_distance: float = 270.0,  # meters
        max_move_angle: float = 180.0,  # degrees
        # UE parameters
        ue_count: int = 20,
        ue_custom_position: list[tuple[float, float, float]] | None = None,
        ue_transmit_power: float = 0.1,  # Watts
        # UE task parameters
        cpu_cycles_range: tuple[int, int] = (100, 200),  # CPU cycles per bit
        data_size_range: tuple[int, int] = (1, 5),  # Mbits
        # UE cluster parameters
        cluster_count: int = 2,
        cluster_radius: float = 50.0,  # meters
        # Fog parameters
        fog_count: int = 2,
        fog_custom_position: list[tuple[float, float, float]] | None = None,
        fog_cpu_speed: float = 10.0,  # GHz
        fog_altitude: float = 100.0,  # meters
    ):
        super().__init__()
        self.render_mode = render_mode
        self.figsize = figsize
        self.fig = None
        self.ax = None

        self._is_closed = False
        self.max_steps = max_steps
        self.steps = 0

        self.area_size = area_size
        self.bandwidth = bandwidth
        self.noise_power = noise_power
        self.unconnected_penalty_factor = unconnected_penalty_factor
        self.coverage_threshold = coverage_threshold
        self.coverage_penalty_weight = coverage_penalty_weight
        self.terminate_unconnection_percentage = terminate_unconnection_persentage

        self.max_move_distance = max_move_distance
        self.max_move_angle = max_move_angle

        self.cpu_cycles_range = cpu_cycles_range
        self.data_size_range = data_size_range

        # Initialize UAVs, UEs, and Fog nodes
        self.uavs = self._init_uavs(
            uav_count,
            uav_transmit_power,
            uav_cpu_speed,
            uav_cpu_power,
            uav_signal_radius,
            uav_altitude,
            uav_custom_position,
        )
        self.ues = self._init_ues(
            ue_count, ue_transmit_power, ue_custom_position, cluster_count, cluster_radius
        )
        self.fogs = self._init_fogs(
            fog_count, fog_cpu_speed, fog_altitude, fog_custom_position
        )

        # record UAV initial positions for reset()
        self.initial_uav_positions: list[np.ndarray] = [
            uav.position.copy() for uav in self.uavs
        ]

        # Define action space and observation space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.uavs), 2 + len(self.ues)),  # 2 for distance and angle
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=max(self.area_size),
            shape=(len(self.uavs), 3),  # 3 for x, y, z positions of each UAV
            dtype=np.float32,
        )

    """
        Inner methods for initializing UAVs, UEs, and Fog nodes.
    """

    def _init_uavs(
        self,
        num_uavs: int,
        transmit_power: float,
        CPU_speed: float,
        CPU_power: float,
        signal_radius: float,
        uav_altitude: float,
        custom_positions: list[tuple[float, float, float]] | None = None,
    ) -> list[UAV]:
        """
        Initialize UAVs by a custom position list (Case 1),
        if the list not provided, initialize them randomly within the area (Case 2).
        """
        uavs = []
        if custom_positions is not None:
            # [Case 1]
            positions = custom_positions
        else:
            # [Case 2]
            positions = [
                (
                    float(np.random.uniform(0, self.area_size[0])),
                    float(np.random.uniform(0, self.area_size[1])),
                    uav_altitude,  # UAV has a fixed altitude
                )
                for _ in range(num_uavs)
            ]
        # create and add entities to the list
        for id, pos in enumerate(positions):
            uavs.append(
                UAV(
                    id,
                    pos,
                    transmit_power,
                    CPU_speed,
                    CPU_power,
                    signal_radius,
                )
            )
        return uavs

    def _generate_ue_clusters(
        self,
        num_ues: int,
        cluster_count: int,
        cluster_radius: float,
    ) -> list[tuple[float, float, float]]:
        """
        generate UE clusters with a given number of clusters and cluster radius
        """
        positions = []

        # generate cluster centers, ensure inside the boundary
        centers = [
            (
                float(
                    np.random.uniform(cluster_radius, self.area_size[0] - cluster_radius)
                ),
                float(
                    np.random.uniform(cluster_radius, self.area_size[1] - cluster_radius)
                ),
            )
            for _ in range(cluster_count)
        ]

        # distribute UEs to clusters
        ues_per_cluster = num_ues // cluster_count
        remainder = num_ues % cluster_count

        for i, (center_x, center_y) in enumerate(centers):
            # Distribute the remainder UEs among the first few clusters
            count = ues_per_cluster + (1 if i < remainder else 0)

            for _ in range(count):
                # use sqrt to ensure uniform distribution within the circle
                r = cluster_radius * math.sqrt(np.random.uniform(0, 1))
                theta = np.random.uniform(0, 2 * math.pi)

                x = round(center_x + r * math.cos(theta), 2)
                y = round(center_y + r * math.sin(theta), 2)

                positions.append((x, y, 0.0))

        return positions

    def _init_ues(
        self,
        num_ues: int,
        ue_transmit_power: float,
        custom_positions: list[tuple[float, float, float]] | None = None,
        cluster_count: int = 0,
        cluster_radius: float = 0.0,
    ) -> list[UE]:
        ues = []
        if custom_positions is not None:
            positions = custom_positions
        elif cluster_count > 0 and cluster_radius > 0.0:
            positions = self._generate_ue_clusters(num_ues, cluster_count, cluster_radius)
        else:
            positions = [
                (
                    float(np.random.uniform(0, self.area_size[0])),
                    float(np.random.uniform(0, self.area_size[1])),
                    0.0,  # UE height is always 0m (ground level)
                )
                for _ in range(num_ues)
            ]

        for id, pos in enumerate(positions):
            ues.append(UE(id, pos, ue_transmit_power))
        return ues

    def _init_fogs(
        self,
        num_fogs: int,
        fog_cpu_speed: float,
        fog_altitude: float,
        custom_positions: list[tuple[float, float, float]] | None = None,
    ) -> list[FogNode]:
        fogs = []
        if custom_positions is not None:
            positions = custom_positions
        else:
            positions = [
                (
                    float(np.random.uniform(0, self.area_size[0])),
                    float(np.random.uniform(0, self.area_size[1])),
                    fog_altitude,  # Fog node has a fixed altitude
                )
                for _ in range(num_fogs)
            ]

        for id, pos in enumerate(positions):
            fogs.append(FogNode(id, pos, fog_cpu_speed))
        return fogs

    """
        Helping inner methods
    """

    @staticmethod
    def _distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two positions
        """
        return float(np.linalg.norm(pos1 - pos2))

    def _channel_gain(self, distance: float) -> float:
        """
        Calculate the channel gain (dB)
        Use the Log distance path loss model.
        """
        distance = max(distance, 1e-8)  # Avoid log(0)
        return -50 - 20 * (math.log10(distance))

    def _data_rate(self, channel_gain: float, transmit_power: float) -> float:
        """
        Calculate the data rate (Mbps) using Shannon's formula.
            P = transmit_power * 10^(gain/10),
            N = 10^-13, SNR = P/N,
            rate = log2(1 + SNR)
        """
        signal_power = transmit_power * (10 ** (channel_gain / 10.0))
        snr = signal_power / self.noise_power
        return self.bandwidth * math.log2(1.0 + snr)

    def _update_connections(self) -> None:
        """
        Update the connections between UEs and UAVs.
        Find all UAVs within the signal range and bind the UE to the nearest one.
        """
        for ue in self.ues:
            ue.connected_uav_id = -1
            min_distance = float("inf")
            nearest_uav_id = -1

            for uav in self.uavs:
                dist = self._distance(ue.position, uav.position)
                # check if in the signal range and find the nearest UAV
                if dist <= uav.signal_radius and dist < min_distance:
                    min_distance = dist
                    nearest_uav_id = uav.id

            ue.connected_uav_id = nearest_uav_id

    @property
    def observation(self) -> np.ndarray:
        """
        Get the current observation for all UAVs.
        The observation contains:
            - UAV position (x, y, z) for each UAV
        Shape: [num_uavs, 3]
        """
        return np.array([uav.position.copy() for uav in self.uavs], dtype=np.float32)

    """
    System cost calculation
    System cost contains:
        1. Total Energy consumption E
        2. Total Time delay T
        3. Bottleneck Throughput Th_b for all UEs (the minimum among all UEs)
    System cost (U) = E + T + 1/Th_b
    """

    def _system_cost(self, actions: list[np.ndarray]) -> tuple[float, float, float]:
        total_energy = 0.0
        total_time = 0.0
        bottleneck_throughput = float("inf")

        for i, uav in enumerate(self.uavs):
            action = actions[i]
            # map [-1, 1] to [0, 1] for offload ratios
            offload_ratios = [action[j] / 2.0 + 0.5 for j in range(2, 2 + len(self.ues))]

            # choose the nearest fog node for each UAV
            assigned_fog = min(
                self.fogs, key=lambda fog: self._distance(uav.position, fog.position)
            )
            # calculate the data rate between UAV and Fog node
            dist_uav_fog = self._distance(uav.position, assigned_fog.position)
            rate_uav_fog = self._data_rate(
                self._channel_gain(dist_uav_fog), uav.transmit_power
            )

            for j, ue in enumerate(self.ues):
                # get task parameters
                if ue.task is None:
                    continue
                cpu_cycles_per_bit, data_size_mb = ue.task

                # get the offload ratio for current UE
                x = offload_ratios[j]

                # 1. UE -> UAV upload phase
                # calculate upload time and energy
                dist_ue_uav = self._distance(ue.position, uav.position)
                rate_ue_uav = self._data_rate(
                    self._channel_gain(dist_ue_uav), ue.transmit_power
                )
                time_upload = data_size_mb / rate_ue_uav
                energy_upload = ue.transmit_power * time_upload

                # 2. UAV local computation phase
                # calculate computation time and energy
                time_uav_compute = (
                    (x * cpu_cycles_per_bit * data_size_mb)
                    / uav.CPU_speed
                    * 1e-3  # convert GHz to MHz
                )
                energy_uav_compute = uav.CPU_power * time_uav_compute

                # 3. UAV -> Fog upload phase
                # calculate upload time and energy
                time_fog_transfer = ((1.0 - x) * data_size_mb) / rate_uav_fog
                energy_fog_transfer = uav.transmit_power * time_fog_transfer
                time_fog_compute = (
                    ((1.0 - x) * cpu_cycles_per_bit * data_size_mb)
                    / assigned_fog.CPU_speed
                    * 1e-3  # convert GHz to MHz
                )

                # update bottleneck throughput (Th_b)
                if x == 1.0:
                    bottleneck_throughput = min(bottleneck_throughput, rate_ue_uav)
                else:
                    bottleneck_throughput = min(
                        bottleneck_throughput, min(rate_ue_uav, rate_uav_fog)
                    )

                # calculate total energy and time with unconnected penalty
                penalty = (
                    1.0
                    if ue.connected_uav_id == uav.id
                    else self.unconnected_penalty_factor
                )

                total_energy += penalty * (
                    energy_upload + energy_uav_compute + energy_fog_transfer
                )
                total_time += penalty * (
                    time_upload + time_uav_compute + time_fog_transfer + time_fog_compute
                )

        return total_energy, total_time, bottleneck_throughput

    """
        Public methods
    """

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # reset step counter
        self.steps = 0

        # reset rendermode
        if options is not None:
            self.render_mode = options.get("render_mode", self.render_mode)

        # reset positions
        for idx, uav in enumerate(self.uavs):
            uav.position = self.initial_uav_positions[idx].copy()

        # reset tasks for UEs
        for ue in self.ues:
            ue.generate_task(self.cpu_cycles_range, self.data_size_range)

        # reset connections between UEs and UAVs
        self._update_connections()

        if self.render_mode == "human":
            self.render()

        # return the initial positions of UAVs
        return self.observation, {}

    def step(
        self, actions: list[np.ndarray]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment with the given actions (shape: [num_uavs, 2 + UEcount]).
        Each action contains:
            - distance: [-1, 1] -> [0, max_move_distance]
            - angle: [-1, 1] -> [-max_move_angle, max_move_angle]
            - offload ratios for each UE: [-1, 1] -> [0, 1]
        """
        # move UAVs according to the actions
        for idx, uav in enumerate(self.uavs):
            action = actions[idx]
            distance = (action[0] / 2.0 + 0.5) * self.max_move_distance
            angle = action[1] * self.max_move_angle
            uav.move(distance, angle, self.area_size)

        # update connections between UEs and UAVs
        self._update_connections()

        # calculate the total system cost U = E + T + 1/Th_b
        energy, time_delay, throughput = self._system_cost(actions)
        system_cost = energy + time_delay + (1.0 / throughput)

        # calculate unconnected penalty if the number of unconnected UEs exceeds the coverage threshold
        unconnected_count = sum(1 for ue in self.ues if ue.connected_uav_id == -1)
        if unconnected_count > self.coverage_threshold:
            system_cost += self.coverage_penalty_weight * unconnected_count

        # calculate reward and next observation
        reward = -1.0 * system_cost

        info = {
            "unconnected_count": unconnected_count,
            "total_energy": energy,
            "total_time": time_delay,
            "bottleneck_throughput": throughput,
        }

        if self.render_mode == "human":
            self.render()

        # check if terminated or truncated
        self.steps += 1
        terminated = (
            unconnected_count / len(self.ues) >= self.terminate_unconnection_percentage
            if self.terminate_unconnection_percentage > 0
            else False
        )
        truncated = self.steps >= self.max_steps

        return self.observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        # initialize the figure and axes
        if self.fig is None or self.ax is None:
            self.fig = Figure(figsize=self.figsize, dpi=200)
            self.ax = self.fig.add_subplot(1, 1, 1)

        # clear the previous frame
        self.ax.cla()

        # set the axis limits and labels
        self.ax.set_xlim(0, self.area_size[0])
        self.ax.set_ylim(0, self.area_size[1])
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True, linestyle="--", alpha=0.5)

        # render Fog nodes
        fog_x = [fog.position[0] for fog in self.fogs]
        fog_y = [fog.position[1] for fog in self.fogs]
        self.ax.scatter(
            fog_x, fog_y, marker="s", s=100, c="black", label="Fog Node", zorder=5
        )

        # render UEs (connected and unconnected) with different markers
        connected_ue_x = [ue.position[0] for ue in self.ues if ue.connected_uav_id != -1]
        connected_ue_y = [ue.position[1] for ue in self.ues if ue.connected_uav_id != -1]
        unconnected_ue_x = [
            ue.position[0] for ue in self.ues if ue.connected_uav_id == -1
        ]
        unconnected_ue_y = [
            ue.position[1] for ue in self.ues if ue.connected_uav_id == -1
        ]

        self.ax.plot(
            connected_ue_x,
            connected_ue_y,
            "o",
            color="dodgerblue",
            markersize=5,
            label="Connected UE",
            zorder=4,
        )
        self.ax.plot(
            unconnected_ue_x,
            unconnected_ue_y,
            "rx",
            markersize=6,
            label="Unconnected UE",
            zorder=4,
        )

        # render UAVs
        uav_colors = ["darkorange", "purple", "green", "magenta"]

        for idx, uav in enumerate(self.uavs):
            color = uav_colors[idx % len(uav_colors)]

            # render UAV
            self.ax.scatter(
                uav.position[0],
                uav.position[1],
                marker="^",
                s=130,
                c=color,
                label=f"UAV {uav.id}",
                zorder=6,
            )

            # render UAV signal coverage circle
            circle = patches.Circle(
                (uav.position[0], uav.position[1]),
                uav.signal_radius,
                alpha=0.08,
                color=color,
            )
            self.ax.add_patch(circle)

            # render UE -> UAV
            for ue in self.ues:
                if ue.connected_uav_id == uav.id:
                    self.ax.plot(
                        [ue.position[0], uav.position[0]],
                        [ue.position[1], uav.position[1]],
                        color=color,
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.6,
                        zorder=2,
                    )

            # render UAV -> Fog
            assigned_fog = min(
                self.fogs, key=lambda f: self._distance(uav.position, f.position)
            )
            self.ax.plot(
                [uav.position[0], assigned_fog.position[0]],
                [uav.position[1], assigned_fog.position[1]],
                color="gray",
                linestyle="--",
                linewidth=1.2,
                alpha=0.5,
                zorder=3,
            )

        # render legend and adjust layout
        self.ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            fontsize=8,
            frameon=True,
        )

        self.fig.subplots_adjust(top=0.85, bottom=0.1, left=0.12, right=0.95)

        if self.render_mode == "human":
            plt.draw()
            return None
        elif self.render_mode == "rgb_array":
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            canvas = FigureCanvasAgg(self.fig)
            canvas.draw()
            image_rgba = canvas.buffer_rgba()
            return np.asarray(image_rgba)[:, :, :3]

    def close(self):
        self._is_closed = True

        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
