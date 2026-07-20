import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a custom environment, which contains:
    - multiple UE clusters
    - multiple UAV
    - multiple fog devices
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, we need to define UAV, UE and fog device classes.
    """)
    return


@app.cell
def _():
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
            battery_capacity: float = 1000.0,  # Wh
        ):
            self.id = id
            self.position = np.array(position, dtype=np.float32)
            self.transmit_power = transmit_power
            self.CPU_speed = CPU_speed
            self.CPU_power = CPU_power
            self.signal_radius = signal_radius

            # extra parameters
            self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.remaining_battery = battery_capacity

        def move(
            self,
            distance: float,
            angle: float,  # radians
            area_size: tuple[float, float],
        ):
            """
            Move the UAV with distance, angle.
            If new position out of range, stay at the current position.
            """
            new_x = self.position[0] + distance * math.cos(angle)
            new_y = self.position[1] + distance * math.sin(angle)

            # check out of range
            if 0.0 <= new_x <= area_size[0] and 0.0 <= new_y <= area_size[1]:
                self.position[0] = new_x
                self.position[1] = new_y

    return (np,)


@app.cell
def _(np):
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

            self.task: tuple[float, float]  # (cpu_cycles_per_bit, data_size_megabits)
            self.connected_uav_id: int = -1

        def generate_task(
            self,
            cpu_cycles_range: tuple[int, int] = (100, 200),  # CPU cycles per bit range
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

    return


@app.cell
def _(np):
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

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then construct the environment:
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
