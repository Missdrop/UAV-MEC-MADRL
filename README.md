# UAV-MEC MADRL

This project provides a MADRL framework for UAV-MEC (Unmanned Aerial Vehicle - Mobile Edge Computing) systems.

It provides a simulation environment and various algorithms to train and evaluate multi-agent policies in UAV-MEC
scenarios.

## Installation

This project uses [**uv**](https://github.com/astral-sh/uv) to manage packages and projects, the environment can be
easily installed by:

```shell
$ uv sync --all-packages
```

> This command will automatically install torch+cu128 (CUDA 12.8) for Windows and linux users.
> If your GPU does not support [CUDA](https://developer.nvidia.com/cuda/toolkit) 12.8, please edit `pyproject.toml` to
> switch [torch version](https://pytorch.org/get-started/locally/).

## Run

This project widely uses Jupyter notebook, please make sure you have a Jupyter environment.