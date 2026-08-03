import argparse
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from algorithm import Algorithm
from environment import Environment


def save_figure(fig, figure_dir: str, stem: str, scale: float = 2.0) -> None:
    """Save an interactive figure and, when Kaleido is available, a PNG copy."""
    fig.write_html(os.path.join(figure_dir, f"{stem}.html"))
    try:
        fig.write_image(os.path.join(figure_dir, f"{stem}.png"), scale=scale)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Warning: could not write {stem}.png: {exc}")


def draw_training_curves(result_dir: str, figure_dir: str) -> None:
    df = pd.read_csv(os.path.join(result_dir, "training_data.csv"))
    x_col = df.columns[0]
    data_cols = df.columns[1:]
    df_smoothed = df[data_cols].rolling(window=100, min_periods=1).mean()

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, col in enumerate(data_cols):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=-df[col],
                mode="lines",
                name=f"{col} (Raw)",
                line={"color": color, "width": 1},
                opacity=0.25,
                legendgroup=col,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=-df_smoothed[col],
                mode="lines",
                name=col,
                line={"color": color, "width": 2.5},
                legendgroup=col,
            )
        )
    fig.update_yaxes(
        type="log", autorange="reversed", title_text="System cost (Log Scale)"
    )
    fig.update_xaxes(title_text=x_col)
    fig.update_layout(
        title="Training Trends",
        template="plotly_white",
        height=600,
        width=950,
        hovermode="x unified",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
    )
    save_figure(fig, figure_dir, "line_graph")

    bins = [0, 200, 400, 600, 800, 1000]
    labels = ["1-200", "200-400", "400-600", "600-800", "800-1000"]
    filtered = df[(df[x_col] >= 1) & (df[x_col] <= 1000)].copy()
    filtered["stage"] = pd.cut(filtered[x_col], bins=bins, labels=labels)
    melted = pd.melt(
        filtered,
        id_vars=["stage"],
        value_vars=data_cols,
        var_name="Algorithm",
        value_name="raw_value",
    )
    melted["value"] = -melted["raw_value"]
    fig = px.box(
        melted,
        x="stage",
        y="value",
        color="Algorithm",
        points="outliers",
        title="Algorithm Performance Comparison",
        labels={"stage": "Episode Range", "value": "System cost (Log Scale)"},
    )
    fig.update_yaxes(type="log", autorange="reversed")
    fig.update_layout(
        template="plotly_white",
        boxmode="group",
        height=600,
        width=950,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
    )
    save_figure(fig, figure_dir, "box_plot")

    last_100 = df[(df[x_col] > 900) & (df[x_col] <= 1000)].copy()
    melted = pd.melt(
        last_100, value_vars=data_cols, var_name="Algorithm", value_name="raw_value"
    )
    melted["value"] = -melted["raw_value"]
    fig = px.violin(
        melted,
        x="Algorithm",
        y="value",
        color="Algorithm",
        box=True,
        points="all",
        title="Final 100 Episodes Reward Distribution",
        labels={"value": "System cost (Log Scale)"},
    )
    fig.update_yaxes(type="log", autorange="reversed")
    fig.update_layout(template="plotly_white", height=600, width=900, showlegend=False)
    save_figure(fig, figure_dir, "violin_plot")


def load_models(
    result_dir: str, device: torch.device
) -> tuple[dict[str, Algorithm], dict]:
    """Load every saved algorithm and the shared environment configuration."""
    checkpoint_root = os.path.join(result_dir, "checkpoint")
    env_config_path = os.path.join(checkpoint_root, "environment.json")
    with open(env_config_path, encoding="utf-8") as config_file:
        env_config = json.load(config_file)

    models = {}
    for name in os.listdir(checkpoint_root):
        model_config_path = os.path.join(checkpoint_root, name, "config.json")
        if not os.path.isfile(model_config_path):
            continue
        with open(model_config_path, encoding="utf-8") as config_file:
            model_config = json.load(config_file)
        model_config["dtype"] = getattr(torch, model_config["dtype"])
        model = Algorithm(device=device, **model_config)
        model.load(os.path.dirname(model_config_path))
        model.eval()
        models[name] = model

    if not models:
        raise RuntimeError(f"No algorithm configs found in {checkpoint_root}")
    return models, env_config


def evaluate_trace(model: Algorithm, env_config: dict) -> tuple[np.ndarray, Environment]:
    """Evaluate one model in a fresh copy of the common environment."""
    env = Environment(**env_config)
    state, _ = env.reset(options={"render_mode": "trace"})

    for _ in range(env.max_steps):
        state_tensor = model.utils.np_to_tensor(state)
        with torch.no_grad():
            actions_tensor = torch.cat(
                [
                    agent.act(state_tensor[i : i + 1])
                    for i, agent in enumerate(model.agents)
                ],
                dim=0,
            )
        state, _, terminated, truncated, _ = env.step(actions_tensor.cpu().numpy())
        if terminated or truncated:
            break

    return np.asarray(env.uav_position_history), env


def draw_uav_trace_comparison(
    result_dir: str, figure_dir: str, device: torch.device
) -> None:
    models, env_config = load_models(result_dir, device)
    data_columns = pd.read_csv(
        os.path.join(result_dir, "training_data.csv"), nrows=0
    ).columns[1:]
    algorithm_names = [name for name in data_columns if name in models]
    algorithm_names.extend(name for name in models if name not in algorithm_names)

    traces = {}
    environments = {}
    for name in algorithm_names:
        traces[name], environments[name] = evaluate_trace(models[name], env_config)

    reference_env = environments[algorithm_names[0]]
    figure = go.Figure()

    ue_positions = np.asarray([ue.position for ue in reference_env.ues])
    edge_positions = np.asarray([edge.position for edge in reference_env.edges])
    figure.add_trace(
        go.Scatter(
            x=ue_positions[:, 0],
            y=ue_positions[:, 1],
            mode="markers",
            name="UE",
            marker={"size": 7, "color": "dodgerblue", "opacity": 0.8},
            legend="legend2",
            hovertemplate="UE<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=edge_positions[:, 0],
            y=edge_positions[:, 1],
            mode="markers",
            name="Edge",
            marker={"size": 12, "color": "black", "symbol": "square"},
            legend="legend2",
            hovertemplate="Edge<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
        )
    )

    colors = px.colors.qualitative.Plotly
    line_styles = ["solid", "dash", "dashdot", "dot"]
    for algorithm_id, name in enumerate(algorithm_names):
        trace = traces[name]
        color = colors[algorithm_id % len(colors)]
        for uav_id in range(trace.shape[1]):
            figure.add_trace(
                go.Scatter(
                    x=trace[:, uav_id, 0],
                    y=trace[:, uav_id, 1],
                    mode="lines",
                    name=f"{name} / UAV {uav_id}",
                    line={
                        "color": color,
                        "dash": line_styles[uav_id % len(line_styles)],
                        "width": 1.4,
                    },
                    opacity=0.9,
                    showlegend=False,
                    hovertemplate=(
                        f"{name} / UAV {uav_id}<br>"
                        "x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>"
                    ),
                )
            )

    initial_positions = next(iter(traces.values()))[0]
    figure.add_trace(
        go.Scatter(
            x=initial_positions[:, 0],
            y=initial_positions[:, 1],
            mode="markers",
            name="UAV start",
            marker={
                "size": 10,
                "symbol": "circle-open",
                "color": "black",
                "line": {"color": "black", "width": 2},
            },
            legend="legend2",
            hovertemplate="UAV start<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
        )
    )

    for algorithm_id, name in enumerate(algorithm_names):
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=name,
                line={"color": colors[algorithm_id % len(colors)], "width": 3},
            )
        )
    uav_count = next(iter(traces.values())).shape[1]
    for uav_id in range(uav_count):
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"UAV {uav_id}",
                line={
                    "color": "black",
                    "dash": line_styles[uav_id % len(line_styles)],
                    "width": 2,
                },
                legend="legend2",
            )
        )

    figure.update_xaxes(
        title="X (m)", range=[0, reference_env.area_size[0]], constrain="domain"
    )
    figure.update_yaxes(
        title="Y (m)",
        range=[0, reference_env.area_size[1]],
        scaleanchor="x",
        scaleratio=1,
    )
    figure.update_layout(
        template="plotly_white",
        width=512,
        height=512,
        margin={"l": 70, "r": 45, "t": 85, "b": 60},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.08,
            "yanchor": "bottom",
        },
        legend2={
            "x": 0.99,
            "xanchor": "right",
            "y": 0.99,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "lightgray",
            "borderwidth": 1,
            "font": {"size": 10},
        },
    )
    save_figure(figure, figure_dir, "uav_trace_comparison")

    for env in environments.values():
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results", help="Result directory")
    parser.add_argument(
        "--device", default="cpu", help="PyTorch device, e.g. cpu or cuda"
    )
    args = parser.parse_args()

    figure_dir = os.path.join(args.dir, "figure")
    os.makedirs(figure_dir, exist_ok=True)
    draw_training_curves(args.dir, figure_dir)
    draw_uav_trace_comparison(args.dir, figure_dir, torch.device(args.device))
    print("Draw Complete")


if __name__ == "__main__":
    main()
