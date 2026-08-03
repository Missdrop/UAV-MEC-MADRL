import argparse
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from imageio.v3 import imwrite

from algorithm import Algorithm
from environment import Environment


def save_figure(fig, figure_dir: str, stem: str) -> None:
    """Save an interactive figure and, when Kaleido is available, a PNG copy."""
    fig.write_html(os.path.join(figure_dir, f"{stem}.html"))
    try:
        fig.write_image(os.path.join(figure_dir, f"{stem}.png"))
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


def load_attention_model(
    result_dir: str, device: torch.device
) -> tuple[Algorithm, Environment]:
    """Rebuild the training setup and load the DSPAC-Attn checkpoint."""
    checkpoint_root = os.path.join(result_dir, "checkpoint")
    env_config_path = os.path.join(checkpoint_root, "environment.json")
    with open(env_config_path, encoding="utf-8") as config_file:
        env = Environment(**json.load(config_file))

    attention_checkpoints = []
    for name in os.listdir(checkpoint_root):
        model_config_path = os.path.join(checkpoint_root, name, "config.json")
        if not os.path.isfile(model_config_path):
            continue
        with open(model_config_path, encoding="utf-8") as config_file:
            model_config = json.load(config_file)
        if model_config["encoder_layer_count"] > 0:
            attention_checkpoints.append((os.path.dirname(model_config_path), model_config))

    if len(attention_checkpoints) != 1:
        raise RuntimeError(
            "Expected exactly one attention checkpoint, found "
            f"{len(attention_checkpoints)}"
        )
    checkpoint, model_config = attention_checkpoints[0]
    model_config["dtype"] = getattr(torch, model_config["dtype"])
    model = Algorithm(device=device, **model_config)
    model.load(checkpoint)
    model.eval()
    return model, env


def draw_uav_trace(
    result_dir: str, figure_dir: str, device: torch.device
) -> None:
    model, env = load_attention_model(result_dir, device)
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

    trace_image = env.render()
    if trace_image is None:
        raise RuntimeError("Trace rendering did not return an image")
    imwrite(os.path.join(figure_dir, "uav_trace.png"), trace_image)
    step_count = len(env.action_history)
    env.close()
    print(f"Loaded {len(model.agents)}-agent attention model; traced {step_count} steps.")


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
    draw_uav_trace(args.dir, figure_dir, torch.device(args.device))


if __name__ == "__main__":
    main()
