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

ALGORITHM_COLORS = {
    "MADDPG": "#4361EE",
    "MATD3": "#FF4D6D",
    "DSPAC": "#06B6A9",
    "DSPAC-Attn": "#9B5DE5",
}
FALLBACK_COLORS = px.colors.qualitative.Bold
PAPER_COLOR = "#F4F7FC"
PLOT_COLOR = "#FFFFFF"
GRID_COLOR = "#DCE4F0"
TEXT_COLOR = "#24324A"
HIGHLIGHT_TRACE_COLOR = "#9B5DE5"
HIGHLIGHT_TRACE_WIDTH = 4.0


def algorithm_color(name: str, index: int) -> str:
    return ALGORITHM_COLORS.get(name, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def apply_axes_style(fig) -> None:
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
        showline=True,
        linecolor="#AAB7C9",
        linewidth=1,
        ticks="outside",
        tickcolor="#AAB7C9",
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
        showline=True,
        linecolor="#AAB7C9",
        linewidth=1,
        ticks="outside",
        tickcolor="#AAB7C9",
    )


def apply_layout_style(fig) -> None:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=PLOT_COLOR,
        font={
            "family": "Arial, Microsoft YaHei, sans-serif",
            "size": 19,
            "color": TEXT_COLOR,
        },
        title=None,
        hoverlabel={
            "bgcolor": "white",
            "bordercolor": "#CBD5E1",
            "font": {"color": TEXT_COLOR, "size": 16},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "font": {"size": 18},
        },
    )
    apply_axes_style(fig)
    fig.update_xaxes(tickfont={"size": 18}, title_font={"size": 22})
    fig.update_yaxes(tickfont={"size": 18}, title_font={"size": 22})


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
    df_smoothed = df[data_cols].rolling(window=50, min_periods=1).mean()

    fig = go.Figure()
    for i, col in enumerate(data_cols):
        color = algorithm_color(col, i)
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=-df[col],
                mode="lines",
                name=f"{col} (Raw)",
                line={"color": color, "width": 0.8},
                opacity=0.3,
                legendgroup=col,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=-df_smoothed[col],
                mode="lines",
                line={"color": color, "width": 8},
                opacity=0.09,
                legendgroup=col,
                showlegend=False,
                hoverinfo="skip",
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
                hovertemplate=(
                    f"<b>{col}</b><br>Episode %{{x}}<br>"
                    "Smoothed cost %{y:.3f}<extra></extra>"
                ),
            )
        )
    fig.update_yaxes(
        type="log", autorange="reversed", title_text="System cost (Log Scale)"
    )
    fig.update_xaxes(title_text=x_col)
    fig.update_layout(
        title=None,
        height=650,
        width=820,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 100, "r": 35, "t": 85, "b": 80},
    )
    apply_layout_style(fig)
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
        color_discrete_map={
            name: algorithm_color(name, i) for i, name in enumerate(data_cols)
        },
        points="outliers",
        title=None,
        labels={"stage": "Episode Range", "value": "System cost (Log Scale)"},
    )
    fig.update_yaxes(type="log", autorange="reversed")
    fig.update_layout(
        boxmode="group",
        height=650,
        width=820,
        legend_title_text="",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 100, "r": 35, "t": 85, "b": 80},
    )
    fig.update_traces(
        opacity=0.78,
        line={"width": 1.6},
        marker={"size": 4, "opacity": 0.55, "line": {"width": 0.4}},
    )
    apply_layout_style(fig)
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
        color_discrete_map={
            name: algorithm_color(name, i) for i, name in enumerate(data_cols)
        },
        box=True,
        points="all",
        title=None,
        labels={"value": "System cost (Log Scale)"},
    )
    fig.update_yaxes(type="log", autorange="reversed")
    fig.update_traces(
        opacity=0.78,
        meanline_visible=True,
        line={"width": 1.8},
        marker={"size": 4, "opacity": 0.42},
        points="all",
        jitter=0.2,
    )
    fig.update_layout(
        height=700,
        width=700,
        showlegend=False,
        margin={"l": 100, "r": 35, "t": 45, "b": 85},
    )
    apply_layout_style(fig)
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
            marker={
                "size": 8,
                "color": "#111111",
                "opacity": 0.88,
                "line": {"color": "white", "width": 1},
            },
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
            marker={
                "size": 16,
                "color": "#17233C",
                "symbol": "square",
                "line": {"color": "white", "width": 1.2},
            },
            legend="legend2",
            hovertemplate="Edge<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
        )
    )

    line_styles = ["solid", "dash", "dashdot", "dot"]
    highlighted_algorithm = algorithm_names[-1]
    for algorithm_id, name in enumerate(algorithm_names):
        trace = traces[name]
        is_highlighted = name == highlighted_algorithm
        color = (
            HIGHLIGHT_TRACE_COLOR
            if is_highlighted
            else algorithm_color(name, algorithm_id)
        )
        for uav_id in range(trace.shape[1]):
            line_dash = line_styles[uav_id % len(line_styles)]
            if is_highlighted:
                figure.add_trace(
                    go.Scatter(
                        x=trace[:, uav_id, 0],
                        y=trace[:, uav_id, 1],
                        mode="lines",
                        line={
                            "color": "white",
                            "dash": line_dash,
                            "width": HIGHLIGHT_TRACE_WIDTH + 3.0,
                        },
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            figure.add_trace(
                go.Scatter(
                    x=trace[:, uav_id, 0],
                    y=trace[:, uav_id, 1],
                    mode="lines",
                    name=f"{name} / UAV {uav_id}",
                    line={
                        "color": color,
                        "dash": line_dash,
                        "width": HIGHLIGHT_TRACE_WIDTH if is_highlighted else 1.65,
                    },
                    opacity=1.0 if is_highlighted else 0.68,
                    showlegend=False,
                    hovertemplate=(
                        f"{name} / UAV {uav_id}<br>"
                        "x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>"
                    ),
                )
            )
            if is_highlighted:
                figure.add_trace(
                    go.Scatter(
                        x=[trace[-1, uav_id, 0]],
                        y=[trace[-1, uav_id, 1]],
                        mode="markers",
                        marker={
                            "size": 10,
                            "color": HIGHLIGHT_TRACE_COLOR,
                            "line": {"color": "white", "width": 2},
                        },
                        showlegend=False,
                        hovertemplate=(
                            f"{name} / UAV {uav_id} endpoint<br>"
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
                "size": 14,
                "symbol": "circle-open",
                "color": "black",
                "line": {"color": "black", "width": 2},
            },
            legend="legend2",
            hovertemplate="UAV start<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
        )
    )

    for algorithm_id, name in enumerate(algorithm_names):
        is_highlighted = name == highlighted_algorithm
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=name,
                line={
                    "color": (
                        HIGHLIGHT_TRACE_COLOR
                        if is_highlighted
                        else algorithm_color(name, algorithm_id)
                    ),
                    "width": 5 if is_highlighted else 3,
                },
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
        width=700,
        height=700,
        margin={"l": 65, "r": 35, "t": 75, "b": 55},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.02,
            "yanchor": "bottom",
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "#D7DFEA",
            "borderwidth": 0,
            "font": {"size": 16},
        },
        legend2={
            "x": 0.99,
            "xanchor": "right",
            "y": 0.99,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.94)",
            "bordercolor": "#D7DFEA",
            "borderwidth": 1,
            "font": {"size": 14},
        },
    )
    apply_layout_style(figure)
    figure.update_xaxes(mirror=True)
    figure.update_yaxes(mirror=True)
    save_figure(figure, figure_dir, "uav_trace_comparison")

    for env in environments.values():
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results", help="Result directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: input_dir/figure)",
    )
    parser.add_argument(
        "--device", default="cpu", help="PyTorch device, e.g. cpu or cuda"
    )
    parser.add_argument("--type", default="all", help="Type of figure to draw: all, curve, or trace")
    args = parser.parse_args()

    figure_dir = args.output_dir or os.path.join(args.dir, "figure")
    os.makedirs(figure_dir, exist_ok=True)
    if args.type in ("all", "curve"):
        draw_training_curves(args.dir, figure_dir)
    if args.type in ("all", "trace"):
        draw_uav_trace_comparison(args.dir, figure_dir, torch.device(args.device))
    print("Draw Complete")


if __name__ == "__main__":
    main()
