import os
import argparse

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="results", help="Result directory")
result_dir = parser.parse_args().dir

df = pd.read_csv(os.path.join(result_dir, "training_data.csv"))

figure_dir = os.path.join(result_dir, "figure")
os.makedirs(figure_dir, exist_ok=True)

"""
Line graph
"""

x_col = df.columns[0]
data_cols = df.columns[1:]

window_size = 100
df_smoothed = df[data_cols].rolling(window=window_size, min_periods=1).mean()

fig = go.Figure()

colors = px.colors.qualitative.Plotly

for i, col in enumerate(data_cols):
    color = colors[i % len(colors)]

    raw_y = -df[col]

    smooth_y = -df_smoothed[col]

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=raw_y,
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
            y=smooth_y,
            mode="lines",
            name=col,
            line={"color": color, "width": 2.5},
            legendgroup=col,
        )
    )

fig.update_yaxes(type="log", autorange="reversed", title_text="System cost (Log Scale)")

fig.update_xaxes(title_text=x_col)

fig.update_layout(
    title="Training Trends",
    template="plotly_white",
    height=600,
    width=950,
    hovermode="x unified",
    legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
)

fig.write_image(os.path.join(figure_dir, "line_graph.png"))
fig.write_html(os.path.join(figure_dir, "line_graph.html"))

"""
Box plot
"""

# bins = [0, 250, 500, 750, 1000]
# labels = ['1-250', '250-500', '500-750', '750-1000']

bins = [0, 200, 400, 600, 800, 1000]
labels = ["1-200", "200-400", "400-600", "600-800", "800-1000"]

df_filtered = df[(df[x_col] >= 1) & (df[x_col] <= 1000)].copy()
df_filtered["stage"] = pd.cut(df_filtered[x_col], bins=bins, labels=labels)

melted_df = pd.melt(
    df_filtered,
    id_vars=["stage"],
    value_vars=data_cols,
    var_name="Algorithm",
    value_name="raw_value",
)
melted_df["value"] = -melted_df["raw_value"]


fig = px.box(
    melted_df,
    x="stage",
    y="value",
    color="Algorithm",
    points="outliers",
    title="Algorithm Performance Comparison",
    labels={
        "stage": "Episode Range",
        "value": "System cost (Log Scale)",
        "Algorithm": "Algorithm",
    },
)

fig.update_yaxes(type="log", autorange="reversed")

fig.update_layout(
    template="plotly_white",
    boxmode="group",
    height=600,
    width=950,
    legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
)

fig.write_image(os.path.join(figure_dir, "box_plot.png"))
fig.write_html(os.path.join(figure_dir, "box_plot.html"))

"""
Violin plot
"""

last_100_df = df[(df[x_col] > 900) & (df[x_col] <= 1000)].copy()

melted_df = pd.melt(
    last_100_df, value_vars=data_cols, var_name="Algorithm", value_name="raw_value"
)
melted_df["value"] = -melted_df["raw_value"]

fig = px.violin(
    melted_df,
    x="Algorithm",
    y="value",
    color="Algorithm",
    box=True,
    points="all",
    hover_data=["value"],
    title="Final 100 Episodes Reward Distribution",
    labels={"Algorithm": "Algorithm", "value": "System cost (Log Scale)"},
)

fig.update_yaxes(type="log", autorange="reversed")

fig.update_layout(template="plotly_white", height=600, width=900, showlegend=False)

fig.write_image(os.path.join(figure_dir, "violin_plot.png"))
fig.write_html(os.path.join(figure_dir, "violin_plot.html"))
