"""
Helper functions for plotting benchmarking graphs.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_grouped_bar_by_columns(
    df, 
    x_column, 
    color_column, 
    metric, 
    color_map=None,
    ylabel=None,
    xlabel=None,
    title=None,
    ylim=(0, 1.1),
    figsize=(8, 4),
    bar_width=0.35,
    rotate_xticks=45,
    bar_label_fmt="%.2f"
):
    """
    Plot grouped bar chart with two columns with y axis as a metric.
    :param df: DataFrame with grouped results (output of .groupby(...).agg().reset_index())
    :param x_column: column for the x-axis (categorical)
    :param color_column: column used for color/legend (categorical, 2 values)
    :param metric: name of metric column to plot
    :param color_map: dict mapping color_column values to colors
    :param ylabel: label for y axis
    :param xlabel: x axis label
    :param title: plot title
    :param ylim: y-axis limits tuple
    :param figsize: figure size
    :param bar_width: bar width
    :param rotate_xticks: rotation for x-tick labels
    :param bar_label_fmt: bar label format (e.g., "%.2f")
    """

    x_categories = df[x_column].unique()
    color_categories = df[color_column].unique()
    x = np.arange(len(x_categories))
    if color_map is None:
        # Default color map for two colors
        default_colors = ["green", "red"]
        color_map = {val: default_colors[i % 2] for i, val in enumerate(sorted(color_categories))}
    fig, ax = plt.subplots(figsize=figsize)

    all_bar_handles = []
    # Compute bar positions for each color value
    for i, color_val in enumerate(sorted(color_categories)):
        heights = []
        for cat in x_categories:
            sel = df[(df[x_column] == cat) & (df[color_column] == color_val)][metric]
            heights.append(sel.values[0] if len(sel) > 0 else 0)
        offsets = x - bar_width / 2 + i * bar_width if len(color_categories)==2 else x + i * bar_width
        bars = ax.bar(offsets, heights, bar_width, 
                      color=color_map.get(color_val, None), 
                      alpha=0.7, 
                      label=str(color_val))
        ax.bar_label(bars, fmt=bar_label_fmt, padding=5)
        all_bar_handles.append(bars)

    ax.set_ylabel(ylabel if ylabel else metric.capitalize())
    ax.set_xlabel(xlabel if xlabel else x_column.capitalize())
    ax.set_title(title if title else f"{metric.capitalize()} by {x_column} & {color_column}")
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(x_categories, rotation=rotate_xticks, ha="right")
    ax.legend(title=color_column.replace("_", " ").capitalize(), loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def plot_single_bar_metric(
    df,
    x_column,
    metric,
    title=None,
    ylabel=None,
    xlabel=None,
    ylim=(0, 1.1),
    figsize=(8, 4),
    bar_label_fmt="%.2f",
    color="blue",
    rotate_xticks=45,
):
    """Plot a single bar chart: one bar per row, one metric."""
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(df[x_column], df[metric], color=color, alpha=0.7)
    ax.set_ylabel(ylabel if ylabel else metric)
    ax.set_xlabel(xlabel if xlabel else x_column.replace("_", " ").title())
    ax.set_title(title if title else f"{metric} by {x_column.replace('_', ' ').title()}")
    ax.set_ylim(*ylim)
    ax.bar_label(bars, fmt=bar_label_fmt, padding=5)
    plt.xticks(rotation=rotate_xticks)
    plt.show()


def plot_rank_comparison(
    df,
    x_column,
    title=None,
    xlabel=None,
    ylim=(0, 1.1),
    figsize=(8, 4),
    bar_width=0.35,
    rotate_xticks=45,
    dataset_clip_model="CLIP DFN5B-CLIP-ViT-H-14-378",
    metric="NDCG"
):
    """
    Plot rank metrics as grouped bars (one label per x_column). Available rank metrics are NDCG and MRR.

    :param df: DataFrame with rank metrics, and the categorical column (e.g. category_metrics)
    :param x_column: column for the x-axis (one bar per value, e.g. 'category')
    :param title: plot title
    :param xlabel: x-axis label
    :param ylim: y-axis limits
    :param figsize: figure size
    :param bar_width: bar width
    :param rotate_xticks: rotation for x-tick labels
    :param dataset_clip_model: legend label for rank metric series
    :param metric: rank metric to plot (e.g. "NDCG" or "MRR")
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df))
    metric_up = metric.upper()
    metric_low = metric.lower()

    if metric_low not in ["ndcg", "mrr"]:
        raise ValueError(f"Metric {metric} not supported")

    bars1 = ax.bar(x - bar_width / 2, df[metric_up], width=bar_width, label="Hybrid Search", color="blue", alpha=0.7)
    bars2 = ax.bar(x + bar_width / 2, df[f"clip_{metric_up}"], width=bar_width, label=dataset_clip_model, color="green", alpha=0.7)
    ax.set_ylabel(metric_up)
    ax.set_xlabel(xlabel if xlabel else x_column.replace("_", " ").title())
    ax.set_title(title if title else f"{metric_up} by {x_column.replace('_', ' ').title()}")
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_column], rotation=rotate_xticks)
    ax.legend()
    plt.show()


def plot_multi_rank_comparison(
    df,
    x_column,
    group_column,
    title=None,
    xlabel=None,
    ylim=(0, 1.1),
    figsize=(8, 4),
    bar_width=0.35,
    rotate_xticks=45,
    bar_label_fmt="%.2f",
    dataset_clip_model="CLIP DFN5B-CLIP-ViT-H-14-378",
    divider_linestyle=":",
    group_label_y=1.0,
    group_label_x_offset=0.3,
    metric="NDCG"
):
    """
    Plot rank metrics as grouped bars with a second column used for visual grouping. Available rank metrics are NDCG and MRR.

    Draws dotted vertical lines between groups and places group labels above each group
    (e.g. category on x-axis, supercategory as groups with dividers and labels).

    :param df: DataFrame with rank metrics, and the two categorical columns (e.g. category_metrics)
    :param x_column: column for the x-axis (one bar pair per value, e.g. 'category')
    :param group_column: column used for grouping (dividers and labels, e.g. 'supercategory')
    :param title: plot title
    :param xlabel: x-axis label
    :param ylim: y-axis limits
    :param figsize: figure size
    :param bar_width: bar width
    :param rotate_xticks: rotation for x-tick labels
    :param bar_label_fmt: format for bar labels (e.g. "%.2f")
    :param dataset_clip_model: legend label for rank metric series
    :param divider_linestyle: linestyle for vertical dividers (e.g. ':')
    :param group_label_y: y position for group labels above the bars
    :param group_label_x_offset: horizontal offset for group label position (e.g. 0.3)
    :param metric: rank metric to plot (e.g. "NDCG" or "MRR")
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df))
    metric_up = metric.upper()
    metric_low = metric.lower()
    if metric_low not in ["ndcg", "mrr"]:
        raise ValueError(f"Metric {metric} not supported")
    bars1 = ax.bar(
        x - bar_width / 2, df[metric_up], width=bar_width,
        label="Hybrid Search", color="blue", alpha=0.7
    )
    bars2 = ax.bar(
        x + bar_width / 2, df[f"clip_{metric_up}"], width=bar_width,
        label=dataset_clip_model, color="green", alpha=0.7
    )
    ax.bar_label(bars1, fmt=bar_label_fmt, padding=5)
    ax.bar_label(bars2, fmt=bar_label_fmt, padding=5)

    ax.set_ylabel(metric_up)
    ax.set_xlabel(xlabel if xlabel else x_column.replace("_", " ").title())
    ax.set_title(title if title else f"{metric_up} and clip_{metric_up} by {x_column.replace('_', ' ').title()}")
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_column], rotation=rotate_xticks, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Dotted vertical lines between groups when group_column value changes
    group_vals = df[group_column].values
    for i in range(1, len(group_vals)):
        if group_vals[i] != group_vals[i - 1]:
            ax.axvline(x=i - 0.5, color="black", linestyle=divider_linestyle, linewidth=2)

    # Group labels above each contiguous block of the same group
    for group_val in df[group_column].unique():
        indices = np.where(df[group_column] == group_val)[0]
        center_x = (indices[0] + indices[-1]) / 2 + group_label_x_offset
        ax.text(
            center_x, group_label_y, group_val,
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    plt.show()