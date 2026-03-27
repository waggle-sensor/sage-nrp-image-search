"""
Helper functions for plotting benchmarking graphs.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import re

# Cross-version leaderboard helpers
LEADERBOARD_METRIC_COLUMNS = {
    "MRR": "rerank_score_reciprocal_rank",
    "Success@25": "hit",
    "Precision@25": "precision",
    "NDCG@25": "rerank_score_NDCG",
    "Recall@25": "recall",
    "Diversity@25": "diversity",
}

PRIMARY_METRICS = ["MRR", "Success@25"]
PRIMARY_PLUS_DIVERSITY_METRICS = PRIMARY_METRICS + ["Diversity@25"]

DEFAULT_PRIMARY_WEIGHTS = {metric: 1 / len(PRIMARY_METRICS) for metric in PRIMARY_METRICS}
DEFAULT_PRIMARY_PLUS_DIVERSITY_WEIGHTS = {
    metric: 1 / len(PRIMARY_PLUS_DIVERSITY_METRICS) for metric in PRIMARY_PLUS_DIVERSITY_METRICS
}

def discover_benchmark_names(base_path: Path) -> list[str]:
    """
    Discover benchmark directories under benchmarking/benchmarks.
    Excludes helper/template/overall folders.
    """
    base_path = Path(base_path)
    excluded = {"template", "overall", "__pycache__", "helpers"}
    benchmarks = []
    for p in base_path.iterdir():
        if not p.is_dir():
            continue
        if p.name in excluded:
            continue
        if p.name.startswith("."):
            continue
        benchmarks.append(p.name)
    return sorted(benchmarks)

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

def plot_overall_hitrate(
    system_version: str,
    base_path: Path,
    benchmarks: list[str] | None = None,
    response_limit:int=25
):
    """
    Plot overall hit rate as a single bar chart.
    :param system_version: version of the system
    :param base_path: base path to the benchmarks directory
    :param benchmarks: list of benchmarks to plot, if None, all benchmarks will be plotted.
    :param response_limit: number of images returned by the benchmark results for each query. Default is 25.
    """
    k=response_limit
    if benchmarks is None:
        benchmarks = discover_benchmark_names(base_path)
    # Get the paths for the benchmarks
    bench_paths = []
    for benchmark in benchmarks:
        bench_paths.append((benchmark, base_path / f"{benchmark}/results/{system_version}/query_eval_metrics.csv"))

    # Get the hit rates for the benchmarks
    names, hit_rates = [], []
    for label, path in bench_paths:
        if not path.exists():
            print(f"Skip {label}: not found")
            continue
        df = pd.read_csv(path)
        if "hit" not in df.columns:
            print(f"Skip {label}: no 'hit' column")
            continue
        names.append(label)
        hit_rates.append(df["hit"].mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    colors = ["#2e86ab", "#a23b72", "#f18f01", "#c73e1d"]
    bars = ax.bar(x, hit_rates, color=colors[: len(names)], alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel(f"Hit Rate (Success@{k})", fontsize=12)
    ax.set_title(f"Overall Hit Rate by Benchmark ({system_version})", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars, fmt=lambda v: f"{v * 100:.1f}%", padding=8, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_overall_ndcg(
    system_version: str,
    base_path: Path,
    benchmarks: list[str] | None = None,
):
    """
    Plot overall NDCG (rerank_score_NDCG) as a single bar chart for each benchmark.
    :param system_version: version of the system (e.g. "v10")
    :param base_path: base path to the benchmarks directory
    :param benchmarks: list of benchmark names to plot, if None, all benchmarks will be plotted.
    """
    if benchmarks is None:
        benchmarks = discover_benchmark_names(base_path)
    bench_paths = [
        (b, base_path / f"{b}/results/{system_version}/query_eval_metrics.csv")
        for b in benchmarks
    ]
    names, ndcg_values = [], []
    for label, path in bench_paths:
        if not path.exists():
            print(f"Skip {label}: not found")
            continue
        df = pd.read_csv(path)
        if "rerank_score_NDCG" not in df.columns:
            print(f"Skip {label}: no 'rerank_score_NDCG' column")
            continue
        names.append(label)
        ndcg_values.append(df["rerank_score_NDCG"].mean())

    if not names:
        print("No benchmark data found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    colors = ["#2e86ab", "#a23b72", "#f18f01", "#c73e1d"]
    bars = ax.bar(x, ndcg_values, color=colors[: len(names)], alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("NDCG", fontsize=12)
    ax.set_title(f"Overall NDCG by Benchmark ({system_version})", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars, fmt=lambda v: f"{v:.2f}", padding=8, fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_overall_diversity(
    system_version: str,
    base_path: Path,
    benchmarks: list[str] | None = None,
):
    """
    Plot overall diversity as a single bar chart for each benchmark.
    :param system_version: version of the system (e.g. "v10")
    :param base_path: base path to the benchmarks directory
    :param benchmarks: list of benchmark names to plot, if None, all benchmarks will be plotted.
    """
    if benchmarks is None:
        benchmarks = discover_benchmark_names(base_path)
    # Get the paths for the benchmarks
    bench_paths = []
    for benchmark in benchmarks:
        bench_paths.append((benchmark, base_path / f"{benchmark}/results/{system_version}/query_eval_metrics.csv"))

    # Get the hit rates for the benchmarks
    names, diversity_values = [], []
    for label, path in bench_paths:
        if not path.exists():
            print(f"Skip {label}: not found")
            continue
        df = pd.read_csv(path)
        if "diversity" not in df.columns:
            print(f"Skip {label}: no 'diversity' column")
            continue
        names.append(label)
        diversity_values.append(df["diversity"].mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    colors = ["#2e86ab", "#a23b72", "#f18f01", "#c73e1d"]
    bars = ax.bar(x, diversity_values, color=colors[: len(names)], alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel(f"Diversity", fontsize=12)
    ax.set_title(f"Overall Diversity by Benchmark ({system_version})", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars, fmt=lambda v: f"{v:.2f}", padding=8, fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_overall_mrr(
    system_version: str,
    base_path: Path,
    benchmarks: list[str] | None = None,
):
    """
    Plot overall MRR as a single bar chart for each benchmark.
    :param system_version: version of the system (e.g. "v10")
    :param base_path: base path to the benchmarks directory
    :param benchmarks: list of benchmark names to plot, if None, all benchmarks will be plotted.
    """
    if benchmarks is None:
        benchmarks = discover_benchmark_names(base_path)
    bench_paths = [
        (b, base_path / f"{b}/results/{system_version}/query_eval_metrics.csv")
        for b in benchmarks
    ]
    names, mrr_values = [], []
    for label, path in bench_paths:
        if not path.exists():
            print(f"Skip {label}: not found")
            continue
        df = pd.read_csv(path)
        if "rerank_score_reciprocal_rank" not in df.columns:
            print(f"Skip {label}: no 'rerank_score_reciprocal_rank' column")
            continue
        names.append(label)
        mrr_values.append(df["rerank_score_reciprocal_rank"].mean())

    if not names:
        print("No benchmark data found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    colors = ["#2e86ab", "#a23b72", "#f18f01", "#c73e1d"]
    bars = ax.bar(x, mrr_values, color=colors[: len(names)], alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("MRR", fontsize=12)
    ax.set_title(f"Overall MRR by Benchmark ({system_version})", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars, fmt=lambda v: f"{v:.2f}", padding=8, fontsize=11)
    plt.tight_layout()
    plt.show()

def _version_key(version: str):
    match = re.fullmatch(r"v(\d+)", str(version))
    return int(match.group(1)) if match else float("inf")


def normalize_weights(weight_map: dict[str, float], expected_keys: list[str]) -> dict[str, float]:
    """
    Normalize a set of weights so they sum to 1. Missing keys default to 0.
    """
    values = {k: float(weight_map.get(k, 0.0)) for k in expected_keys}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Weight sum must be > 0")
    return {k: v / total for k, v in values.items()}


def discover_benchmark_versions(
    base_path: Path,
    benchmarks: list[str] | None = None,
    min_version: int = 10,
) -> dict[str, list[str]]:
    """
    Discover benchmark versions with query_eval_metrics.csv from v{min_version}+.
    Returns mapping benchmark -> sorted versions.
    """
    base_path = Path(base_path)
    if benchmarks is None:
        benchmarks = [b for b in discover_benchmark_names(base_path) if (base_path / b / "results").exists()]

    out = {}
    for benchmark in benchmarks:
        result_dir = base_path / benchmark / "results"
        if not result_dir.exists():
            out[benchmark] = []
            continue
        versions = []
        for version_dir in result_dir.iterdir():
            if not version_dir.is_dir():
                continue
            match = re.fullmatch(r"v(\d+)", version_dir.name)
            if not match:
                continue
            if int(match.group(1)) < min_version:
                continue
            if (version_dir / "query_eval_metrics.csv").exists():
                versions.append(version_dir.name)
        out[benchmark] = sorted(versions, key=_version_key)
    return out


def load_cross_version_metrics(
    base_path: Path,
    benchmarks: list[str] | None = None,
    min_version: int = 10,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load and aggregate query metrics for benchmark/version combinations.
    Aggregation is the mean across queries for each metric.
    """
    discovered = discover_benchmark_versions(base_path, benchmarks=benchmarks, min_version=min_version)
    rows = []

    for benchmark, versions in discovered.items():
        for version in versions:
            csv_path = Path(base_path) / benchmark / "results" / version / "query_eval_metrics.csv"
            if not csv_path.exists():
                if verbose:
                    print(f"Skip {benchmark} {version}: query_eval_metrics.csv not found")
                continue
            df = pd.read_csv(csv_path)
            row = {
                "benchmark": benchmark,
                "system_version": version,
                "query_count": len(df),
            }
            missing_cols = []
            for display_metric, col in LEADERBOARD_METRIC_COLUMNS.items():
                if col in df.columns:
                    row[display_metric] = float(df[col].mean())
                else:
                    row[display_metric] = np.nan
                    missing_cols.append(col)
            if missing_cols and verbose:
                print(f"Warning {benchmark} {version}: missing columns {missing_cols}")
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["benchmark", "system_version", "query_count"] + list(LEADERBOARD_METRIC_COLUMNS.keys()))

    out = pd.DataFrame(rows)
    return out.sort_values(["benchmark", "system_version"], key=lambda s: s.map(_version_key) if s.name == "system_version" else s).reset_index(drop=True)


def _get_metric_set_and_default_weights(mode: str):
    mode = mode.lower()
    if mode == "primary":
        return PRIMARY_METRICS, DEFAULT_PRIMARY_WEIGHTS
    if mode in {"primary_plus_diversity", "primary+diversity"}:
        return PRIMARY_PLUS_DIVERSITY_METRICS, DEFAULT_PRIMARY_PLUS_DIVERSITY_WEIGHTS
    raise ValueError("mode must be 'primary' or 'primary_plus_diversity'")


def add_composite_score(
    df: pd.DataFrame,
    mode: str = "primary",
    metric_weights: dict[str, float] | None = None,
    score_column: str = "composite_score",
) -> pd.DataFrame:
    """
    Add a weighted composite score column to a leaderboard dataframe.
    """
    metrics, defaults = _get_metric_set_and_default_weights(mode)
    metric_weights = defaults if metric_weights is None else normalize_weights(metric_weights, metrics)
    out = df.copy()
    out[score_column] = 0.0
    for metric in metrics:
        out[score_column] = out[score_column] + (out[metric].fillna(0.0) * metric_weights[metric])
    return out


def build_benchmark_version_leaderboard(
    base_path: Path,
    benchmarks: list[str] | None = None,
    min_version: int = 10,
    mode: str = "primary",
    metric_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build a per-benchmark leaderboard ranking system versions.
    """
    df = load_cross_version_metrics(base_path, benchmarks=benchmarks, min_version=min_version)
    if df.empty:
        return df
    out = add_composite_score(df, mode=mode, metric_weights=metric_weights)
    out = out.sort_values(["benchmark", "composite_score", "system_version"], ascending=[True, False, True]).reset_index(drop=True)
    out["rank_within_benchmark"] = out.groupby("benchmark")["composite_score"].rank(method="dense", ascending=False).astype(int)
    return out


def build_overall_version_leaderboard(
    base_path: Path,
    benchmarks: list[str] | None = None,
    min_version: int = 10,
    mode: str = "primary",
    metric_weights: dict[str, float] | None = None,
    benchmark_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build a version-level leaderboard aggregated across benchmarks.
    benchmark_weights default to equal benchmark weighting.
    """
    per_benchmark = build_benchmark_version_leaderboard(
        base_path=base_path,
        benchmarks=benchmarks,
        min_version=min_version,
        mode=mode,
        metric_weights=metric_weights,
    )
    if per_benchmark.empty:
        return per_benchmark

    benchmark_names = sorted(per_benchmark["benchmark"].unique())
    if benchmark_weights is None:
        benchmark_weights = {b: 1.0 for b in benchmark_names}
    normalized_benchmark_weights = normalize_weights(benchmark_weights, benchmark_names)

    weighted = per_benchmark.copy()
    weighted["benchmark_weight"] = weighted["benchmark"].map(normalized_benchmark_weights).fillna(0.0)

    # Re-normalize per version to avoid penalizing versions missing some benchmark runs.
    weight_sum_by_version = weighted.groupby("system_version")["benchmark_weight"].transform("sum")
    weighted["effective_weight"] = np.where(weight_sum_by_version > 0, weighted["benchmark_weight"] / weight_sum_by_version, 0.0)

    metrics, _ = _get_metric_set_and_default_weights(mode)
    agg_rows = []
    for version, part in weighted.groupby("system_version"):
        row = {"system_version": version}
        row["benchmark_count"] = int(part["benchmark"].nunique())
        for metric in metrics:
            row[metric] = float((part[metric].fillna(0.0) * part["effective_weight"]).sum())
        row["composite_score"] = float((part["composite_score"] * part["effective_weight"]).sum())
        agg_rows.append(row)

    out = pd.DataFrame(agg_rows).sort_values(["composite_score", "system_version"], ascending=[False, True]).reset_index(drop=True)
    out["rank_overall"] = np.arange(1, len(out) + 1)
    return out


def plot_leaderboard_scores(
    leaderboard_df: pd.DataFrame,
    label_column: str,
    score_column: str = "composite_score",
    title: str = "Leaderboard",
    figsize=(10, 5),
):
    """
    Plot leaderboard scores as a descending bar chart.
    """
    if leaderboard_df.empty:
        print("No leaderboard data found.")
        return
    df = leaderboard_df.sort_values(score_column, ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(df[label_column].astype(str), df[score_column], alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Composite Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars, fmt=lambda v: f"{v:.3f}", padding=6, fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def render_single_benchmark_leaderboard(
    base_path: Path,
    benchmark: str,
    min_version: int = 10,
    mode: str = "primary",
    metric_weights: dict[str, float] | None = None,
    display_fn=None,
) -> pd.DataFrame:
    """
    Build, render table, and plot leaderboard for one benchmark.
    Returns the benchmark-only leaderboard dataframe.
    """
    leaderboard_df = build_benchmark_version_leaderboard(
        base_path=base_path,
        benchmarks=[benchmark],
        min_version=min_version,
        mode=mode,
        metric_weights=metric_weights,
    )

    part = leaderboard_df[leaderboard_df["benchmark"] == benchmark].copy()
    if part.empty:
        print(f"Skip {benchmark}: no v{min_version}+ results")
        return part

    mode_label = "Primary + Diversity" if mode in {"primary_plus_diversity", "primary+diversity"} else "Primary"
    print(f"\n{benchmark} ({mode_label}):")

    if display_fn is not None:
        display_fn(part)
    else:
        print(part.to_string(index=False))

    part["label"] = part["benchmark"] + "-" + part["system_version"]
    plot_leaderboard_scores(
        part,
        label_column="label",
        title=f"{benchmark} {mode_label} Leaderboard",
    )
    return part