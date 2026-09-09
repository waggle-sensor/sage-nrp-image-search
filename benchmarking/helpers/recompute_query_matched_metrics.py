"""Recompute archived NDCG and reciprocal rank with query-matched relevance.

Hit, precision, and recall in the archived CSVs already require the returned
record's originating query_id to match the searched query. NDCG and reciprocal
rank originally used the stored relevance label without that check. This script
rewrites only those ranking columns, in place, from image_search_results.csv.

A result is credited only when its originating query matches the submitted
query and its label is positive. Equal scores keep CSV order for reciprocal
rank. NDCG uses sklearn.metrics.ndcg_score on the returned list, matching the
evaluator, after zeroing labels from other queries.

Usage:
    python benchmarking/helpers/recompute_query_matched_metrics.py
    python benchmarking/helpers/recompute_query_matched_metrics.py --root /path/to/sage-nrp-image-search
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score

RANK_SUFFIXES = ("_NDCG", "_reciprocal_rank")
LEGACY_NDCG = {
    "NDCG": "rerank_score",
    "clip_NDCG": "clip_score",
}


def norm_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if math.isfinite(number) and number == int(number):
        return str(int(number))
    return text


def label_column(fields: list[str]) -> str:
    if "relevance_label" in fields:
        return "relevance_label"
    if "relevant" in fields:
        return "relevant"
    raise KeyError(f"No relevance column in {fields}")


def ranking_columns(fields: list[str]) -> list[str]:
    columns = []
    for name in fields:
        if name in LEGACY_NDCG or name.endswith(RANK_SUFFIXES):
            columns.append(name)
    return columns


def sort_column_for(metric_name: str) -> str:
    if metric_name in LEGACY_NDCG:
        return LEGACY_NDCG[metric_name]
    for suffix in RANK_SUFFIXES:
        if metric_name.endswith(suffix):
            return metric_name[: -len(suffix)]
    raise KeyError(metric_name)


def to_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def matched_relevance(row: dict, query_id: str, label: str) -> float:
    if norm_id(row.get("query_id")) != query_id:
        return 0.0
    return to_float(row.get(label), 0.0)


def compute_ndcg(relevances: list[float], scores: list[float]) -> float:
    if len(relevances) < 2:
        return 0.0
    y_true = np.asarray(relevances, dtype=float).reshape(1, -1)
    if float(y_true.sum()) == 0.0:
        return 0.0
    y_score = np.asarray(scores, dtype=float).reshape(1, -1)
    return float(ndcg_score(y_true, y_score))


def compute_reciprocal_rank(relevances: list[float], scores: list[float]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    for rank, index in enumerate(order, start=1):
        if relevances[index] > 0:
            return 1.0 / rank
    return 0.0


def format_metric(value: float) -> str:
    if value == 0:
        return "0.0"
    return repr(float(value))


def recompute_run(metrics_path: Path) -> dict:
    results_path = metrics_path.with_name("image_search_results.csv")
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        metrics = list(reader)
    with results_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    rank_cols = ranking_columns(fieldnames)
    if not rank_cols:
        raise ValueError(f"No ranking columns in {metrics_path}")
    if not rows:
        raise ValueError(f"Empty results in {results_path}")

    label = label_column(list(rows[0].keys()))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[norm_id(row.get("queried_on_query_id"))].append(row)

    sort_by = {column: sort_column_for(column) for column in rank_cols}
    missing = [column for column, name in sort_by.items() if name not in rows[0]]
    if missing:
        raise KeyError(f"{metrics_path}: missing score columns for {missing}")

    old_means = {column: [] for column in rank_cols}
    new_means = {column: [] for column in rank_cols}
    raw_ndcg_mismatch = 0
    hit_mismatch = 0
    relevant_mismatch = 0
    changed_cells = 0

    for metric in metrics:
        query_id = norm_id(metric.get("query_id"))
        retrieved = grouped.get(query_id, [])
        matched_flags = [matched_relevance(row, query_id, label) for row in retrieved]
        raw_flags = [to_float(row.get(label), 0.0) for row in retrieved]
        matched_count = sum(flag for flag in matched_flags if flag > 0)

        if "relevant_images" in metric:
            archived = to_float(metric.get("relevant_images"), 0.0)
            if not math.isclose(matched_count, archived, abs_tol=1e-8):
                relevant_mismatch += 1
        if "hit" in metric:
            archived_hit = int(to_float(metric.get("hit"), 0.0))
            if int(matched_count > 0) != archived_hit:
                hit_mismatch += 1

        for column in rank_cols:
            scores = [to_float(row.get(sort_by[column]), 0.0) for row in retrieved]
            if column.endswith("_reciprocal_rank"):
                raw_value = compute_reciprocal_rank(raw_flags, scores)
                new_value = compute_reciprocal_rank(matched_flags, scores)
            else:
                raw_value = compute_ndcg(raw_flags, scores)
                new_value = compute_ndcg(matched_flags, scores)
                archived = to_float(metric.get(column), 0.0)
                if abs(raw_value - archived) > 1e-8:
                    raw_ndcg_mismatch += 1
            archived = to_float(metric.get(column), 0.0)
            old_means[column].append(archived)
            new_means[column].append(new_value)
            new_text = format_metric(new_value)
            if metric.get(column) != new_text:
                changed_cells += 1
            metric[column] = new_text

    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)

    summary = {
        "path": str(metrics_path),
        "queries": len(metrics),
        "changed_cells": changed_cells,
        "hit_mismatch": hit_mismatch,
        "relevant_mismatch": relevant_mismatch,
        "raw_ndcg_mismatch": raw_ndcg_mismatch,
    }
    for column in rank_cols:
        old = sum(old_means[column]) / len(old_means[column])
        new = sum(new_means[column]) / len(new_means[column])
        summary[f"old_{column}"] = old
        summary[f"new_{column}"] = new
        summary[f"delta_{column}"] = new - old
    return summary


def find_metrics(root: Path) -> list[Path]:
    return sorted(root.glob("benchmarks/*/results/**/query_eval_metrics.csv"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="benchmarking directory (default: this helpers/ parent)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    paths = find_metrics(root)
    if not paths:
        print(f"No query_eval_metrics.csv files under {root}", file=sys.stderr)
        return 1

    summaries = []
    failures = []
    for path in paths:
        try:
            summaries.append(recompute_run(path))
            print(f"updated {path.relative_to(root)}")
        except Exception as exc:
            failures.append((path, exc))
            print(f"FAILED {path.relative_to(root)}: {exc}", file=sys.stderr)

    report_path = root / "query_matched_recompute_summary.csv"
    if summaries:
        columns = [
            "benchmark",
            "version",
            "queries",
            "changed_cells",
            "hit_mismatch",
            "relevant_mismatch",
            "raw_ndcg_mismatch",
            "old_mrr",
            "new_mrr",
            "delta_mrr",
            "old_ndcg",
            "new_ndcg",
            "delta_ndcg",
        ]
        with report_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for summary in summaries:
                path = Path(summary["path"])
                try:
                    relative = path.relative_to(root / "benchmarks")
                    benchmark = relative.parts[0]
                    version = str(Path(*relative.parts[2:-1]))
                except ValueError:
                    benchmark = ""
                    version = str(path)
                mrr_old = summary.get("old_rerank_score_reciprocal_rank")
                mrr_new = summary.get("new_rerank_score_reciprocal_rank")
                ndcg_old = summary.get("old_rerank_score_NDCG", summary.get("old_NDCG"))
                ndcg_new = summary.get("new_rerank_score_NDCG", summary.get("new_NDCG"))
                writer.writerow(
                    {
                        "benchmark": benchmark,
                        "version": version,
                        "queries": summary["queries"],
                        "changed_cells": summary["changed_cells"],
                        "hit_mismatch": summary["hit_mismatch"],
                        "relevant_mismatch": summary["relevant_mismatch"],
                        "raw_ndcg_mismatch": summary["raw_ndcg_mismatch"],
                        "old_mrr": "" if mrr_old is None else f"{mrr_old:.6f}",
                        "new_mrr": "" if mrr_new is None else f"{mrr_new:.6f}",
                        "delta_mrr": ""
                        if mrr_old is None
                        else f"{(mrr_new - mrr_old):.6f}",
                        "old_ndcg": "" if ndcg_old is None else f"{ndcg_old:.6f}",
                        "new_ndcg": "" if ndcg_new is None else f"{ndcg_new:.6f}",
                        "delta_ndcg": ""
                        if ndcg_old is None
                        else f"{(ndcg_new - ndcg_old):.6f}",
                    }
                )

    print(f"\nupdated {len(summaries)} / {len(paths)} runs")
    print(f"summary: {report_path}")
    if failures:
        print(f"{len(failures)} failures", file=sys.stderr)
        return 1
    hit_issues = [s for s in summaries if s["hit_mismatch"] or s["relevant_mismatch"]]
    if hit_issues:
        print(
            f"WARNING: {len(hit_issues)} runs disagreed with archived hit/relevant counts",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
