"""Query-wise ranking metrics with explicit gains and stable, label-blind ties."""

import numpy as np
import polars as pl

ESCI_GAINS = {"I": 0.0, "C": 0.01, "S": 0.1, "E": 1.0}
ESCI_LABELS = {"I": 0, "C": 1, "S": 2, "E": 3}


def ndcg(gains, scores, k=10, tie_order=None):
    gains, scores = np.asarray(gains, float), np.asarray(scores, float)
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer")
    if gains.ndim != 1 or scores.ndim != 1 or len(gains) != len(scores):
        raise ValueError("Gains/scores must be equal-length vectors")
    if not np.isfinite(scores).all() or not np.isfinite(gains).all() or (gains < 0).any():
        raise ValueError("Scores must be finite; gains must be finite and nonnegative")
    if not len(gains):
        return 0.0
    tie = np.arange(len(gains)) if tie_order is None else np.asarray(tie_order)
    if tie.shape != gains.shape:
        raise ValueError("Tie order must have the same shape as gains")
    order = np.lexsort((tie, -scores))[:k]
    discount = 1 / np.log2(np.arange(2, len(order) + 2))
    ideal = np.sort(gains)[::-1][:len(order)] @ discount
    return float(gains[order] @ discount / ideal) if ideal > 0 else 0.0


def evaluate(frame, score_columns, gain_map=ESCI_GAINS):
    rows = []
    for part in frame.partition_by("query_id", maintain_order=True):
        gains = np.array([gain_map[v] for v in part["label"]])
        tie = part["tie_order"].to_numpy()
        exact = np.array([v in {"E", "Exact"} for v in part["label"]])
        row = {"query_id": part["query_id"][0], "family": part["family"][0],
               "candidates": len(gains), "has_exact": bool(exact.any()), "zero_idcg": not bool(gains.sum())}
        for name in score_columns:
            scores = part[name].to_numpy()
            order = np.lexsort((tie, -scores))
            row[name + "_ndcg10"] = ndcg(gains, scores, 10, tie)
            row[name + "_ndcg_all"] = ndcg(gains, scores, len(gains), tie)
            row[name + "_exact_recall10"] = float(exact[order[:10]].sum() / exact.sum()) if exact.any() else None
            positions = np.flatnonzero(exact[order])
            row[name + "_mrr"] = 1.0 / (int(positions[0]) + 1) if len(positions) else 0.0
        rows.append(row)
    per_query = pl.DataFrame(rows)
    means = {name: {"ndcg10": float(per_query[name + "_ndcg10"].mean()),
                    "ndcg_all": float(per_query[name + "_ndcg_all"].mean()),
                    "exact_recall10": per_query[name + "_exact_recall10"].mean(),
                    "mrr": float(per_query[name + "_mrr"].mean())} for name in score_columns}
    return means, per_query


def paired_bootstrap(per_query, candidate="attribute", baseline="baseline", repeats=5000, seed=20261026):
    if not isinstance(repeats, (int, np.integer)) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    grouped = per_query.group_by("family").agg(
        (pl.col(candidate + "_ndcg10") - pl.col(baseline + "_ndcg10")).sum().alias("delta"), pl.len().alias("n")).sort("family")
    values, sizes = grouped["delta"].to_numpy(), grouped["n"].to_numpy()
    if not len(values):
        raise ValueError("No query families")
    rng, draws = np.random.default_rng(seed), []
    for _ in range(repeats):
        idx = rng.integers(0, len(values), len(values))
        draws.append(values[idx].sum() / sizes[idx].sum())
    return {"difference": float(values.sum() / sizes.sum()), "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)), "families": len(values), "queries": per_query.height,
            "bootstrap_repeats": repeats, "unit": "normalized query family", "metric": "mean query NDCG@10"}
