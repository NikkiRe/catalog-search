"""Refit the three sealed final rankers and compare every saved test score.

This audit uses the verified feature caches, the original prepared labels and
query groups, and parameters selected before test evaluation. It does not tune,
select, refit the encoder, or overwrite the published model artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from catalog_search.features import ALL_NAMES


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def groups(frame: pl.DataFrame) -> np.ndarray:
    query = frame["query_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(query[1:] != query[:-1]) + 1]
    require(len(starts) == frame["query_id"].n_unique(), "Non-contiguous query groups")
    return np.diff(np.r_[starts, len(query)]).astype(np.int32)


def replay(data: Path, artifacts: Path, reports: Path, output: Path) -> dict:
    started = time.perf_counter()
    require(not output.exists(), "Replay output already exists; choose a new directory")
    report_path = reports / "replay_training.json"
    require(not report_path.exists(), "Replay report already exists; preserve it or use another reports directory")
    protocol = json.loads((artifacts / "protocol.json").read_text())
    selection = json.loads((artifacts / "selection.json").read_text())
    summary = json.loads((reports / "summary.json").read_text())
    require(protocol["status"] == "frozen_before_training_and_test_quality", "Protocol is not frozen")
    require(selection["status"] == "selected_before_any_test_quality", "Selection is not sealed")
    require(digest(artifacts / "protocol.json") == selection["protocol_sha256"] == summary["provenance"]["protocol_sha256"], "Protocol checksum mismatch")
    require(digest(artifacts / "selection.json") == summary["provenance"]["selection_sha256"], "Selection checksum mismatch")
    require(canonical(protocol["config"]) == protocol["config_sha256"] == selection["config_sha256"], "Configuration checksum mismatch")
    require(protocol["feature_names"] == list(ALL_NAMES), "Feature order changed")
    project = Path(__file__).resolve().parents[1]
    for name, expected in protocol["source_sha256"].items():
        require(digest(project / name) == expected, f"Frozen source changed: {name}")
    prepared = {}
    for name, expected in protocol["esci_prepared_sha256"].items():
        actual = digest(data / name)
        require(actual == expected, f"Prepared data checksum mismatch: {name}")
        prepared[name] = actual
    matrices, frames, cache_hashes = {}, {}, {}
    for part in ("train", "dev", "test"):
        path = artifacts / "cache" / f"{part}.npy"
        actual = digest(path)
        require(actual == summary["feature_compute"][part]["sha256"], f"Feature cache changed: {part}")
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        frame = pl.read_parquet(data / f"{part}.parquet").sort("query_id", "tie_order")
        require(matrix.shape == (frame.height, len(ALL_NAMES)), f"Wrong {part} matrix shape")
        require(np.isfinite(matrix).all(), f"Non-finite {part} matrix")
        matrices[part], frames[part], cache_hashes[part] = matrix, frame, actual
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        require(not (set(frames[left]["family"]) & set(frames[right]["family"])), "Family leakage")
        require(not (set(frames[left]["query_id"]) & set(frames[right]["query_id"])), "Query leakage")
    fit = pl.concat([frames["train"], frames["dev"]])
    fit_groups = groups(fit)
    labels = {"I": 0, "C": 1, "S": 2, "E": 3}
    y = np.asarray([labels[value] for value in fit["label"]], dtype=np.int32)
    x = np.concatenate([matrices["train"], matrices["dev"]], axis=0)
    saved = pl.read_parquet(reports / "test_scores.parquet")
    key_columns = ["query_id", "product_id", "family", "label", "tie_order"]
    require(saved.select(key_columns).equals(frames["test"].select(key_columns)), "Saved scores and test cache row order differ")
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, choice in selection["models"].items():
        columns = choice["columns"]
        require([ALL_NAMES[index] for index in columns] == choice["feature_names"] == protocol["variants"][name], f"Wrong feature selection: {name}")
        winner = min(choice["trials"], key=lambda row: (-row["dev_ndcg10"], row["num_leaves"], row["trees"]))
        require(all(choice[key] == winner[key] for key in ("num_leaves", "trees", "dev_ndcg10")), f"Selection differs from sealed dev rule: {name}")
        params = dict(protocol["model_params"][str(choice["num_leaves"])])
        params["n_estimators"] = choice["trees"]
        require(params["objective"] == "lambdarank" and params["label_gain"] == [0., .01, .1, 1.], "Unexpected objective or gain map")
        original_path = artifacts / f"{name}.txt"
        require(digest(original_path) == summary["models"][name]["sha256"], f"Original model checksum mismatch: {name}")
        print(f"Replay final fit {name}: {fit.height:,} rows, {len(columns)} features, {choice['trees']} trees", flush=True)
        fit_started = time.perf_counter()
        model = lgb.LGBMRanker(**params)
        model.fit(np.ascontiguousarray(x[:, columns]), y, group=fit_groups)
        fit_seconds = time.perf_counter() - fit_started
        replay_path = output / f"{name}.txt"
        model.booster_.save_model(str(replay_path))
        booster = lgb.Booster(model_file=str(replay_path))
        prediction = booster.predict(np.ascontiguousarray(matrices["test"][:, columns]), num_threads=params["n_jobs"])
        error = float(np.max(np.abs(prediction - saved[name].to_numpy())))
        require(booster.num_feature() == len(columns), f"Wrong replay feature count: {name}")
        require(booster.num_trees() == choice["trees"], f"Wrong replay tree count: {name}")
        require(np.isfinite(prediction).all() and error <= 1e-12, f"Replay prediction mismatch {name}: {error}")
        results[name] = {"parameters": params, "features": booster.num_feature(), "trees": booster.num_trees(),
                         "test_rows": len(prediction), "max_absolute_prediction_error": error,
                         "original_model_sha256": digest(original_path), "replay_model_sha256": digest(replay_path),
                         "model_bytes_identical": digest(original_path) == digest(replay_path), "fit_seconds": fit_seconds}
        print(f"  all {len(prediction):,} predictions match; max error={error:.3g}", flush=True)
        del model, booster, prediction
        gc.collect()
    for name, expected in protocol["source_sha256"].items():
        require(digest(project / name) == expected, f"Frozen source changed during replay: {name}")
    result = {"status": "passed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Independent final train+dev refits from verified caches and sealed development selections; no tuning or new quality claim",
              "feature_extraction": "Verified original caches reused; this replay does not rerun the encoder fit or full feature extraction",
              "tolerance": 1e-12, "models": results,
              "inputs": {"protocol_sha256": digest(artifacts / "protocol.json"), "selection_sha256": digest(artifacts / "selection.json"),
                         "summary_sha256": digest(reports / "summary.json"), "test_scores_sha256": digest(reports / "test_scores.parquet"),
                         "prepared_sha256": prepared, "feature_cache_sha256": cache_hashes, "source_sha256": protocol["source_sha256"]},
              "training": {"rows": fit.height, "queries": len(fit_groups), "group_size_sum": int(fit_groups.sum()),
                           "label_counts": {label: int(np.count_nonzero(y == value)) for label, value in labels.items()}},
              "software": {name: importlib.metadata.version(name) for name in ("numpy", "polars", "lightgbm")},
              "hardware": {"machine": platform.machine(), "threads": protocol["config"]["threads"]},
              "runtime_seconds": time.perf_counter() - started}
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.data, args.artifacts, args.reports, args.output)
    print(f"Replay passed in {result['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
