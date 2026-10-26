"""Frozen development selection followed by one held-out ranking evaluation.

The encoder sees training products only. All configurations, including the
optional no-product-evidence diagnostic, are chosen on development before any
test predictions are evaluated. WANDS is never used for fitting or selection.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl

from .data import load_pairs, save_json, sha256
from .features import ALL_NAMES, LEXICAL_NAMES, FeatureEncoder
from .metrics import ESCI_GAINS, ESCI_LABELS, evaluate, paired_bootstrap
from .text import family, stable_hash


def _now():
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _source_hashes():
    root = Path(__file__).resolve().parent
    return {f"src/catalog_search/{name}": sha256(root / name) for name in
            ("data.py", "text.py", "features.py", "attributes.py", "metrics.py", "experiment.py")}


def _check_frozen(protocol):
    if _source_hashes() != protocol["source_sha256"]:
        raise RuntimeError("Model/evaluation source changed after protocol freeze; stop before further evaluation")


def _validate_data(data_dir, config):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for name, digest in manifest["prepared_sha256"].items():
        if sha256(data_dir / name) != digest:
            raise ValueError(f"Prepared ESCI checksum mismatch: {name}")
    for key in ("seed", "locale", "development_fraction"):
        if manifest["config"].get(key) != config.get(key):
            raise ValueError(f"Prepared data/config disagree on {key}")
    if manifest.get("normalization_source_sha256") != sha256(Path(__file__).with_name("text.py")):
        raise ValueError("Prepared split uses another text/family implementation; prepare data again")
    queries = pl.read_parquet(data_dir / "queries.parquet")
    families = {part: set(queries.filter(pl.col("partition") == part)["family"])
                for part in ("train", "dev", "test")}
    if any(not value for value in families.values()):
        raise ValueError("Each partition must contain at least one family")
    if any(families[left] & families[right] for left, right in
           (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Query family leakage between partitions")
    if dict(config["gain_map"]) != ESCI_GAINS:
        raise ValueError("Configured primary gains differ from E1/S.1/C.01/I0")
    return manifest


def _variants(config):
    if len(LEXICAL_NAMES) != 28 or len(ALL_NAMES) != 58:
        raise ValueError("Feature contract changed: expected lexical28 and complete58")
    variants = {"baseline": list(range(28)), "attribute": list(range(58))}
    if config.get("include_query_only_ablation", False):
        variants["no_product_evidence"] = list(range(28)) + [
            index for index, name in enumerate(ALL_NAMES)
            if index >= 28 and name.endswith("_query_count")]
        if len(variants["no_product_evidence"]) != 35:
            raise ValueError("Expected seven query-only attribute features in diagnostic ablation")
    return variants


def _ordered_pairs(data_dir, partition):
    frame = load_pairs(data_dir, partition).sort("query_id", "tie_order")
    if frame.height == 0:
        raise ValueError(f"Empty {partition} partition")
    return frame


def _groups(frame):
    # Input must consist of contiguous query blocks. The caller sorts once and
    # keeps exactly that row order in its feature matrices and label arrays.
    identifiers = frame["query_id"].to_numpy()
    runs = 1 + int(np.count_nonzero(identifiers[1:] != identifiers[:-1])) if len(identifiers) else 0
    if frame["query_id"].n_unique() != runs:
        raise ValueError("Rows are not contiguous by query")
    return frame.group_by("query_id", maintain_order=True).len()["len"].to_numpy().astype(np.int32)


def _labels(frame):
    return np.array([ESCI_LABELS[value] for value in frame["label"]], dtype=np.int32)


class _GroupedNDCG:
    """Vectorized custom dev metric with the study's gains and fixed tie order."""

    def __init__(self, labels, groups, tie_order, k=10):
        self.labels = np.asarray(labels)
        self.groups = np.asarray(groups)
        self.codes = np.repeat(np.arange(len(groups)), groups)
        starts = np.cumsum(np.r_[0, groups[:-1]])
        rank = np.arange(len(labels)) - np.repeat(starts, groups)
        self.discount = np.where(rank < k, 1 / np.log2(rank + 2), 0)
        self.tie = np.asarray(tie_order)
        self.gains = np.array([0., .01, .1, 1.])[self.labels]
        ideal_order = np.lexsort((self.tie, -self.gains, self.codes))
        self.ideal = np.bincount(self.codes, weights=self.gains[ideal_order] * self.discount,
                                 minlength=len(groups))

    def __call__(self, labels, scores, weight, group):
        if not np.array_equal(labels, self.labels) or not np.array_equal(group, self.groups):
            raise ValueError("Custom NDCG received another validation row/group order")
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite validation prediction")
        order = np.lexsort((self.tie, -scores, self.codes))
        dcg = np.bincount(self.codes, weights=self.gains[order] * self.discount,
                          minlength=len(self.groups))
        value = np.divide(dcg, self.ideal, out=np.zeros_like(dcg), where=self.ideal > 0).mean()
        return "protocol_ndcg10", float(value), True


def _params(config, leaves, trees):
    return {"objective": "lambdarank", "metric": "None", "label_gain": [0., .01, .1, 1.],
            "n_estimators": int(trees), "num_leaves": int(leaves),
            "learning_rate": float(config["learning_rate"]),
            "min_child_samples": int(config["min_child_samples"]),
            "random_state": int(config["seed"]), "n_jobs": int(config["threads"]),
            "deterministic": True, "force_col_wise": True, "verbosity": -1,
            "lambdarank_truncation_level": 20}


def _matrix(encoder, frame, output, name):
    started = time.perf_counter()
    print(f"Computing {name} features: {frame.height:,} pairs", flush=True)
    values = encoder.transform(frame, progress=True)
    if values.shape != (frame.height, len(ALL_NAMES)) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {name} matrix")
    path = output / "cache" / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    seconds = time.perf_counter() - started
    print(f"{name} features saved: {values.shape}, {seconds:.1f}s", flush=True)
    del values
    gc.collect()
    return np.load(path, mmap_mode="r"), {"rows": frame.height, "features": len(ALL_NAMES),
                                          "seconds": seconds, "sha256": sha256(path)}


def _select(train, dev, x_train, x_dev, variants, config):
    y_train, y_dev = _labels(train), _labels(dev)
    groups_train, groups_dev = _groups(train), _groups(dev)
    metric = _GroupedNDCG(y_dev, groups_dev, dev["tie_order"].to_numpy())
    selections = {}
    for variant, columns in variants.items():
        trial_records = []
        fit_x = np.ascontiguousarray(x_train[:, columns])
        valid_x = np.ascontiguousarray(x_dev[:, columns])
        for leaves in config["num_leaves"]:
            print(f"Development selection: {variant}, leaves={leaves}", flush=True)
            started = time.perf_counter()
            model = lgb.LGBMRanker(**_params(config, leaves, config["max_iterations"]))
            model.fit(fit_x, y_train, group=groups_train,
                      eval_set=[(valid_x, y_dev)], eval_group=[groups_dev], eval_metric=metric,
                      callbacks=[lgb.early_stopping(config["early_stopping_rounds"], first_metric_only=True, verbose=False),
                                 lgb.log_evaluation(50)])
            trees = int(model.best_iteration_ or config["max_iterations"])
            scores = model.predict(valid_x, num_iteration=trees)
            measured = metric(y_dev, scores, None, groups_dev)[1]
            trial_records.append({"num_leaves": int(leaves), "trees": trees,
                                  "dev_ndcg10": measured, "fit_seconds": time.perf_counter() - started})
            print(f"Selected-iteration dev NDCG@10: {measured:.6f}; trees={trees}", flush=True)
            del model
            gc.collect()
        best = min(trial_records, key=lambda row: (-row["dev_ndcg10"], row["num_leaves"], row["trees"]))
        selections[variant] = {**best, "trials": trial_records, "columns": columns,
                               "feature_names": [ALL_NAMES[index] for index in columns]}
        del fit_x, valid_x
    return selections


def _predict(models, matrices, variants):
    result = {}
    for name, model in models.items():
        scores = model.predict(np.ascontiguousarray(matrices[:, variants[name]]))
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite {name} scores")
        result[name] = np.asarray(scores, float)
    return result


def _with_scores(frame, matrix, models, variants):
    scores = _predict(models, matrix, variants)
    scores["bm25"] = np.asarray(matrix[:, ALL_NAMES.index("body_bm25")], float)
    scores["tfidf"] = np.asarray(matrix[:, ALL_NAMES.index("body_tfidf")], float)
    conflicts = np.asarray(matrix[:, ALL_NAMES.index("attr_has_conflict")]) > .5
    baseline = scores["baseline"]
    rule = baseline.copy()
    offset = 0
    for count in _groups(frame):
        end = offset + int(count)
        penalty = np.ptp(baseline[offset:end]) + 1
        rule[offset:end] -= conflicts[offset:end] * penalty
        offset = end
    scores["rule"] = rule
    metadata = frame.select("query_id", "product_id", "family", "label", "tie_order")
    return metadata.with_columns([pl.Series(name, values) for name, values in scores.items()]), list(scores)


def _metric_aliases(means):
    return {name: {**values, "ndcg_at_10": values["ndcg10"]} for name, values in means.items()}


def _slices(frame, matrix, per_query, score_columns, fitted_product_ids):
    flags = frame.select("query_id").with_columns(
        pl.Series("has_attribute", np.asarray(matrix[:, ALL_NAMES.index("attr_query_count")]) > 0),
        pl.Series("product_seen", [value in fitted_product_ids for value in frame["product_id"]])
    ).group_by("query_id").agg(pl.col("has_attribute").any(), pl.col("product_seen").any())
    joined = per_query.join(flags, on="query_id", validate="1:1")
    conditions = {"attribute_mentioned": pl.col("has_attribute"), "no_attribute_mentioned": ~pl.col("has_attribute"),
                  "all_candidates_new_to_final_fit": ~pl.col("product_seen"),
                  "some_candidate_seen_in_final_fit": pl.col("product_seen")}
    result = {}
    for name, condition in conditions.items():
        selected = joined.filter(condition)
        means = {mode: {metric: selected[f"{mode}_{metric}"].mean()
                        for metric in ("ndcg10", "ndcg_all", "exact_recall10", "mrr")}
                 for mode in score_columns} if selected.height else {}
        result[name] = {"queries": selected.height, "families": selected["family"].n_unique(),
                        "metrics": means, "status": "predeclared descriptive slice; no multiple-testing claim"}
    return result


def _external_frame(path, config, label_column="label"):
    frame = pl.read_parquet(path)
    if label_column != "label":
        frame = frame.with_columns(pl.col(label_column).alias("label"))
    return frame.with_columns(
        pl.col("query_id").cast(pl.String), pl.col("product_id").cast(pl.String),
        pl.col("title").fill_null("").str.slice(0, 512), pl.col("brand").fill_null("").str.slice(0, 128),
        pl.col("description").fill_null("").str.slice(0, 1536), pl.col("attributes").fill_null("").str.slice(0, 1536),
        pl.col("query").map_elements(family, return_dtype=pl.String).alias("family"),
        pl.struct("query_id", "product_id").map_elements(
            lambda row: stable_hash(row["query_id"] + "|" + row["product_id"], config["seed"]),
            return_dtype=pl.UInt64).alias("tie_order")
    ).sort("query_id", "tie_order")


def _evaluate_wands(wands_dir, encoder, models, variants, output, reports, config):
    main = _external_frame(wands_dir / "pairs.parquet", config)
    matrix, timing = _matrix(encoder, main, output, "wands")
    scored, columns = _with_scores(main, matrix, models, variants)
    scored.write_parquet(reports / "wands_scores.parquet")
    datasets = {"consensus_main": scored}
    ambiguity_path = wands_dir / "ambiguous_pairs.parquet"
    if ambiguity_path.exists():
        ambiguous = _external_frame(ambiguity_path, config, label_column="min_label")
        if ambiguous.height:
            ambiguity_matrix, _ = _matrix(encoder, ambiguous, output, "wands_ambiguous")
            ambiguity_scores, _ = _with_scores(ambiguous, ambiguity_matrix, models, variants)
            # Rule demotion depends on the full candidate list, so recompute it
            # after concatenation under each label scenario (never using labels).
            for scenario, source_label in (("ambiguous_min", "min_label"), ("ambiguous_max", "max_label")):
                joined = ambiguity_scores.with_columns(pl.Series("label", ambiguous[source_label]))
                combined = pl.concat([scored, joined]).sort("query_id", "tie_order")
                base = combined["baseline"].to_numpy()
                # The base/attribute scores are independent of candidate pool.
                # Reconstruct has_conflict for full-list rule scoring from IDs.
                conflict_map = dict(zip(zip(main["query_id"], main["product_id"]),
                                        np.asarray(matrix[:, ALL_NAMES.index("attr_has_conflict")]) > .5))
                conflict_map.update(dict(zip(zip(ambiguous["query_id"], ambiguous["product_id"]),
                                            np.asarray(ambiguity_matrix[:, ALL_NAMES.index("attr_has_conflict")]) > .5)))
                conflict = np.array([conflict_map[(q, p)] for q, p in combined.select("query_id", "product_id").iter_rows()])
                rule, start = base.copy(), 0
                for count in _groups(combined):
                    end = start + int(count)
                    rule[start:end] -= conflict[start:end] * (np.ptp(base[start:end]) + 1)
                    start = end
                datasets[scenario] = combined.with_columns(pl.Series("rule", rule))
    maps = {"exact_binary": config["wands_primary_gain_map"]}
    maps.update({f"partial_{gain:g}": {"Irrelevant": 0., "Partial": float(gain), "Exact": 1.}
                 for gain in config["wands_sensitivity_partial_gains"]})
    outcomes = {}
    for dataset_name, dataset in datasets.items():
        outcomes[dataset_name] = {}
        for map_name, gain_map in maps.items():
            means, per_query = evaluate(dataset, columns, gain_map=gain_map)
            comparison = paired_bootstrap(per_query, repeats=config["bootstrap_repeats"], seed=config["seed"])
            filename = f"wands_{dataset_name}_{map_name.replace('.', 'p')}_per_query.parquet"
            per_query.write_parquet(reports / filename)
            outcomes[dataset_name][map_name] = {"models": _metric_aliases(means),
                "primary_difference": comparison, "gain_map": gain_map, "queries": per_query.height,
                "families": per_query["family"].n_unique(), "pairs": dataset.height,
                "zero_idcg_queries": int(per_query["zero_idcg"].sum())}
    return {"status": "frozen external evaluation; no target fitting/selection", "primary": "consensus_main/exact_binary",
            "scenarios": outcomes, "feature_compute": timing,
            "ambiguity_note": "Min/max are predetermined label scenarios, not adjudicated truth or sharp metric bounds."}


def run_experiment(data_dir, wands_dir, output, config):
    """Fit/select on ESCI train/dev, seal selection, refit and evaluate once.

    ``output`` is the artifact directory. Reports live in its sibling reports/.
    The caller supplies already prepared, checksum-verified datasets. Existing
    protocol or selection artifacts are refused to prevent accidental retuning.
    """
    started = time.perf_counter()
    data_dir, wands_dir, output = Path(data_dir), Path(wands_dir), Path(output)
    config = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else json.loads(json.dumps(config))
    reports = output.parent / "reports"
    if (output / "protocol.json").exists() or (output / "selection.json").exists() or (reports / "summary.json").exists():
        raise FileExistsError("Experiment already frozen or evaluated; use a new artifact/report directory")
    manifest = _validate_data(data_dir, config)
    variants = _variants(config)
    wands_manifest = json.loads((wands_dir / "manifest.json").read_text())
    for name, record in wands_manifest["artifacts"].items():
        if sha256(wands_dir / name) != record["sha256"]:
            raise ValueError(f"Prepared WANDS checksum mismatch: {name}")
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    protocol = {"status": "frozen_before_training_and_test_quality", "frozen_at_utc": _now(),
        "config": config, "config_sha256": _canonical_hash(config), "source_sha256": _source_hashes(),
        "esci_manifest_sha256": sha256(data_dir / "manifest.json"),
        "wands_manifest_sha256": sha256(wands_dir / "manifest.json"),
        "esci_prepared_sha256": manifest["prepared_sha256"],
        "wands_prepared_sha256": {name: value["sha256"] for name, value in wands_manifest["artifacts"].items()},
        "feature_names": list(ALL_NAMES), "variants": {name: [ALL_NAMES[i] for i in cols] for name, cols in variants.items()},
        "model_params": {str(leaves): _params(config, leaves, config["max_iterations"]) for leaves in config["num_leaves"]},
        "selection": "Each variant: highest dev linear-gain NDCG@10; ties fewer leaves then fewer trees. Equal two-leaf budget.",
        "encoder_fit": "Unique TRAIN products only; frozen during train+dev final refit and external transfer",
        "primary": config["primary"], "rule": "Demote attr_has_conflict to bottom of each query, preserve baseline order",
        "bm25": "body_bm25 feature", "tfidf": "body_tfidf feature",
        "bootstrap": "Paired normalized-family resampling; query-weighted statistic; seeded 95% percentile interval",
        "wands": "Exact binary primary; predeclared Partial gains and ambiguity min/max scenarios; no tuning",
        "software": {name: importlib.metadata.version(name) for name in ("numpy", "polars", "scikit-learn", "lightgbm", "joblib")},
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "threads": config["threads"]}}
    save_json(output / "protocol.json", protocol)
    print(f"Protocol frozen: {_canonical_hash(protocol)}", flush=True)

    train, dev = _ordered_pairs(data_dir, "train"), _ordered_pairs(data_dir, "dev")
    training_products = train.select("product_id", "title", "brand", "description", "attributes").unique("product_id").sort("product_id")
    print(f"Encoder fit: {training_products.height:,} unique training products", flush=True)
    encoder = FeatureEncoder(config["word_features"], config["character_features"]).fit(training_products)
    joblib.dump(encoder, output / "encoder.joblib", compress=3)
    x_train, train_timing = _matrix(encoder, train, output, "train")
    x_dev, dev_timing = _matrix(encoder, dev, output, "dev")
    _check_frozen(protocol)
    choices = _select(train, dev, x_train, x_dev, variants, config)
    examples = train.select("query").unique()["query"].to_list()
    examples = sorted(examples, key=lambda text: stable_hash(text, config["seed"]))[:6]
    selection = {"status": "selected_before_any_test_quality", "selected_at_utc": _now(),
                 "protocol_sha256": sha256(output / "protocol.json"), "config_sha256": protocol["config_sha256"],
                 "models": choices, "examples": examples, "encoder_documents": encoder.n_fit_documents,
                 "encoder_sha256": sha256(output / "encoder.joblib")}
    save_json(output / "selection.json", selection)
    print("Selection sealed; starting final train+dev refits", flush=True)

    fit_frame = pl.concat([train, dev])
    fit_x = np.concatenate([x_train, x_dev], axis=0)
    fit_y, fit_groups = _labels(fit_frame), _groups(fit_frame)
    models, model_metadata = {}, {}
    for name, columns in variants.items():
        _check_frozen(protocol)
        choice = choices[name]
        print(f"Final fit {name}: leaves={choice['num_leaves']}, trees={choice['trees']}", flush=True)
        model = lgb.LGBMRanker(**_params(config, choice["num_leaves"], choice["trees"]))
        model.fit(np.ascontiguousarray(fit_x[:, columns]), fit_y, group=fit_groups)
        model.booster_.save_model(str(output / f"{name}.txt"))
        models[name] = model
        model_metadata[name] = {"features": len(columns), "num_leaves": choice["num_leaves"],
                                "trees": choice["trees"], "sha256": sha256(output / f"{name}.txt")}
    fitted_product_ids = set(fit_frame["product_id"])
    del fit_x, fit_frame, train, dev, x_train, x_dev, training_products
    gc.collect()
    _check_frozen(protocol)
    print("Final models frozen; opening ESCI test quality", flush=True)
    test = _ordered_pairs(data_dir, "test")
    x_test, test_timing = _matrix(encoder, test, output, "test")
    scored, score_columns = _with_scores(test, x_test, models, variants)
    scored.write_parquet(reports / "test_scores.parquet")
    means, per_query = evaluate(scored, score_columns)
    per_query.write_parquet(reports / "per_query.parquet")
    difference = paired_bootstrap(per_query, repeats=config["bootstrap_repeats"], seed=config["seed"])
    slices = _slices(test, x_test, per_query, score_columns, fitted_product_ids)
    del test, x_test, scored
    gc.collect()
    _check_frozen(protocol)
    print("Opening frozen WANDS transfer evaluation", flush=True)
    wands = _evaluate_wands(wands_dir, encoder, models, variants, output, reports, config)
    _check_frozen(protocol)
    summary = {"status": "completed", "completed_at_utc": _now(),
        "dataset": {"name": "ESCI", "locale": config["locale"], "subset": "small_version == 1",
                    "source_revision": manifest["source_revision"], "products": manifest["products"],
                    "queries": manifest["queries"], "pairs": manifest["rows"],
                    "excluded_official_train_queries": manifest["excluded_official_train_queries"],
                    "excluded_families": manifest["excluded_families"],
                    "test_products_seen_in_train": manifest["test_products_seen_in_train"]},
        "splits": manifest["partition_counts"], "gains": config["gain_map"],
        "models": model_metadata, "esci": _metric_aliases(means), "primary_difference": difference,
        "slices": slices, "wands": wands, "selection": selection,
        "feature_compute": {"train": train_timing, "dev": dev_timing, "test": test_timing},
        "provenance": {"protocol_sha256": sha256(output / "protocol.json"),
                       "selection_sha256": sha256(output / "selection.json"),
                       "source_sha256": protocol["source_sha256"], "config_sha256": protocol["config_sha256"],
                       "esci_input_files": manifest["input_files"], "wands_sources": wands_manifest["sources"]},
        "runtime_seconds": time.perf_counter() - started,
        "limitations": ["Reranking judged pools; no full-catalogue recall or purchase/profit effect measured",
                        "Challenge-selected queries are not representative traffic",
                        "Current released test is public, with an internally frozen selection protocol",
                        "WANDS Partial is not ESCI Substitute; transfer gain/ambiguity scenarios are descriptive",
                        "Attribute evidence is conservative and soft, not certified physical compatibility",
                        "No new independent human attribute labels are claimed by this experiment"]}
    save_json(reports / "metrics.json", {"esci": summary["esci"], "primary_difference": difference,
                                        "slices": slices, "wands": wands})
    save_json(reports / "summary.json", summary)
    print(f"Experiment completed in {summary['runtime_seconds']:.1f}s", flush=True)
    return summary
