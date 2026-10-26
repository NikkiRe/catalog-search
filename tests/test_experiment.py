"""Actual tiny ranking lifecycle; synthetic scores are not benchmark evidence."""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from catalog_search import experiment
from catalog_search.data import save_json, sha256
from catalog_search.features import ALL_NAMES
from catalog_search.text import family, stable_hash


def _prepared_fixture(tmp_path, config):
    data, wands = tmp_path / "synthetic_esci", tmp_path / "synthetic_wands"
    data.mkdir()
    wands.mkdir()
    products, queries, partitions = [], [], {}
    seed = config["seed"]
    for partition, marker in (("train", "trainexclusive"), ("dev", "devexclusive"),
                              ("test", "testexclusive")):
        items = [
            ("USB charger 65W", "charger"), ("USB charger 30W", "charger"),
            ("SSD storage 1TB", "storage"), ("SSD storage 2TB", "storage"),
            ("wooden shelf 24 inch", "shelf"), ("wooden shelf 36 inch", "shelf"),
        ]
        ids = []
        for index, (title, _) in enumerate(items):
            identifier = f"{partition}-product-{index}"
            ids.append(identifier)
            products.append({"product_id": identifier, "product_locale": "us",
                             "title": f"catalog product {marker} {title}", "brand": "",
                             "description": f"catalog product {marker} available", "attributes": ""})
        rows = []
        for index, (query, relevant) in enumerate((("USB charger 65W", 0),
                                                   ("SSD storage 1TB", 2),
                                                   ("wooden shelf 24 inch", 4))):
            query = f"{partition} {query}"
            identifier = f"{partition}-query-{index}"
            queries.append({"query_id": identifier, "query": query, "split": "test" if partition == "test" else "train",
                            "family": family(query), "partition": partition})
            for item_index, product_id in enumerate(ids):
                rows.append({"query_id": identifier, "query": query, "product_id": product_id,
                             "product_locale": "us", "label": "E" if item_index == relevant else "I",
                             "family": family(query), "partition": partition,
                             "tie_order": stable_hash(identifier + "|" + product_id, seed)})
        frame = pl.DataFrame(rows).with_columns(pl.col("tie_order").cast(pl.UInt64))
        frame.write_parquet(data / f"{partition}.parquet")
        partitions[partition] = frame
    pl.DataFrame(products).write_parquet(data / "products.parquet")
    pl.DataFrame(queries).write_parquet(data / "queries.parquet")
    save_json(data / "manifest.json", {
        "source_revision": "synthetic-fixture-not-upstream-data", "input_files": {},
        "config": config, "normalization_source_sha256": sha256(Path(experiment.__file__).with_name("text.py")),
        "prepared_sha256": {path.name: sha256(path) for path in sorted(data.glob("*.parquet"))},
        "products": len(products), "queries": len(queries), "rows": sum(f.height for f in partitions.values()),
        "excluded_official_train_queries": 0, "excluded_families": 0, "test_products_seen_in_train": 0,
        "partition_counts": [{"partition": name, "pairs": frame.height, "queries": frame["query_id"].n_unique(),
                              "families": frame["family"].n_unique()} for name, frame in partitions.items()],
    })
    external = pl.DataFrame([
        {"query_id": f"w-query-{query_index}", "query": f"external {kind}",
         "product_id": f"w-product-{item}", "title": f"catalog product wandsexclusive {title}",
         "brand": "", "description": "external catalog product", "attributes": "",
         "label": label}
        for query_index, kind in enumerate(("charger 65W", "shelf 24 inch"))
        for item, (title, label) in enumerate((("charger 65W", "Exact"),
                                               ("charger 30W", "Partial"), ("wooden shelf", "Irrelevant")))
    ])
    external.write_parquet(wands / "pairs.parquet")
    save_json(wands / "manifest.json", {"sources": [], "artifacts": {
        "pairs.parquet": {"sha256": sha256(wands / "pairs.parquet")}}})
    return data, wands


def test_real_tiny_train_select_save_evaluate_and_refuse_overwrite(tmp_path, monkeypatch):
    config = {"seed": 31, "locale": "us", "development_fraction": .15,
              "word_features": 200, "character_features": 200, "num_leaves": [2, 3],
              "max_iterations": 3, "early_stopping_rounds": 1, "min_child_samples": 1,
              "learning_rate": .1, "threads": 1, "bootstrap_repeats": 20,
              "include_query_only_ablation": True, "primary": "synthetic integration only",
              "gain_map": {"I": 0, "C": .01, "S": .1, "E": 1},
              "wands_primary_gain_map": {"Irrelevant": 0, "Partial": 0, "Exact": 1},
              "wands_sensitivity_partial_gains": [.1, .5]}
    data, wands = _prepared_fixture(tmp_path, config)
    artifacts = tmp_path / "run" / "artifacts"
    reports = artifacts.parent / "reports"
    original_ordered = experiment._ordered_pairs
    test_opened = []

    def guarded_ordered(directory, partition):
        if partition == "test":
            # These must exist before any test features or quality are opened.
            selected = json.loads((artifacts / "selection.json").read_text())
            assert selected["status"] == "selected_before_any_test_quality"
            assert selected["protocol_sha256"] == sha256(artifacts / "protocol.json")
            for name in ("baseline", "attribute", "no_product_evidence"):
                assert (artifacts / f"{name}.txt").is_file()
            test_opened.append(True)
        return original_ordered(directory, partition)

    monkeypatch.setattr(experiment, "_ordered_pairs", guarded_ordered)
    summary = experiment.run_experiment(data, wands, artifacts, config)
    assert test_opened == [True]
    assert summary["status"] == "completed"
    assert summary["gains"] == config["gain_map"]
    assert summary["wands"]["scenarios"]["consensus_main"]["exact_binary"]["gain_map"] == config["wands_primary_gain_map"]
    assert set(summary["selection"]["models"]) == {"baseline", "attribute", "no_product_evidence"}
    for name, dimension in (("baseline", 28), ("attribute", 58), ("no_product_evidence", 35)):
        selected = summary["selection"]["models"][name]
        assert len(selected["trials"]) == 2
        assert {row["num_leaves"] for row in selected["trials"]} == {2, 3}
        assert 1 <= selected["trees"] <= 3
        booster = lgb.Booster(model_file=str(artifacts / f"{name}.txt"))
        assert booster.num_feature() == dimension
        assert summary["models"][name]["sha256"] == sha256(artifacts / f"{name}.txt")
    encoder = joblib.load(artifacts / "encoder.joblib")
    assert encoder.n_fit_documents == 6
    vocabulary = set(encoder.word.vocabulary_)
    assert "trainexclusive" in vocabulary
    assert vocabulary.isdisjoint({"devexclusive", "testexclusive", "wandsexclusive"})
    test = original_ordered(data, "test")
    matrix = encoder.transform(test)
    scored = pl.read_parquet(reports / "test_scores.parquet")
    assert scored.height == test.height == 18
    assert scored["query_id"].n_unique() == 3
    for name, selected in summary["selection"]["models"].items():
        booster = lgb.Booster(model_file=str(artifacts / f"{name}.txt"))
        np.testing.assert_allclose(booster.predict(matrix[:, selected["columns"]]), scored[name].to_numpy())
    assert matrix.shape[1] == len(ALL_NAMES)
    assert pl.read_parquet(reports / "per_query.parquet").height == 3
    assert json.loads((reports / "summary.json").read_text()) == summary
    before = {path.relative_to(artifacts.parent): sha256(path) for path in artifacts.parent.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="already frozen"):
        experiment.run_experiment(data, wands, artifacts, config)
    assert before == {path.relative_to(artifacts.parent): sha256(path) for path in artifacts.parent.rglob("*") if path.is_file()}
