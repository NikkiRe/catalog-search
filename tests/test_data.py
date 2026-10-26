"""Synthetic schema checks for leakage prevention and checked joins."""

import json

import polars as pl
import pytest

from catalog_search import data
from catalog_search.text import stable_hash


def _source(tmp_path, monkeypatch, duplicate=False, missing=False):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        ("test", "shelf 3.5 inch", "test"),
        ("overlap", "shelf 3 1/2 inch", "train"),
        ("one", "red kettle", "train"),
        ("two", "blue chair", "train"),
    ]
    examples = pl.DataFrame([
        {"query_id": query_id, "query": query, "product_id": f"{query_id}-{i}",
         "product_locale": "us", "esci_label": label, "split": split, "small_version": 1}
        for query_id, query, split in rows for i, label in enumerate(("E", "I"))
    ])
    products = pl.DataFrame([
        {"product_id": product_id, "product_locale": "us", "product_title": "A product",
         "product_brand": None, "product_description": None, "product_bullet_point": "width 3.5 inch"}
        for product_id in examples["product_id"]
    ])
    if duplicate:
        examples = pl.concat([examples, examples.head(1)])
    if missing:
        products = products.slice(1)
    examples.write_parquet(raw / "shopping_queries_dataset_examples.parquet")
    products.write_parquet(raw / "shopping_queries_dataset_products.parquet")
    monkeypatch.setattr(data, "ESCI_FILES", {name: data.sha256(raw / name) for name in data.ESCI_FILES})
    return raw


def test_numeric_families_never_cross_partitions_and_test_stays_complete(tmp_path, monkeypatch):
    raw = _source(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    config = {"locale": "us", "seed": 20261026, "development_fraction": .5}
    manifest = data.prepare(raw, output, config)
    queries = pl.read_parquet(output / "queries.parquet")
    assert set(queries["query_id"]) == {"test", "one", "two"}
    assert manifest["excluded_official_train_queries"] == 1
    assert pl.read_parquet(output / "test.parquet").height == 2
    for row in queries.filter(pl.col("query_id") != "test").iter_rows(named=True):
        expected = "dev" if stable_hash(row["family"]) / 2**64 < .5 else "train"
        assert row["partition"] == expected
    assert queries.group_by("family").agg(pl.col("partition").n_unique())["partition"].max() == 1
    assert data.load_pairs(output, "test")["brand"].to_list() == ["", ""]
    assert json.loads((output / "manifest.json").read_text())["prepared_sha256"]["test.parquet"] == data.sha256(output / "test.parquet")
    with pytest.raises(FileExistsError):
        data.prepare(raw, output, config)


@pytest.mark.parametrize("duplicate,missing,message", [(True, False, "Duplicate"), (False, True, "Missing")])
def test_corrupt_candidate_keys_are_errors(tmp_path, monkeypatch, duplicate, missing, message):
    raw = _source(tmp_path, monkeypatch, duplicate=duplicate, missing=missing)
    with pytest.raises(ValueError, match=message):
        data.prepare(raw, tmp_path / "prepared", {"locale": "us", "seed": 1, "development_fraction": .5})


def test_pinned_data_cannot_silently_change(tmp_path, monkeypatch):
    raw = _source(tmp_path, monkeypatch)
    (raw / "shopping_queries_dataset_examples.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="pinned"):
        data.prepare(raw, tmp_path / "prepared", {"locale": "us", "seed": 1, "development_fraction": .5})
