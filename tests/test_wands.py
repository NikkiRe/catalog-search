"""WANDS adapter checks use synthetic fixtures, never fabricated evaluation labels."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest


SPEC = importlib.util.spec_from_file_location("catalog_search_prepare_wands", Path(__file__).parents[1] / "scripts" / "prepare_wands.py")
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
REVISION = "a" * 40


@pytest.fixture
def contents():
    return {
        "query.csv": b"query_id\tquery\tquery_class\n001\t18 inch chair\tChairs\n002\tblue sofa\tSofas\n",
        "product.csv": ("product_id\tproduct_name\tproduct_description\tproduct_features\tcategory_hierarchy\n"
                        "010\tChair 18 inches\t\theight:18 inches\tFurniture/Chairs\n"
                        '011\tSofa\t"Blue, comfortable\nsofa"\tcolor:blue\tFurniture/Sofas\n').encode(),
        "label.csv": b"id\tquery_id\tproduct_id\tlabel\n0\t001\t010\tExact\n1\t001\t011\tIrrelevant\n2\t002\t011\tPartial\n",
        "LICENSE": b"MIT License\nCopyright fixture\nPermission is hereby granted\n",
    }


def test_normalized_schema_preserves_ids_labels_and_missing_metadata(contents):
    pairs, queries, products = adapter.normalize_wands(contents)
    assert pairs.height == 3 and queries.height == products.height == 2
    assert pairs["query_id"].to_list() == ["001", "001", "002"]
    assert pairs["product_id"].to_list() == ["010", "011", "011"]
    assert set(pairs["label"]) == {"Exact", "Partial", "Irrelevant"}
    assert pairs["is_exact"].to_list() == [1, 0, 0]
    assert products["brand"].to_list() == ["", ""]
    assert products["description"].to_list() == ["", "Blue, comfortable\nsofa"]
    assert pairs["domain"].unique().to_list() == ["wands"]
    assert pairs["split"].unique().to_list() == ["test"]
    assert "gain" not in pairs.columns


@pytest.mark.parametrize("field,replacement,match", [
    ("label.csv", b"query_id\tproduct_id\tlabel\n001\t010\tSubstitute\n", "unknown label"),
    ("label.csv", b"query_id\tproduct_id\tlabel\n999\t010\tExact\n", "unknown query_id"),
    ("label.csv", b"query_id\tproduct_id\tlabel\n001\t999\tExact\n", "unknown product_id"),
    ("query.csv", b"query_id\tquery\n001\tfirst\n001\tsecond\n", "duplicate query_id"),
    ("product.csv", b"product_id\tproduct_name\tproduct_description\n010\ta\tb\n010\tc\td\n", "duplicate product_id"),
])
def test_invalid_labels_or_relations_fail_loudly(contents, field, replacement, match):
    contents[field] = replacement
    with pytest.raises(ValueError, match=match):
        adapter.normalize_wands(contents)


def test_git_lfs_pointer_and_wrong_delimiter_are_rejected(contents):
    contents["query.csv"] = b"version https://git-lfs.github.com/spec/v1\noid sha256:example\n"
    with pytest.raises(ValueError, match="LFS pointer"):
        adapter.normalize_wands(contents)
    contents["query.csv"] = b"query_id,query\n1,chair\n"
    with pytest.raises(ValueError, match="missing required columns"):
        adapter.normalize_wands(contents)


def test_offline_preparation_has_actual_file_hashes_license_and_roundtrip(contents, tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    for name, content in contents.items():
        (source / name).write_bytes(content)
    output = tmp_path / "normalized"
    manifest = adapter.prepare_wands(output, source_dir=source, revision=REVISION)
    assert manifest["counts"] == {"queries": 2, "products": 2, "pairs": 3}
    assert manifest["resolved_revision"] == REVISION
    assert not manifest["raw_files_saved"]
    assert "remote association not checked" in manifest["provenance_verification"]
    assert manifest["license"]["text"] == contents["LICENSE"].decode()
    for entry in manifest["sources"]:
        name = entry["path"].split("/")[-1]
        assert entry["sha256"] == hashlib.sha256(contents[name]).hexdigest()
        assert entry["bytes"] == len(contents[name])
        assert f"/{REVISION}/" in entry["url"]
    assert set(p.name for p in output.iterdir()) == {"pairs.parquet", "queries.parquet", "products.parquet", "ambiguous_pairs.parquet", "manifest.json"}
    assert json.loads((output / "manifest.json").read_text())["label_counts"] == {"Exact": 1, "Irrelevant": 1, "Partial": 1}
    assert pl.read_parquet(output / "pairs.parquet").equals(adapter.normalize_wands(contents)[0])
    second = tmp_path / "second"
    adapter.prepare_wands(second, source_dir=source, revision=REVISION)
    assert (second / "pairs.parquet").read_bytes() == (output / "pairs.parquet").read_bytes()


def test_online_download_resolves_branch_before_requesting_data(contents, tmp_path, monkeypatch):
    urls = []
    def fetch(url, **kwargs):
        urls.append(url)
        if "/commits/" in url:
            return json.dumps({"sha": REVISION}).encode()
        return contents[url.split("/")[-1]]
    monkeypatch.setattr(adapter, "_fetch_bytes", fetch)
    manifest = adapter.prepare_wands(tmp_path / "download", revision="main")
    assert manifest["resolved_revision"] == REVISION
    assert len(urls) == 5 and "/commits/main" in urls[0]
    assert all(f"/{REVISION}/" in url for url in urls[1:])


def test_offline_source_cannot_claim_unresolved_main_revision(tmp_path):
    with pytest.raises(ValueError, match="full source commit SHA"):
        adapter.prepare_wands(tmp_path / "out", source_dir=tmp_path, revision="main")


def test_same_label_duplicates_collapse_with_all_annotation_ids(contents):
    contents["label.csv"] += b"3\t001\t010\tExact\n"
    pairs, _, _, ambiguous, audit = adapter._normalize_with_audit(contents)
    assert pairs.height == 3 and ambiguous.height == 0
    pair = pairs.filter((pl.col("query_id") == "001") & (pl.col("product_id") == "010"))
    assert pair["annotation_ids"].to_list() == [["0", "3"]]
    assert audit["raw_annotation_rows"] == 4
    assert audit["unique_candidate_pairs"] == 3
    assert audit["same_label_duplicate_groups"] == 1
    assert audit["extra_duplicate_annotation_rows"] == 1


def test_conflicting_pair_is_excluded_without_favourable_label_selection(contents, tmp_path):
    contents["label.csv"] += b"3\t001\t010\tPartial\n4\t001\t010\tExact\n"
    pairs, _, _, ambiguous, audit = adapter._normalize_with_audit(contents)
    assert pairs.height == 2 and ambiguous.height == 1
    assert not pairs.filter((pl.col("query_id") == "001") & (pl.col("product_id") == "010")).height
    row = ambiguous.row(0, named=True)
    assert row["labels"] == ["Exact", "Partial", "Exact"]
    assert row["annotation_ids"] == ["0", "3", "4"]
    assert row["min_label"] == "Partial" and row["max_label"] == "Exact"
    assert audit["ambiguity_ledger"] == [{"query_id": "001", "product_id": "010", "annotations": [
        {"annotation_id": "0", "label": "Exact"}, {"annotation_id": "3", "label": "Partial"},
        {"annotation_id": "4", "label": "Exact"}]}]
    source = tmp_path / "raw"
    source.mkdir()
    for name, content in contents.items():
        (source / name).write_bytes(content)
    manifest = adapter.prepare_wands(tmp_path / "out", source_dir=source, revision=REVISION)
    assert manifest["annotation_audit"]["ambiguous_pair_count"] == 1
    assert pl.read_parquet(tmp_path / "out" / "ambiguous_pairs.parquet").height == 1


def test_real_category_hierarchy_header_is_retained(contents):
    contents["product.csv"] = contents["product.csv"].replace(b"category_hierarchy", b"category hierarchy")
    _, _, products = adapter.normalize_wands(contents)
    assert products["category"].to_list() == ["Furniture/Chairs", "Furniture/Sofas"]
