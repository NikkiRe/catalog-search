#!/usr/bin/env python3
"""Prepare the official WANDS release for a held-out catalogue transfer test.

The files named *.csv are TAB-separated. Labels remain Exact/Partial/Irrelevant:
Partial is deliberately not silently mapped to an ESCI class. This adapter never
trains, selects, ranks, or calculates a model performance metric. Raw downloads
are kept in memory; the caller must explicitly choose an output directory.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import polars as pl


REPOSITORY = "https://github.com/wayfair/WANDS"
API_BASE = "https://api.github.com/repos/wayfair/WANDS"
RAW_BASE = "https://raw.githubusercontent.com/wayfair/WANDS"
FILES = {
    "query.csv": "dataset/query.csv",
    "product.csv": "dataset/product.csv",
    "label.csv": "dataset/label.csv",
    "LICENSE": "LICENSE",
}
LABELS = frozenset({"Exact", "Partial", "Irrelevant"})
REQUIRED = {
    "query.csv": {"query_id", "query"},
    "product.csv": {"product_id", "product_name", "product_description"},
    "label.csv": {"query_id", "product_id", "label"},
}


def _fetch_bytes(url: str, *, timeout: float = 120.0) -> bytes:
    request = Request(url, headers={"User-Agent": "catalog-search-wands-adapter/1"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _resolve_revision(revision: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    payload = json.loads(_fetch_bytes(f"{API_BASE}/commits/{quote(revision, safe='')}"))
    resolved = payload.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("GitHub did not return a full commit SHA for the requested revision")
    return resolved


def _read_tsv(content: bytes, filename: str) -> list[dict[str, str]]:
    if content.startswith(b"version https://git-lfs.github.com/spec/"):
        raise ValueError(f"{filename}: received a Git LFS pointer, not the dataset")
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
    header = reader.fieldnames or []
    if len(header) != len(set(header)):
        raise ValueError(f"{filename}: duplicate CSV header")
    missing = REQUIRED[filename] - set(header)
    if missing:
        raise ValueError(f"{filename}: missing required columns {sorted(missing)}")
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(16 * 1024 * 1024)
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"{filename}:{line_number}: inconsistent number of fields")
            rows.append({key: value for key, value in raw.items()})
    except csv.Error as error:
        raise ValueError(f"{filename}: invalid tab-separated file: {error}") from error
    finally:
        csv.field_size_limit(previous_limit)
    if not rows:
        raise ValueError(f"{filename}: contains no data rows")
    return rows


def _unique_index(rows: list[dict[str, str]], key: str, filename: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row[key]
        if not value.strip():
            raise ValueError(f"{filename}: blank {key}")
        if value in result:
            raise ValueError(f"{filename}: duplicate {key} {value!r}")
        result[value] = row
    return result


def _normalize_with_audit(contents: dict[str, bytes]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Validate the three TSVs and return pairs, queries and products.

    Identifier strings are preserved, including leading zeroes. Use the domain
    column when combining with ESCI. Brand remains empty because WANDS does not
    publish a dedicated brand column; guessing it here would confound transfer.
    query_category is retained only for diagnostic grouping, never as a feature.
    """
    query_rows = _read_tsv(contents["query.csv"], "query.csv")
    product_rows = _read_tsv(contents["product.csv"], "product.csv")
    label_rows = _read_tsv(contents["label.csv"], "label.csv")
    queries_by_id = _unique_index(query_rows, "query_id", "query.csv")
    products_by_id = _unique_index(product_rows, "product_id", "product.csv")
    if any(not row["query"].strip() for row in query_rows):
        raise ValueError("query.csv: blank query text")

    query_schema = {"query_id": pl.String, "query": pl.String, "query_category": pl.String,
                    "domain": pl.String, "split": pl.String}
    product_schema = {"product_id": pl.String, "title": pl.String, "brand": pl.String,
                      "description": pl.String, "attributes": pl.String, "category": pl.String,
                      "domain": pl.String}
    queries = pl.DataFrame([
        {"query_id": row["query_id"], "query": row["query"],
         "query_category": row.get("query_class", ""), "domain": "wands", "split": "test"}
        for row in query_rows
    ], schema=query_schema)
    products = pl.DataFrame([
        {"product_id": row["product_id"], "title": row["product_name"], "brand": "",
         "description": row["product_description"], "attributes": row.get("product_features", ""),
         "category": row.get("category_hierarchy", row.get("category hierarchy", row.get("product_class", ""))), "domain": "wands"}
        for row in product_rows
    ], schema=product_schema)

    grouped_labels: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in label_rows:
        query_id, product_id, label = row["query_id"], row["product_id"], row["label"]
        if label not in LABELS:
            raise ValueError(f"label.csv: unknown label {label!r}")
        if query_id not in queries_by_id:
            raise ValueError(f"label.csv: unknown query_id {query_id!r}")
        if product_id not in products_by_id:
            raise ValueError(f"label.csv: unknown product_id {product_id!r}")
        key = (query_id, product_id)
        grouped_labels.setdefault(key, []).append({"annotation_id": row.get("id", ""), "label": label})
    labels, ambiguous, ambiguity_ledger = [], [], []
    label_order = {"Irrelevant": 0, "Partial": 1, "Exact": 2}
    for (query_id, product_id), annotations in grouped_labels.items():
        distinct_labels = sorted({entry["label"] for entry in annotations}, key=label_order.get)
        annotation_ids = [entry["annotation_id"] for entry in annotations]
        if len(distinct_labels) > 1:
            ambiguous.append({"query_id": query_id, "product_id": product_id,
                              "labels": [entry["label"] for entry in annotations],
                              "annotation_ids": annotation_ids,
                              "min_label": distinct_labels[0], "max_label": distinct_labels[-1]})
            ambiguity_ledger.append({"query_id": query_id, "product_id": product_id,
                                     "annotations": annotations})
        else:
            label = distinct_labels[0]
            labels.append({"query_id": query_id, "product_id": product_id,
                           "label": label, "is_exact": int(label == "Exact"),
                           "annotation_id": annotation_ids[0], "annotation_ids": annotation_ids})
    labels_frame = pl.DataFrame(labels, schema={"query_id": pl.String, "product_id": pl.String,
                                               "label": pl.String, "is_exact": pl.Int8,
                                               "annotation_id": pl.String, "annotation_ids": pl.List(pl.String)})
    pairs = labels_frame.join(queries, on="query_id", how="left", validate="m:1").join(
        products.drop("domain"), on="product_id", how="left", validate="m:1"
    ).select("query_id", "query", "product_id", "title", "brand", "description", "label",
             "is_exact", "domain", "split", "attributes", "category", "query_category", "annotation_id", "annotation_ids")
    ambiguous_frame = pl.DataFrame(ambiguous, schema={"query_id": pl.String, "product_id": pl.String,
                                                     "labels": pl.List(pl.String), "annotation_ids": pl.List(pl.String),
                                                     "min_label": pl.String, "max_label": pl.String})
    ambiguous_pairs = ambiguous_frame.join(queries, on="query_id", how="left", validate="m:1").join(
        products.drop("domain"), on="product_id", how="left", validate="m:1"
    ).sort("query_id", "product_id")
    duplicate_groups = sum(len(entries) > 1 for entries in grouped_labels.values())
    audit = {"raw_annotation_rows": len(label_rows), "unique_candidate_pairs": len(grouped_labels),
             "duplicate_pair_groups": duplicate_groups,
             "extra_duplicate_annotation_rows": len(label_rows) - len(grouped_labels),
             "same_label_duplicate_groups": duplicate_groups - len(ambiguous),
             "ambiguous_pair_count": len(ambiguous), "ambiguity_ledger": ambiguity_ledger}
    return pairs.sort("query_id", "product_id"), queries.sort("query_id"), products.sort("product_id"), ambiguous_pairs, audit


def normalize_wands(contents: dict[str, bytes]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Normalize using the frozen duplicate policy: consensus collapse, conflict exclusion.

    Every same-label candidate appears once, retaining all annotation IDs.
    Conflicting labels are not adjudicated by this adapter and are excluded from
    the main set. prepare_wands writes their complete ledger and a separate
    sensitivity artifact, including deterministic lower/upper label assignments.
    """
    pairs, queries, products, _, _ = _normalize_with_audit(contents)
    return pairs, queries, products


def prepare_wands(output_dir: Path, *, source_dir: Path | None = None,
                  revision: str = "main") -> dict[str, Any]:
    """Write normalized parquet and provenance; do not persist the raw CSVs.

    Online branch names resolve to immutable commits before downloading. For
    offline input a full SHA is required, and the manifest explicitly records
    that its association with these local bytes is user-supplied, not remotely
    verified. File SHA256 values always describe the bytes actually parsed.
    """
    output_dir = Path(output_dir)
    if source_dir is not None:
        source_dir = Path(source_dir)
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("--source-dir requires --revision with the full source commit SHA")
        resolved = revision.lower()
        contents = {name: (source_dir / name).read_bytes() for name in FILES}
        verification = "local bytes hashed; source commit supplied by caller; remote association not checked"
    else:
        resolved = _resolve_revision(revision)
        contents = {name: _fetch_bytes(f"{RAW_BASE}/{resolved}/{relative}")
                    for name, relative in FILES.items()}
        verification = "downloaded from official repository at resolved full commit SHA"
    license_text = contents["LICENSE"].decode("utf-8-sig")
    if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
        raise ValueError("LICENSE does not contain the expected official MIT grant; review before using")
    pairs, queries, products, ambiguous_pairs, label_audit = _normalize_with_audit(contents)

    manifest: dict[str, Any] = {
        "dataset": "WANDS", "repository": REPOSITORY, "requested_revision": revision,
        "resolved_revision": resolved, "provenance_verification": verification,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "raw_files_saved": False,
        "license": {"identifier": "MIT", "text": license_text,
                    "notice": "Preserve this license notice with copies or substantial portions."},
        "sources": [{"source_id": "wands_" + name.replace(".", "_").lower(),
                     "path": relative, "url": f"{RAW_BASE}/{resolved}/{relative}",
                     "bytes": len(contents[name]), "sha256": hashlib.sha256(contents[name]).hexdigest()}
                    for name, relative in FILES.items()],
        "counts": {"queries": queries.height, "products": products.height, "pairs": pairs.height},
        "annotation_audit": label_audit,
        "ambiguity_policy": "Frozen before model evaluation: collapse identical-label duplicate pairs; exclude label-conflicting pairs from primary. ambiguous_pairs.parquet retains every original label/ID and min/max assignments for sensitivity, never favourable-label selection.",
        "label_counts": {label: pairs.filter(pl.col("label") == label).height for label in sorted(LABELS)},
        "columns": {name: str(dtype) for name, dtype in pairs.schema.items()},
        "split": "all WANDS labels held out as external-domain test; no official WANDS split asserted",
        "label_semantics": "Exact/Partial/Irrelevant preserved; no implicit Partial-to-ESCI mapping",
        "feature_limits": ["brand has no dedicated source field and is empty",
                           "query_category, label, is_exact and annotation_id are not inference features",
                           "Unjudged query-product pairs have no inferred relevance label",
                           "Catalogue and CSV row order do not supply event chronology"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, frame in (("pairs", pairs), ("queries", queries), ("products", products), ("ambiguous_pairs", ambiguous_pairs)):
        destination = output_dir / f"{name}.parquet"
        frame.write_parquet(destination, compression="zstd")
        artifacts[destination.name] = {"bytes": destination.stat().st_size,
                                       "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    manifest["artifacts"] = artifacts
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Explicit local data directory; keep raw/normalized datasets out of source control")
    parser.add_argument("--source-dir", type=Path, help="Offline folder containing query.csv, product.csv, label.csv, LICENSE")
    parser.add_argument("--revision", default="main", help="Official Git revision; offline mode requires a full commit SHA")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_wands(args.output_dir, source_dir=args.source_dir, revision=args.revision)
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(2, f"WANDS preparation failed: {error}\n")
    print(json.dumps({"dataset": manifest["dataset"], "revision": manifest["resolved_revision"],
                      "counts": manifest["counts"], "raw_files_saved": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
