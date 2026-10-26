"""Describe the frozen data, not the quality of a fitted model."""

import argparse
from pathlib import Path

import polars as pl

from catalog_search.data import save_json, sha256


def audit(directory):
    directory = Path(directory)
    products = pl.read_parquet(directory / "products.parquet")
    partitions = {}
    for name in ("train", "dev", "test"):
        pairs = pl.read_parquet(directory / f"{name}.parquet")
        counts = pairs.group_by("query_id").len()["len"]
        queries = pairs.select("query_id", "query").unique()
        partitions[name] = {
            "pairs": pairs.height, "queries": queries.height,
            "label_counts": dict(pairs.group_by("label").len().sort("label").iter_rows()),
            "candidates_per_query": {"min": counts.min(), "median": counts.median(),
                                     "p90": counts.quantile(.9), "max": counts.max()},
            "numeric_query_fraction": queries["query"].str.contains(r"\d").mean(),
        }
    return {
        "scope": "Descriptive audit of prepared metadata and released judgments; no model scores used",
        "manifest_sha256": sha256(directory / "manifest.json"),
        "products": products.height,
        "empty_field_fraction": {field: products[field].str.strip_chars().eq("").mean()
                                 for field in ("title", "brand", "description", "attributes")},
        "partitions": partitions,
        "note": "Fields are shortened under the fixed source limits; empty is not a contradictory specification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data)
    save_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
