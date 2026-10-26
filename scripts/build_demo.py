"""Build the published demo using preselected TRAIN query texts, without labels."""

import argparse
import json
from pathlib import Path

import joblib
import polars as pl

from catalog_search.data import save_json
from catalog_search.engine import build_index


EXAMPLE_IDS = ["2212", "108169", "3250", "41324", "2995", "10858"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--size", type=int, default=30000)
    args = parser.parse_args()
    # Selection reads IDs/text/partition only; no relevance value is consulted.
    queries = pl.read_parquet(args.data / "queries.parquet").filter(pl.col("query_id").is_in(EXAMPLE_IDS))
    if queries.height != len(EXAMPLE_IDS) or set(queries["partition"]) != {"train"}:
        raise ValueError("Published example IDs must exist in the pinned training partition")
    candidates = pl.read_parquet(args.data / "train.parquet", columns=["query_id", "product_id"])
    include_ids = candidates.filter(pl.col("query_id").is_in(EXAMPLE_IDS))["product_id"].unique().to_list()
    encoder = joblib.load(args.artifacts / "encoder.joblib")
    manifest = build_index(encoder, pl.read_parquet(args.data / "products.parquet"),
                           args.artifacts, size=args.size, include_ids=include_ids)
    texts = dict(queries.select("query_id", "query").iter_rows())
    metadata = {"examples": [texts[value] for value in EXAMPLE_IDS],
                "description": "Исторический англоязычный каталог ESCI. Характеристики сравниваются по тексту товара.",
                "example_provenance": {"partition": "train", "query_ids": EXAMPLE_IDS,
                    "selection": "Manually selected query texts cover power, model, capacity and length; no score-based selection",
                    "candidate_policy": "Include every candidate of those TRAIN queries, regardless of relevance label",
                    "mandatory_product_count": len(include_ids)}}
    save_json(args.artifacts / "metadata.json", metadata)
    print(json.dumps({"index": manifest, "metadata": metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
