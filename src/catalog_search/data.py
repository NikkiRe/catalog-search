"""Pinned public data ingestion and family-disjoint ranking splits."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import polars as pl

from .text import family, stable_hash

ESCI_REVISION = "7916cdf6ab75a462e77f20ab40428a10923998d5"
ESCI_FILES = {
    "shopping_queries_dataset_examples.parquet": "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
    "shopping_queries_dataset_products.parquet": "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def download(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in ESCI_FILES.items():
        url = f"https://media.githubusercontent.com/media/amazon-science/esci-data/{ESCI_REVISION}/shopping_queries_dataset/{name}"
        path = output / name
        if not path.exists():
            part = path.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as stream:
                while block := response.read(1 << 20):
                    stream.write(block)
            part.rename(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}")
        records.append({"name": name, "url": url, "bytes": path.stat().st_size, "sha256": actual})
    for name in ("LICENSE", "NOTICE"):
        url = f"https://raw.githubusercontent.com/amazon-science/esci-data/{ESCI_REVISION}/{name}"
        path = output / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
    save_json(output / "manifest.json", {"revision": ESCI_REVISION, "files": records})
    return records


def prepare(raw, output, config):
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared data exists; use a new directory")
    input_hashes = {name: sha256(raw / name) for name in ESCI_FILES}
    if input_hashes != ESCI_FILES:
        raise ValueError("Input data differs from the pinned ESCI snapshot")
    examples = pl.read_parquet(raw / "shopping_queries_dataset_examples.parquet").filter(
        (pl.col("small_version") == 1) & (pl.col("product_locale") == config["locale"]))
    examples = examples.select("query_id", "query", "product_id", "product_locale", "esci_label", "split").rename({"esci_label": "label"})
    examples = examples.with_columns(pl.col("query_id").cast(pl.String), pl.col("product_id").cast(pl.String))
    if set(examples["label"].unique()) - {"E", "S", "C", "I"}:
        raise ValueError("Unknown relevance label")
    if set(examples["split"].unique()) != {"train", "test"}:
        raise ValueError("Unexpected released split")
    query_table = examples.select("query_id", "query", "split").unique()
    if query_table["query_id"].n_unique() != query_table.height:
        raise ValueError("Query ID has multiple texts or splits")
    duplicates = examples.group_by("query_id", "product_id").len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError("Duplicate ESCI judged pair; inspect before defining policy")
    query_table = query_table.with_columns(pl.col("query").map_elements(family, return_dtype=pl.String).alias("family"))
    test_families = set(query_table.filter(pl.col("split") == "test")["family"])
    train_queries = query_table.filter(pl.col("split") == "train")
    removed = train_queries.filter(pl.col("family").is_in(test_families))
    allowed = train_queries.filter(~pl.col("family").is_in(test_families)).with_columns(
        pl.col("family").map_elements(lambda x: "dev" if stable_hash(x, config["seed"]) / 2**64 < config["development_fraction"] else "train", return_dtype=pl.String).alias("partition"))
    queries = pl.concat([allowed, query_table.filter(pl.col("split") == "test").with_columns(pl.lit("test").alias("partition"))])
    examples = examples.join(queries.select("query_id", "family", "partition"), on="query_id", how="inner", validate="m:1")
    examples = examples.with_columns(
        pl.struct("query_id", "product_id").map_elements(lambda x: stable_hash(x["query_id"] + "|" + x["product_id"], config["seed"]), return_dtype=pl.UInt64).alias("tie_order"))
    keys = examples.select("product_id", "product_locale").unique()
    products = (pl.scan_parquet(raw / "shopping_queries_dataset_products.parquet")
        .filter(pl.col("product_locale") == config["locale"])
        .join(keys.lazy(), on=["product_id", "product_locale"], how="semi")
        .select(pl.col("product_id").cast(pl.String), "product_locale",
                pl.col("product_title").fill_null("").str.slice(0, 512).alias("title"),
                pl.col("product_brand").fill_null("").str.slice(0, 128).alias("brand"),
                pl.col("product_description").fill_null("").str.slice(0, 1536).alias("description"),
                pl.col("product_bullet_point").fill_null("").str.slice(0, 1536).alias("attributes"))
        .collect().sort("product_locale", "product_id"))
    if products.height != products.select("product_locale", "product_id").unique().height:
        raise ValueError("Duplicate product key")
    if examples.join(products.select("product_id", "product_locale"), on=["product_id", "product_locale"], how="anti").height:
        raise ValueError("Missing product metadata")
    products.write_parquet(output / "products.parquet")
    queries.sort("partition", "query_id").write_parquet(output / "queries.parquet")
    for part in ("train", "dev", "test"):
        frame = examples.filter(pl.col("partition") == part).sort("query_id", "tie_order")
        frame.write_parquet(output / f"{part}.parquet")
    removed.sort("query_id").write_parquet(output / "excluded_overlap_queries.parquet")
    train_ids = set(examples.filter(pl.col("partition") == "train")["product_id"])
    test_ids = set(examples.filter(pl.col("partition") == "test")["product_id"])
    manifest = {"source_revision": ESCI_REVISION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": input_hashes, "config": config,
        "normalization_source_sha256": sha256(Path(__file__).with_name("text.py")),
        "rows": examples.height, "products": products.height, "queries": queries.height,
        "excluded_official_train_queries": removed.height, "excluded_families": removed["family"].n_unique(),
        "partition_counts": examples.group_by("partition").agg(pl.len().alias("pairs"), pl.col("query_id").n_unique().alias("queries"), pl.col("family").n_unique().alias("families")).sort("partition").to_dicts(),
        "test_product_count": len(test_ids), "test_products_seen_in_train": len(test_ids & train_ids),
        "prepared_sha256": {p.name: sha256(p) for p in sorted(output.glob("*.parquet"))},
        "text_limits": {"title": 512, "brand": 128, "description": 1536, "attributes": 1536},
        "note": "Split audit only; no model performance evaluated. All released test queries retained."}
    save_json(output / "manifest.json", manifest)
    return manifest


def load_pairs(directory, partition):
    directory = Path(directory)
    return pl.read_parquet(directory / f"{partition}.parquet").join(
        pl.read_parquet(directory / "products.parquet"), on=["product_id", "product_locale"], how="left", validate="m:1")
