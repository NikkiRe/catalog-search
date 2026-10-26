"""Recompute every ESCI query NDCG using SQL, without importing Python metrics.

SQLite is bundled with Python. Query logic lives in sql/*.sql: joins, conditional
aggregates and window ranks. This is an analytical audit, not a production DB.
"""

import argparse
import json
import math
from pathlib import Path
import sqlite3
import time

import polars as pl

from catalog_search.data import save_json, sha256

ROOT = Path(__file__).resolve().parents[1]
MODES = ("bm25", "tfidf", "baseline", "rule", "no_product_evidence", "attribute")


def load_scores(connection, frame):
    connection.execute("CREATE TABLE judged(query_id TEXT, product_id TEXT, family TEXT, label TEXT, tie_key TEXT, bm25 REAL, tfidf REAL, baseline REAL, rule REAL, no_product_evidence REAL, attribute REAL)")
    fields = ["query_id", "product_id", "family", "label", "tie_order", *MODES]
    rows = []
    for query_id, product_id, family, label, tie, *scores in frame.select(fields).iter_rows():
        if tie is None or not 0 <= int(tie) < 2**64:
            raise ValueError("Invalid unsigned tie key")
        if not all(value is not None and math.isfinite(value) for value in scores):
            raise ValueError("Non-finite ranking score")
        rows.append((str(query_id), str(product_id), family, label, f"{int(tie):016x}", *scores))
    connection.executemany("INSERT INTO judged VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
    connection.execute("CREATE INDEX judged_query ON judged(query_id)")


def check_contracts(connection, sql_dir, data_dir=None):
    result = dict(connection.execute((sql_dir / "contracts.sql").read_text()).fetchall())
    if data_dir is not None:
        queries = pl.read_parquet(data_dir / "queries.parquet", columns=["query_id", "family", "partition"])
        products = pl.read_parquet(data_dir / "products.parquet", columns=["product_locale", "product_id"])
        connection.execute("CREATE TABLE split_queries(query_id TEXT, family TEXT, partition TEXT)")
        connection.executemany("INSERT INTO split_queries VALUES(?,?,?)", queries.iter_rows())
        connection.execute("CREATE INDEX split_query ON split_queries(query_id)")
        connection.execute("CREATE TABLE products(product_locale TEXT, product_id TEXT)")
        connection.executemany("INSERT INTO products VALUES(?,?)", products.iter_rows())
        connection.execute("CREATE INDEX product_key ON products(product_id, product_locale)")
        result.update(connection.execute((sql_dir / "split_contracts.sql").read_text()).fetchall())
    bad = {name: count for name, count in result.items() if count}
    if bad:
        raise ValueError(f"SQL data contracts failed: {bad}")
    return result


def run_audit(report_dir, output, data_dir=None, sql_dir=None):
    start = time.perf_counter()
    report_dir, output = Path(report_dir), Path(output)
    sql_dir = Path(sql_dir) if sql_dir else ROOT / "sql"
    data_dir = Path(data_dir) if data_dir else None
    scores = pl.read_parquet(report_dir / "test_scores.parquet")
    expected = pl.read_parquet(report_dir / "per_query.parquet")
    summary = json.loads((report_dir / "summary.json").read_text())
    with sqlite3.connect(":memory:") as connection:
        try:
            connection.execute("SELECT log2(2)").fetchone()
            math_backend = "native SQLite log2"
        except sqlite3.OperationalError:
            connection.create_function("log2", 1, math.log2, deterministic=True)
            math_backend = "registered standard-library math.log2; all ranking/aggregation is SQL"
        load_scores(connection, scores)
        contracts = check_contracts(connection, sql_dir, data_dir)
        result = pl.DataFrame(connection.execute((sql_dir / "ndcg.sql").read_text()).fetchall(),
            schema=["query_id", "family", "method", "candidates", "dcg10", "idcg10", "ndcg10"], orient="row")
    expected_keys = set(expected["query_id"])
    comparisons = {}
    for mode in MODES:
        method = result.filter(pl.col("method") == mode)
        if set(method["query_id"]) != expected_keys or method["query_id"].n_unique() != method.height:
            raise ValueError("SQL query cohort differs from saved per-query results")
        comparison = method.join(expected.select("query_id", pl.col(mode + "_ndcg10").alias("reference")), on="query_id", validate="1:1")
        maximum = comparison.select((pl.col("ndcg10") - pl.col("reference")).abs().max()).item()
        mean = method["ndcg10"].mean()
        mean_difference = abs(mean - summary["esci"][mode]["ndcg10"])
        if maximum > 1e-12 or mean_difference > 1e-12:
            raise ValueError(f"SQL/Python disagreement for {mode}: {maximum}, {mean_difference}")
        comparisons[mode] = {"ndcg10": mean, "max_per_query_absolute_error": maximum,
                             "mean_absolute_error": mean_difference, "queries": method.height}
    output.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output / "sql_per_query.parquet")
    audit = {"status": "passed", "scope": "Independent SQL verification of all six ESCI ranking methods and optional split/foreign-key contracts; no training or model selection",
             "sqlite_version": sqlite3.sqlite_version, "math_backend": math_backend,
             "pairs": scores.height, "queries": len(expected_keys), "contracts": contracts,
             "models": comparisons, "tolerance": 1e-12, "seconds": time.perf_counter()-start,
             "inputs_sha256": {name: sha256(report_dir / name) for name in ("test_scores.parquet", "per_query.parquet", "summary.json")},
             "sql_sha256": {path.name: sha256(path) for path in sorted(sql_dir.glob("*.sql"))}}
    save_json(output / "sql_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--data", type=Path, help="Optional prepared ESCI directory for join/split checks")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    print(json.dumps(run_audit(args.reports, args.output, args.data), indent=2))


if __name__ == "__main__":
    main()
