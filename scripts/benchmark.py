"""Sequential local-engine timings for six declared demo queries; no load claims."""

import argparse
import platform
import random
import resource
import time
from pathlib import Path

import numpy as np

from catalog_search.data import save_json
from catalog_search.engine import SearchEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/latency.json"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    start = time.perf_counter()
    engine = SearchEngine(args.artifacts)
    loading_ms = (time.perf_counter() - start) * 1000
    queries = engine.meta()["examples"]
    modes = ("bm25", "baseline", "attribute")
    for query in queries:
        for mode in modes:
            engine.search(query, mode=mode, limit=12)
    schedule = [(query, mode) for _ in range(args.repeats) for query in queries for mode in modes]
    random.Random(20261026).shuffle(schedule)
    rows = []
    for query, mode in schedule:
        start = time.perf_counter()
        result = engine.search(query, mode=mode, limit=12)
        rows.append({"query": query, "mode": mode, "milliseconds": (time.perf_counter()-start)*1000,
                     "results": len(result["results"]), "candidates": result["total_candidates"]})
    metrics = {}
    for mode in modes:
        values = [row["milliseconds"] for row in rows if row["mode"] == mode]
        metrics[mode] = {"calls": len(values), "p50_ms": float(np.median(values)),
                         "p95_ms": float(np.quantile(values, .95)), "max_ms": max(values)}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024**2 if platform.system() == "Darwin" else 1024)
    report = {"scope": "Sequential in-process, warm process and shared feature cache; six TRAIN demo queries. Excludes HTTP/browser; not a traffic sample or concurrent load test.",
              "platform": platform.platform(), "machine": platform.machine(), "catalog_size": engine.meta()["catalog_size"],
              "model_version": engine.meta()["model_version"], "model_threads": 1,
              "query_count": len(queries), "repeats": args.repeats, "seed": 20261026,
              "load_ms": loading_ms, "process_peak_rss_mib": peak_mib,
              "memory_scope": "Whole benchmark process including Python, imported libraries, catalog, index and models; not model-only RAM",
              "methods": metrics, "calls": rows}
    save_json(args.output, report)
    print(metrics)


if __name__ == "__main__":
    main()
