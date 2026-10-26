"""Explicit, reproducible entry points for data, experiments and local search."""

import argparse
import json
from pathlib import Path


def _json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="catalog-search", description="Product search with lexical and typed-attribute ranking")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and verify the pinned ESCI release (~1.2 GB)")
    download.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Build checked, family-disjoint ESCI partitions")
    prepare.add_argument("--raw", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    train = commands.add_parser("train", help="Select on dev, lock models, evaluate once on ESCI and WANDS")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--wands", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("artifacts"))
    train.add_argument("--config", type=Path, required=True)
    index = commands.add_parser("index", help="Build the reproducible demonstration catalogue")
    index.add_argument("--data", type=Path, required=True)
    index.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    index.add_argument("--size", type=int, default=30000)
    search = commands.add_parser("search", help="Run a real search from the terminal")
    search.add_argument("query")
    search.add_argument("--mode", choices=["attribute", "baseline", "bm25"], default="attribute")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    serve = commands.add_parser("serve", help="Serve the API and browser interface")
    serve.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.command == "download":
        from .data import download
        _json(download(args.output))
    elif args.command == "prepare":
        from .data import prepare
        _json(prepare(args.raw, args.output, json.loads(args.config.read_text())))
    elif args.command == "train":
        from .experiment import run_experiment
        _json(run_experiment(args.data, args.wands, args.output, json.loads(args.config.read_text())))
    elif args.command == "index":
        import joblib
        import polars as pl
        from .engine import build_index
        encoder = joblib.load(args.artifacts / "encoder.joblib")
        _json(build_index(encoder, pl.read_parquet(args.data / "products.parquet"), args.artifacts, size=args.size))
    elif args.command == "search":
        from .engine import SearchEngine
        _json(SearchEngine(args.artifacts).search(args.query, args.mode, args.limit))
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(args.artifacts), host=args.host, port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
