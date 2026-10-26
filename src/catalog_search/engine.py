"""Saved-model catalogue retrieval and reranking, with no training at request time.

Artifact contract: encoder.joblib, baseline.txt, attribute.txt, catalog.parquet,
retrieval.joblib and index_manifest.json in one directory. ``build_index`` writes
the last three files from an already fitted training-side FeatureEncoder. It does
not fit the encoder, consult relevance labels, or modify the saved rankers.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import time
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from scipy import sparse

from .attributes import compare_attributes, extract_attributes
from .data import ESCI_REVISION, save_json, sha256
from .features import ALL_NAMES, LEXICAL_NAMES, FeatureEncoder
from .text import clean, document, stable_hash


SCHEMA_VERSION = 1
SEED = 20261026
RETRIEVAL_LIMIT = 150
MAX_RESULTS = 40
PRODUCT_COLUMNS = ("product_id", "title", "brand", "description", "attributes")
ALIASES = {
    "id": "product_id", "product_title": "title", "product_brand": "brand",
    "product_description": "description", "product_bullet_point": "attributes",
}


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _feature_source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {name: sha256(package / name) for name in ("text.py", "features.py", "attributes.py")}


def _encoder_signature(encoder: FeatureEncoder) -> str:
    """Fingerprint the fitted transforms used for both indexing and inference."""
    digest = hashlib.sha256()
    for vectorizer in (encoder.word, encoder.char):
        settings = {}
        for key, value in vectorizer.get_params(deep=False).items():
            if callable(value):
                value = value.__module__ + "." + value.__qualname__
            settings[key] = value
        digest.update(_digest_json(settings).encode())
        digest.update(_digest_json({str(token): int(index) for token, index in vectorizer.vocabulary_.items()}).encode())
        digest.update(np.asarray(vectorizer.idf_, dtype="<f8").tobytes())
    digest.update(np.asarray(encoder.bm25_idf, dtype="<f8").tobytes())
    digest.update(np.asarray([encoder.avg_title, encoder.avg_body], dtype="<f8").tobytes())
    # transform() also reads this lookup for IDF-weighted token coverage. A
    # matching sparse index alone cannot detect a different coverage transform.
    digest.update(_digest_json({str(token): float(value)
                               for token, value in encoder.idf_lookup.items()}).encode())
    digest.update(_digest_json(list(ALL_NAMES)).encode())
    return digest.hexdigest()


def _products(frame: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("products must be a Polars DataFrame")
    renames = {source: target for source, target in ALIASES.items() if source in frame.columns and target not in frame.columns}
    frame = frame.rename(renames)
    if "product_id" not in frame.columns or "title" not in frame.columns:
        raise ValueError("Products require product_id and title")
    if frame["product_id"].null_count():
        raise ValueError("Product identifiers cannot be null")
    for name in PRODUCT_COLUMNS:
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit("").alias(name))
    optional = [name for name in ("product_locale", "category") if name in frame.columns]
    frame = frame.select(*PRODUCT_COLUMNS, *optional).with_columns(
        pl.col(name).cast(pl.String).fill_null("") for name in (*PRODUCT_COLUMNS, *optional)
    )
    if frame["product_id"].n_unique() != frame.height:
        raise ValueError("Product IDs must be unique; restrict to one source locale before indexing")
    if frame.height == 0 or any(not value for value in frame["product_id"]):
        raise ValueError("The catalogue must contain nonempty product IDs")
    return frame


def build_index(
    encoder: FeatureEncoder,
    products: pl.DataFrame,
    output: Path | str,
    size: int = 30000,
    seed: int = SEED,
    include_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Save a label-blind deterministic catalogue and its fitted BM25 index.

    ``include_ids`` can retain candidates of preselected TRAIN queries for usable
    examples. They count toward ``size``; the remaining products are selected by
    seeded ID hash. Input row order and relevance labels never affect selection.
    """
    if not isinstance(size, numbers.Integral) or isinstance(size, bool) or size < 1:
        raise ValueError("size must be a positive integer")
    frame = _products(products)
    required = {str(value) for value in include_ids} if include_ids is not None else set()
    known = set(frame["product_id"])
    if required - known:
        raise ValueError("Some required example product IDs are absent from the source catalogue")
    selected_size = min(int(size), frame.height)
    if len(required) > selected_size:
        raise ValueError("include_ids exceeds the catalogue size budget")
    ordered = frame.with_columns(
        pl.col("product_id").map_elements(lambda value: stable_hash(value, seed), return_dtype=pl.UInt64).alias("_selection_hash")
    ).sort("_selection_hash", "product_id")
    retained = ordered.filter(pl.col("product_id").is_in(required))
    sampled = ordered.filter(~pl.col("product_id").is_in(required)).head(selected_size - len(required))
    catalogue = pl.concat([retained, sampled]).sort("_selection_hash", "product_id").drop("_selection_hash")
    product_ids = catalogue["product_id"].to_list()
    signature = _encoder_signature(encoder)
    matrix = encoder.retrieval_index(catalogue).tocsr()
    if matrix.shape != (catalogue.height, len(encoder.word.vocabulary_)) or not np.isfinite(matrix.data).all():
        raise ValueError("The retrieval matrix does not match the fitted encoder")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    catalogue.write_parquet(output / "catalog.parquet")
    joblib.dump({"schema_version": SCHEMA_VERSION, "matrix": matrix, "product_ids": product_ids, "encoder_signature": signature}, output / "retrieval.joblib", compress=3)
    manifest = {
        "schema_version": SCHEMA_VERSION, "source_product_count": frame.height,
        "catalog_size": catalogue.height, "requested_size": int(size), "seed": int(seed),
        "required_product_count": len(required), "required_ids_sha256": _digest_json(sorted(required)),
        "selection": "required TRAIN-example IDs, then seeded product-ID hash; fixed hash order",
        "product_ids_sha256": _digest_json(product_ids), "encoder_signature": signature,
        "feature_source_sha256": _feature_source_hashes(),
        "source_revision": ESCI_REVISION,
        "files": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size} for name in ("catalog.parquet", "retrieval.joblib")},
        "retrieval_limit": RETRIEVAL_LIMIT,
        "scope": "Catalogue demonstration. Judged-pool model evaluation does not measure full-catalogue retrieval recall.",
    }
    save_json(output / "index_manifest.json", manifest)
    return manifest


def _optional_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def _evidence(query: str, product: Mapping[str, Any]) -> list[dict[str, Any]]:
    title, brand, rest = document(product["title"], product["brand"], product["description"], product["attributes"])
    parsed = compare_attributes(query, title, " ".join((title, brand, rest)))
    source_texts = {"title": title, "description": " ".join((title, brand, rest))}
    rows = []
    for item in parsed["evidence"]:
        requirement = item["query"]
        specifications = item.get("product") or []
        # The frozen parser preserves every candidate source span. Multiple
        # records are meaningful (e.g. variants), so never discard all but one.
        if isinstance(specifications, dict):
            specifications = [specifications]
        specifications = [{**record, "source_text": source_texts[record["source"]],
                           "source_scope": "normalized title" if record["source"] == "title" else "composed title, brand, bullet points and description"}
                          for record in specifications]
        values = list(dict.fromkeys(record["text"] for record in specifications))
        sources = list(dict.fromkeys(record.get("source") for record in specifications))
        rows.append({
            "attribute": item["kind"], "query_value": requirement["text"],
            "product_value": values[0] if len(values) == 1 else values or None,
            "status": item["status"], "reason": item["reason"],
            "query_span": requirement["span"],
            "product_spans": [record["span"] for record in specifications],
            "product_sources": sources,
            "product_records": specifications,
            "ambiguous": bool(requirement.get("ambiguous") or any(record.get("ambiguous") for record in specifications)),
        })
    return rows


def _public_product(product: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "id": str(product["product_id"]), "title": clean(product["title"]),
        "brand": clean(product["brand"]), "description": clean(product["description"]),
        "bullet_points": clean(product["attributes"]),
    }
    if product.get("category"):
        value["category"] = clean(product["category"])
    return value


class SearchEngine:
    def __init__(self, artifact_dir: Path | str):
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        required_files = ("encoder.joblib", "baseline.txt", "attribute.txt", "catalog.parquet", "retrieval.joblib", "index_manifest.json")
        for name in required_files:
            if not (self.artifact_dir / name).is_file():
                raise FileNotFoundError(f"Missing search artifact: {name}")
        self.manifest = json.loads((self.artifact_dir / "index_manifest.json").read_text())
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported catalogue artifact version")
        if self.manifest.get("feature_source_sha256") != _feature_source_hashes():
            raise ValueError("Feature implementation differs from the saved catalogue bundle")
        for name in ("catalog.parquet", "retrieval.joblib"):
            expected = self.manifest.get("files", {}).get(name, {}).get("sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"Missing search artifact checksum: {name}")
            if sha256(self.artifact_dir / name) != expected:
                raise ValueError(f"Search artifact checksum mismatch: {name}")
        self.encoder = joblib.load(self.artifact_dir / "encoder.joblib")
        self.catalogue = _products(pl.read_parquet(self.artifact_dir / "catalog.parquet"))
        saved_index = joblib.load(self.artifact_dir / "retrieval.joblib")
        if not isinstance(saved_index, dict) or saved_index.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported retrieval artifact")
        self.index = saved_index["matrix"]
        product_ids = self.catalogue["product_id"].to_list()
        if saved_index.get("product_ids") != product_ids or _digest_json(product_ids) != self.manifest["product_ids_sha256"]:
            raise ValueError("Retrieval rows and catalogue IDs are not aligned")
        signature = _encoder_signature(self.encoder)
        if signature != saved_index.get("encoder_signature") or signature != self.manifest.get("encoder_signature"):
            raise ValueError("Saved retrieval index and encoder do not match")
        if not sparse.issparse(self.index) or self.index.shape != (self.catalogue.height, len(self.encoder.word.vocabulary_)) or not np.isfinite(self.index.data).all():
            raise ValueError("Invalid retrieval matrix")
        self.index = self.index.tocsr()
        self.models = {mode: lgb.Booster(model_file=str(self.artifact_dir / f"{mode}.txt")) for mode in ("baseline", "attribute")}
        for mode, feature_names in (("baseline", LEXICAL_NAMES), ("attribute", ALL_NAMES)):
            model = self.models[mode]
            if model.num_feature() != len(feature_names):
                raise ValueError(f"Saved {mode} model has the wrong feature count")
            actual = model.feature_name()
            if actual != list(feature_names) and actual != [f"Column_{index}" for index in range(len(feature_names))]:
                raise ValueError(f"Saved {mode} model has a different feature order")
        self._rows = self.catalogue.to_dicts()
        self._by_id = {row["product_id"]: row for row in self._rows}
        self.seed = int(self.manifest.get("seed", SEED))
        model_hashes = {name: sha256(self.artifact_dir / name) for name in ("encoder.joblib", "baseline.txt", "attribute.txt")}
        self.model_version = "catalog-v1-" + _digest_json({**model_hashes, "catalog": self.manifest["product_ids_sha256"]})[:12]

    def _report(self) -> dict[str, Any]:
        candidates = (self.artifact_dir / "reports" / "summary.json", self.artifact_dir.parent / "reports" / "summary.json")
        for path in candidates:
            value = _optional_json(path)
            if value:
                return value
        return {}

    def meta(self) -> dict[str, Any]:
        supplied = _optional_json(self.artifact_dir / "metadata.json")
        report = self._report()
        metrics = report.get("ui_metrics", [])
        if not metrics:
            metrics = _ndcg_metrics(report.get("esci", {}), "ESCI")
        selection = _optional_json(self.artifact_dir / "selection.json")
        examples = supplied.get("examples", selection.get("examples", []))
        examples = [item.strip() for item in examples if isinstance(item, str) and item.strip() and len(item.strip()) <= 256] if isinstance(examples, list) else []
        result = {
            "catalog_size": self.catalogue.height, "model_version": supplied.get("model_version", self.model_version),
            "metrics": metrics, "examples": examples[:6], "languages": ["en"],
            "dataset_name": "Amazon ESCI · English catalogue",
            "source_url": f"https://github.com/amazon-science/esci-data/tree/{self.manifest.get('source_revision', ESCI_REVISION)}",
            "source_revision": self.manifest.get("source_revision", ESCI_REVISION),
            "catalog_sha256": self.manifest["files"]["catalog.parquet"]["sha256"],
            "retrieval_limit": RETRIEVAL_LIMIT, "max_results": MAX_RESULTS,
            "modes": ["attribute", "baseline", "bm25"],
            "description": "Каталог на английском из открытого набора Amazon ESCI. Демо сначала выбирает BM25-кандидатов, затем ранжирует их. Изображения, цены и отзывы не добавлены.",
            "evaluation_note": "Метрики относятся к размеченным спискам кандидатов. Полнота поиска по всему каталогу и поведение покупателей не измерены.",
            "evaluation_available": bool(report),
        }
        if isinstance(report.get("splits"), list):
            result["split_counts"] = report["splits"]
            for partition in report["splits"]:
                if partition.get("partition") == "test":
                    result["evaluation_query_count"] = partition.get("queries")
        if isinstance(report.get("dataset"), dict) and "queries" in report["dataset"]:
            result["query_count"] = report["dataset"]["queries"]
        for key in ("query_count", "training_query_count", "evaluation_query_count"):
            if key in supplied:
                result[key] = supplied[key]
        if report:
            result["report_url"] = "/experiment.html"
        return result

    def product(self, product_id: str) -> dict[str, Any] | None:
        row = self._by_id.get(str(product_id))
        return _public_product(row) if row is not None else None

    def search(self, query: str, mode: str = "attribute", limit: int = 12) -> dict[str, Any]:
        started = time.perf_counter()
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 256:
            raise ValueError("query must contain 1–256 nonempty characters")
        query = query.strip()
        if any(unicodedata.category(character) in {"Cc", "Cs"} for character in query):
            raise ValueError("query contains a control character")
        if mode not in {"attribute", "baseline", "bm25"}:
            raise ValueError("Unknown search mode")
        if not isinstance(limit, numbers.Integral) or isinstance(limit, bool) or not 1 <= limit <= MAX_RESULTS:
            raise ValueError("limit must be between 1 and 40")
        # Retrieve all positive lexical matches first so ties at the top-150
        # boundary follow the same declared hash order as the final reranking.
        lexical_indices, bm25_scores = self.encoder.retrieve(query, self.index, limit=self.catalogue.height)
        tie_keys = np.asarray([stable_hash(query + "|" + self._rows[int(index)]["product_id"], self.seed) for index in lexical_indices], dtype=np.uint64)
        retrieval_order = np.lexsort((tie_keys, -bm25_scores[lexical_indices]))[:RETRIEVAL_LIMIT]
        candidates = np.asarray(lexical_indices[retrieval_order], dtype=np.int64)
        candidate_ties = tie_keys[retrieval_order]
        selected_rows = [self._rows[int(index)] for index in candidates]
        scores = np.asarray(bm25_scores[candidates], dtype=np.float64)
        if selected_rows and mode != "bm25":
            pairs = self.catalogue[candidates.tolist()].with_columns(pl.lit("interactive").alias("query_id"), pl.lit(query).alias("query"))
            features = self.encoder.transform(pairs)
            if mode == "baseline":
                features = features[:, :len(LEXICAL_NAMES)]
            scores = np.asarray(self.models[mode].predict(features, num_threads=1), dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("The ranker returned a non-finite score")
        order = np.lexsort((candidate_ties, -scores))[:int(limit)]
        results = []
        for rank, candidate_position in enumerate(order, 1):
            row = selected_rows[int(candidate_position)]
            value = _public_product(row)
            if not value["description"]:
                value["description"] = value["bullet_points"]
            value.update({"score": float(scores[candidate_position]), "rank": rank, "evidence": _evidence(query, row)})
            results.append(value)
        return {
            "query": query, "mode": mode, "total_candidates": len(selected_rows),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3), "results": results,
            "diagnostics": {"catalog_size": self.catalogue.height, "positive_lexical_matches": len(lexical_indices),
                "candidate_limit": RETRIEVAL_LIMIT, "query_attributes": extract_attributes(query),
                "feature_count": 0 if mode == "bm25" else len(LEXICAL_NAMES) if mode == "baseline" else len(ALL_NAMES),
                "tie_break": "seeded hash of query and product ID", "score_is_probability": False,
                "scope": "retrieved catalogue candidates; full-catalogue relevance is not evaluated"},
        }


def _ndcg_metrics(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Only publish named measured NDCG values, never counts/hyperparameters as scores."""
    if not isinstance(value, dict):
        return []
    result = []
    aliases = ("ndcg_at_10", "ndcg@10", "ndcg10", "mean_ndcg_at_10", "mean_ndcg10")
    preferred = next((key for key in aliases if isinstance(value.get(key), (int, float)) and math.isfinite(value[key])), None)
    if preferred is not None:
        result.append({"label": prefix + " · NDCG@10", "value": value[preferred]})
    for key, child in value.items():
        path = " · ".join(part for part in (prefix, str(key)) if part)
        if isinstance(child, dict):
            result.extend(_ndcg_metrics(child, path))
    return result
