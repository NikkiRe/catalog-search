from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from catalog_search.engine import SearchEngine, build_index
from catalog_search.features import ALL_NAMES, LEXICAL_NAMES, FeatureEncoder
from catalog_search import api


@pytest.fixture
def products():
    return pl.DataFrame({
        "product_id": [f"p-{index}" for index in range(8)],
        "title": ["USB charger 65W", "USB charger 30W", "USB charger 100W", "iPhone 12 case", "iPhone 13 case", "Desk shelf 24 inch", "Desk shelf 36 inch", "Desk shelf 48 inch"],
        "brand": ["Anker", "Anker", "Belkin", "", "", "", "", ""],
        "description": ["A compact USB charger for a desk"] * 3 + ["A protective phone case"] * 2 + ["A wooden desk shelf"] * 3,
        "attributes": ["USB charger output power"] * 3 + ["phone case cover"] * 2 + ["desk shelf width"] * 3,
    })


@pytest.fixture
def fitted_encoder(products):
    return FeatureEncoder(word_features=100, character_features=100).fit(products)


@pytest.fixture
def artifacts(tmp_path, products, fitted_encoder):
    # Tiny synthetic boosters exercise actual LightGBM save/load and prediction.
    # Their values have no interpretation as benchmark quality or project results.
    generator = np.random.default_rng(12)
    for mode, names in (("baseline", LEXICAL_NAMES), ("attribute", ALL_NAMES)):
        features = generator.normal(size=(32, len(names))).astype(np.float32)
        labels = features[:, 0] * 0.3 + features[:, -1] * 0.7
        model = lgb.train({"objective": "regression", "verbosity": -1, "num_threads": 1, "num_leaves": 3, "min_data_in_leaf": 2}, lgb.Dataset(features, label=labels, feature_name=list(names)), num_boost_round=3)
        model.save_model(tmp_path / f"{mode}.txt")
    joblib.dump(fitted_encoder, tmp_path / "encoder.joblib")
    build_index(fitted_encoder, products, tmp_path, size=8)
    return tmp_path


def test_index_selection_is_input_order_invariant(tmp_path, products, fitted_encoder):
    first = build_index(fitted_encoder, products, tmp_path / "first", size=4, include_ids=["p-6"])
    second = build_index(fitted_encoder, products.reverse(), tmp_path / "second", size=4, include_ids=["p-6"])
    assert first["product_ids_sha256"] == second["product_ids_sha256"]
    assert pl.read_parquet(tmp_path / "first/catalog.parquet").equals(pl.read_parquet(tmp_path / "second/catalog.parquet"))
    assert "p-6" in pl.read_parquet(tmp_path / "first/catalog.parquet")["product_id"]
    assert first["required_product_count"] == 1
    assert fitted_encoder.n_fit_documents == products.height


def test_required_products_must_exist_and_fit_budget(tmp_path, products, fitted_encoder):
    with pytest.raises(ValueError, match="absent"):
        build_index(fitted_encoder, products, tmp_path, include_ids=["missing"])
    with pytest.raises(ValueError, match="budget"):
        build_index(fitted_encoder, products, tmp_path, size=1, include_ids=["p-1", "p-2"])


@pytest.mark.parametrize("mode,features", [("bm25", 0), ("baseline", len(LEXICAL_NAMES)), ("attribute", len(ALL_NAMES))])
def test_real_engine_load_retrieve_rerank_and_explain(artifacts, mode, features):
    engine = SearchEngine(artifacts)
    result = engine.search("USB charger exactly 65W", mode=mode, limit=3)
    assert result["mode"] == mode
    assert result["total_candidates"] >= 3
    assert len(result["results"]) == 3
    assert result["diagnostics"]["feature_count"] == features
    assert result["latency_ms"] >= 0
    assert [item["rank"] for item in result["results"]] == [1, 2, 3]
    scores = [item["score"] for item in result["results"]]
    assert scores == sorted(scores, reverse=True)
    assert all(item["evidence"] for item in result["results"])
    exact = next(item for item in result["results"] if item["id"] == "p-0")
    assert any(item["attribute"] == "power" and item["status"] == "match" and item["product_value"] == "65W" for item in exact["evidence"])
    assert engine.product(exact["id"])["title"] == exact["title"]
    assert engine.product("unknown") is None


def test_out_of_vocabulary_query_returns_empty_without_fallback(artifacts):
    response = SearchEngine(artifacts).search("zzzzunseenword", "attribute")
    assert response["results"] == []
    assert response["total_candidates"] == 0


@pytest.mark.parametrize("query", ["iphone 12 case", "65 W charger"])
def test_frozen_parser_product_record_lists_are_preserved(artifacts, query):
    response = SearchEngine(artifacts).search(query, "attribute", limit=8)
    assert response["results"]
    evidence = [item for product in response["results"] for item in product["evidence"]]
    assert evidence
    assert all(isinstance(item["product_records"], list) for item in evidence)
    assert all(isinstance(item["product_spans"], list) for item in evidence)
    assert all(record["source"] in {"title", "description"} for item in evidence for record in item["product_records"])
    assert all(record["source_text"][slice(*record["span"])] == record["text"] for item in evidence for record in item["product_records"])
    if query == "65 W charger":
        matched = [item for item in evidence if item["status"] == "match"]
        assert matched
        assert any(item["product_value"] == "65W" for item in matched)
    else:
        assert any(item["attribute"] == "model" and item["status"] == "match" and str(item["product_value"]).lower() == "iphone 12" for item in evidence)


def test_results_are_deterministic_across_loads(artifacts):
    first = SearchEngine(artifacts).search("charger", "attribute")
    second = SearchEngine(artifacts).search("charger", "attribute")
    assert first["results"] == second["results"]
    assert first["diagnostics"] == second["diagnostics"]


def test_index_encoder_mismatch_is_rejected(artifacts, products):
    other = FeatureEncoder(word_features=70, character_features=70).fit(products)
    joblib.dump(other, artifacts / "encoder.joblib")
    with pytest.raises(ValueError, match="encoder"):
        SearchEngine(artifacts)


def test_frozen_feature_implementation_hashes_are_checked(artifacts):
    path = artifacts / "index_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["feature_source_sha256"]["attributes.py"] = "different-implementation"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="implementation"):
        SearchEngine(artifacts)


def test_real_engine_through_http_and_packaged_web_fallback(artifacts, tmp_path, monkeypatch):
    package = tmp_path / "installed" / "catalog_search"
    web = package / "web"
    web.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "web"
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copyfile(source / name, web / name)
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path / "no-source-checkout")
    monkeypatch.setattr(api, "__file__", str(package / "api.py"))
    with TestClient(api.create_app(artifacts, engine=SearchEngine(artifacts))) as client:
        assert client.get("/").status_code == 200
        assert client.get("/app.js").status_code == 200
        response = client.get("/api/search", params={"q": "65 W charger", "mode": "attribute"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["results"]
        assert any(item["evidence"] for item in payload["results"])
        assert client.get("/api/product/" + payload["results"][0]["id"]).status_code == 200


def test_real_metadata_has_no_fabricated_metrics_or_examples(artifacts):
    engine = SearchEngine(artifacts)
    metadata = engine.meta()
    assert metadata["catalog_size"] == 8
    assert metadata["metrics"] == []
    assert metadata["examples"] == []
    assert metadata["evaluation_available"] is False
    (artifacts / "metadata.json").write_text(json.dumps({"examples": ["USB charger"], "training_query_count": 7}))
    (artifacts / "reports").mkdir()
    (artifacts / "reports/summary.json").write_text(json.dumps({"esci": {"baseline": {"ndcg_at_10": 0.7}, "attribute": {"ndcg_at_10": 0.71}}}))
    measured = engine.meta()
    assert measured["examples"] == ["USB charger"]
    assert [item["value"] for item in measured["metrics"]] == [0.7, 0.71]
    assert measured["report_url"] == "/experiment.html"


@pytest.mark.parametrize("query,mode,limit", [("", "attribute", 12), ("charger", "invented", 12), ("charger", "attribute", 41), ("x" * 257, "bm25", 12)])
def test_engine_rejects_invalid_direct_calls(artifacts, query, mode, limit):
    with pytest.raises(ValueError):
        SearchEngine(artifacts).search(query, mode, limit)
