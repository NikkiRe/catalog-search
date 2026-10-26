from __future__ import annotations

import json
import hashlib
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


def independent_candidates(artifact_dir, query):
    """Rebuild retrieval scores without SearchEngine or FeatureEncoder.retrieve."""
    encoder = joblib.load(artifact_dir / "encoder.joblib")
    catalogue = pl.read_parquet(artifact_dir / "catalog.parquet")
    index = joblib.load(artifact_dir / "retrieval.joblib")["matrix"]
    vector = encoder.word.transform([query])
    vector.data.fill(1)
    scores = np.asarray((index @ vector.T).toarray()).ravel()
    positive = np.flatnonzero(scores > 0)
    seed = json.loads((artifact_dir / "index_manifest.json").read_text())["seed"]
    ids = catalogue["product_id"].to_list()
    ties = np.array([
        int.from_bytes(hashlib.sha256(f"{seed}|{query}|{ids[index]}".encode()).digest()[:8], "big")
        for index in positive
    ], dtype=np.uint64)
    order = np.lexsort((ties, -scores[positive]))[:150]
    return encoder, catalogue, positive[order], scores[positive[order]], ties[order]


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


def test_predictive_idf_lookup_skew_is_rejected(artifacts):
    encoder = joblib.load(artifacts / "encoder.joblib")
    # This mapping affects title/body IDF coverage even when vocabulary,
    # vectorizer IDFs and both BM25 matrices remain byte-for-byte unchanged.
    encoder.idf_lookup = {token: value + 1.0 for token, value in encoder.idf_lookup.items()}
    joblib.dump(encoder, artifacts / "encoder.joblib")
    with pytest.raises(ValueError, match="encoder"):
        SearchEngine(artifacts)


@pytest.mark.parametrize("name", ["catalog.parquet", "retrieval.joblib"])
@pytest.mark.parametrize("missing", ["entry", "sha256"])
def test_artifact_checksums_are_required_not_optional(artifacts, name, missing):
    path = artifacts / "index_manifest.json"
    manifest = json.loads(path.read_text())
    if missing == "entry":
        del manifest["files"][name]
    else:
        del manifest["files"][name]["sha256"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        SearchEngine(artifacts)


def test_frozen_feature_implementation_hashes_are_checked(artifacts):
    path = artifacts / "index_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["feature_source_sha256"]["attributes.py"] = "different-implementation"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="implementation"):
        SearchEngine(artifacts)


@pytest.mark.parametrize("change", ["feature_count", "feature_order"])
def test_incompatible_saved_booster_is_rejected(artifacts, change):
    if change == "feature_count":
        shutil.copyfile(artifacts / "attribute.txt", artifacts / "baseline.txt")
    else:
        path = artifacts / "baseline.txt"
        text = path.read_text()
        original = "feature_names=" + " ".join(LEXICAL_NAMES)
        reordered = list(LEXICAL_NAMES)
        reordered[:2] = reversed(reordered[:2])
        assert original in text
        path.write_text(text.replace(original, "feature_names=" + " ".join(reordered), 1))
    with pytest.raises(ValueError, match="feature count|feature order"):
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


def test_each_mode_matches_independent_saved_booster_predictions(artifacts):
    engine = SearchEngine(artifacts)
    for query in ("65 W charger", "iphone 12 case", "desk shelf 24 inch"):
        encoder, catalogue, candidates, bm25, ties = independent_candidates(artifacts, query)
        pairs = catalogue[candidates.tolist()].with_columns(
            pl.lit("independent-oracle").alias("query_id"), pl.lit(query).alias("query")
        )
        matrix = encoder.transform(pairs)
        ids = catalogue["product_id"].to_list()
        for mode, width in (("bm25", 0), ("baseline", len(LEXICAL_NAMES)), ("attribute", len(ALL_NAMES))):
            # A separately loaded Booster makes a hard-coded ordering or the
            # accidental use of the other mode's model fail this assertion.
            scores = bm25 if not width else lgb.Booster(
                model_file=str(artifacts / f"{mode}.txt")
            ).predict(matrix[:, :width], num_threads=1)
            order = np.lexsort((ties, -scores))
            response = engine.search(query, mode, limit=8)
            assert response["total_candidates"] == len(candidates)
            assert [row["id"] for row in response["results"]] == [ids[candidates[index]] for index in order]
            np.testing.assert_allclose([row["score"] for row in response["results"]], scores[order], rtol=0, atol=1e-12)


def test_model_modes_execute_their_own_predict_on_query_dependent_features(artifacts, monkeypatch):
    engine = SearchEngine(artifacts)
    calls = []
    for mode in ("baseline", "attribute"):
        original = engine.models[mode].predict

        def tracked(matrix, *, _original=original, _mode=mode, **kwargs):
            calls.append((_mode, matrix.copy()))
            return _original(matrix, **kwargs)

        monkeypatch.setattr(engine.models[mode], "predict", tracked)
    engine.search("USB charger", "baseline")
    engine.search("USB charger", "attribute")
    engine.search("USB charger", "bm25")
    engine.search("USB 65 W charger", "attribute")
    engine.search("USB charger", "baseline")
    assert [mode for mode, _ in calls] == ["baseline", "attribute", "attribute", "baseline"]
    assert [matrix.shape[1] for _, matrix in calls] == [28, 58, 58, 28]
    np.testing.assert_array_equal(calls[0][1], calls[1][1][:, :28])
    np.testing.assert_array_equal(calls[0][1], calls[3][1])
    assert not np.array_equal(calls[1][1], calls[2][1])


def test_same_top150_candidates_and_hash_ties_for_every_mode(artifacts, fitted_encoder, monkeypatch):
    # More than 150 exactly equal lexical scores exposes a cutoff tie error;
    # testing only a handful of distinct products cannot exercise that boundary.
    repeated = pl.DataFrame({
        "product_id": [f"tie-{index}" for index in range(220)],
        "title": ["USB charger 65W"] * 220,
        "brand": ["Anker"] * 220,
        "description": ["A compact USB charger for a desk"] * 220,
        "attributes": ["USB charger output power"] * 220,
    })
    build_index(fitted_encoder, repeated, artifacts, size=220)
    engine = SearchEngine(artifacts)
    _, catalogue, candidates, scores, ties = independent_candidates(artifacts, "charger")
    expected_candidates = catalogue[candidates.tolist()]["product_id"].to_list()
    assert len(candidates) == 150
    assert len(np.unique(scores)) == 1
    transformed_ids = []
    original = engine.encoder.transform

    def tracked(pairs):
        transformed_ids.append(pairs["product_id"].to_list())
        return original(pairs)

    monkeypatch.setattr(engine.encoder, "transform", tracked)
    for mode in ("baseline", "attribute", "bm25"):
        response = engine.search("charger", mode, limit=40)
        assert response["total_candidates"] == 150
        assert response["diagnostics"]["positive_lexical_matches"] == 220
        assert [row["id"] for row in response["results"]] == expected_candidates[:40]
        assert len({row["id"] for row in response["results"]}) == 40
    assert transformed_ids == [expected_candidates, expected_candidates]


@pytest.mark.parametrize("change", ["catalogue_order", "index_ids", "index_width", "index_nan"])
def test_semantic_index_corruption_is_rejected_even_with_fresh_file_checksum(artifacts, change):
    from catalog_search.data import sha256

    if change == "catalogue_order":
        name = "catalog.parquet"
        pl.read_parquet(artifacts / name).reverse().write_parquet(artifacts / name)
    else:
        name = "retrieval.joblib"
        saved = joblib.load(artifacts / name)
        if change == "index_ids":
            saved["product_ids"] = list(reversed(saved["product_ids"]))
        elif change == "index_width":
            saved["matrix"] = saved["matrix"][:, :-1]
        else:
            saved["matrix"].data[0] = np.nan
        joblib.dump(saved, artifacts / name)
    manifest_path = artifacts / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name]["sha256"] = sha256(artifacts / name)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="aligned|matrix"):
        SearchEngine(artifacts)


def test_duplicate_product_ids_are_rejected_before_index_write(tmp_path, products, fitted_encoder):
    duplicated = pl.concat([products, products.head(1)])
    with pytest.raises(ValueError, match="unique"):
        build_index(fitted_encoder, duplicated, tmp_path)
    assert not (tmp_path / "catalog.parquet").exists()


@pytest.mark.parametrize("query,limit", [("a\x00b", 12), ("a\u0085b", 12), ("a\ud800b", 12), ("charger", True), ("charger", 1.5), ("charger", -1)])
def test_direct_engine_rejects_controls_surrogates_and_noninteger_limits(artifacts, query, limit):
    with pytest.raises(ValueError):
        SearchEngine(artifacts).search(query, limit=limit)
