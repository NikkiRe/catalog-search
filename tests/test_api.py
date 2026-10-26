from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from catalog_search import api


class FakeEngine:
    def __init__(self):
        self.calls = []

    def meta(self):
        return {"catalog_size": 3, "model_version": "test-version", "metrics": {"ndcg_at_10": 0.5}, "examples": ["65w charger"]}

    def search(self, query, mode="attribute", limit=12):
        self.calls.append((query, mode, limit))
        return {"query": query, "mode": mode, "total_candidates": 1, "latency_ms": 1.0, "results": [{"id": "p-1", "title": '<img src=x onerror="alert(1)">', "rank": 1, "score": 0.5, "evidence": []}], "diagnostics": {}}

    def product(self, product_id):
        return {"id": "p-1", "title": "A product", "description": "Catalog description"} if product_id == "p-1" else None


@pytest.fixture
def project(tmp_path, monkeypatch):
    source_web = Path(__file__).resolve().parents[1] / "web"
    web = tmp_path / "web"
    web.mkdir()
    for name in ("index.html", "styles.css", "app.js", "experiment.html", "report.js"):
        shutil.copyfile(source_web / name, web / name)
    (tmp_path / "private-model.bin").write_text("must not be served")
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def provider():
    return FakeEngine()


@pytest.fixture
def client(project, provider):
    with TestClient(api.create_app(project / "artifacts", engine=provider)) as value:
        yield value


def test_unicode_query_is_trimmed_and_mode_is_forwarded(client, provider):
    response = client.get("/api/search", params={"q": "  café テーブル — зарядка 🔌  ", "mode": "bm25", "limit": 7})
    assert response.status_code == 200
    assert provider.calls == [("café テーブル — зарядка 🔌", "bm25", 7)]
    assert response.json()["query"] == "café テーブル — зарядка 🔌"


@pytest.mark.parametrize("mode", ["attribute", "baseline", "bm25"])
def test_every_mode_is_used_without_substitution(client, provider, mode):
    assert client.get("/api/search", params={"q": "charger", "mode": mode}).status_code == 200
    assert provider.calls[-1] == ("charger", mode, 12)


@pytest.mark.parametrize("params", [
    {}, {"q": "   "}, {"q": "x" * 257}, {"q": "a\x00b"}, {"q": "a\nb"},
    {"q": "charger", "limit": 0}, {"q": "charger", "limit": 41}, {"q": "charger", "limit": "a"},
    {"q": "charger", "mode": "invented"},
])
def test_invalid_parameters_are_400_and_never_reach_engine(client, provider, params):
    response = client.get("/api/search", params=params)
    assert response.status_code == 400
    assert "detail" in response.json()
    assert provider.calls == []


def test_query_boundaries_are_applied_after_trimming(client):
    assert client.get("/api/search", params={"q": " " + "x" * 256 + " ", "limit": 40}).status_code == 200


def test_product_and_missing_product(client):
    assert client.get("/api/product/p-1").json()["description"] == "Catalog description"
    response = client.get("/api/product/missing")
    assert response.status_code == 404
    assert response.json() == {"detail": "Товар не найден в каталоге."}


def test_health_and_metadata_come_from_provider(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "test-version", "catalog_size": 3}
    assert client.get("/api/meta").json()["metrics"] == {"ndcg_at_10": 0.5}


def test_actual_frontend_assets_and_no_model_file_access(client):
    index = client.get("/")
    assert index.status_code == 200
    assert 'id="search-input"' in index.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200
    assert client.get("/experiment.html").status_code == 200
    assert client.get("/report.js").status_code == 200
    assert client.get("/private-model.bin").status_code == 404
    assert client.get("/%2e%2e/private-model.bin").status_code == 404
    assert client.get("/src/catalog_search/api.py").status_code == 404


def test_xss_payload_stays_json_and_responses_are_not_cached(client):
    response = client.get("/api/search", params={"q": "charger"})
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.json()["results"][0]["title"].startswith("<img")


def test_report_serves_existing_json_and_is_not_a_placeholder(project, provider):
    report_dir = project / "reports"
    report_dir.mkdir()
    payload = {"source": "measured", "test_queries": 17}
    (report_dir / "summary.json").write_text(json.dumps(payload))
    with TestClient(api.create_app(project / "artifacts", engine=provider)) as client:
        assert client.get("/api/report").json() == payload
        assert client.get("/api/meta").json()["report_url"] == "/api/report"


def test_missing_report_is_404(client):
    assert client.get("/api/report").status_code == 404
    assert "report_url" not in client.get("/api/meta").json()


def test_explicit_artifact_bundle_does_not_inherit_another_checkout_report(project, provider):
    report_dir = project / "reports"
    report_dir.mkdir()
    (report_dir / "summary.json").write_text(json.dumps({"source": "different-model-bundle"}))
    other_bundle = project / "other-release" / "artifacts"
    with TestClient(api.create_app(other_bundle, engine=provider)) as client:
        assert client.get("/api/report").status_code == 404
        assert "report_url" not in client.get("/api/meta").json()


def test_missing_model_is_503_but_frontend_still_loads(project, monkeypatch):
    calls = []

    def missing(path):
        calls.append(path)
        raise FileNotFoundError("/private/secret/path/model.bin")

    monkeypatch.setattr(api, "_load_engine", missing)
    with TestClient(api.create_app(project / "artifacts")) as client:
        for path in ("/health", "/api/meta", "/api/search?q=charger", "/api/product/p-1"):
            response = client.get(path)
            assert response.status_code == 503
            assert "secret" not in response.text
            assert "Traceback" not in response.text
        assert client.get("/").status_code == 200
    assert calls == [project / "artifacts"]


def test_engine_is_loaded_once_with_explicit_path_overriding_env(project, provider, monkeypatch):
    calls = []
    monkeypatch.setenv("CATALOG_SEARCH_ARTIFACT_DIR", str(project / "environment-artifacts"))

    def load(path):
        calls.append(path)
        return provider

    monkeypatch.setattr(api, "_load_engine", load)
    with TestClient(api.create_app(project / "chosen-artifacts")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/meta").status_code == 200
        assert client.get("/api/search?q=charger").status_code == 200
    assert calls == [project / "chosen-artifacts"]


def test_model_failure_does_not_silently_fall_back_or_expose_query(client, provider, monkeypatch, caplog):
    def failure(*args, **kwargs):
        raise RuntimeError("sensitive-query /private/model.bin")

    monkeypatch.setattr(provider, "search", failure)
    response = client.get("/api/search", params={"q": "sensitive-query", "mode": "baseline"})
    assert response.status_code == 500
    assert "sensitive-query" not in response.text
    assert "/private" not in response.text
    assert "sensitive-query" not in caplog.text
    assert "catalog_search_failed" in caplog.text


@pytest.mark.parametrize("query", ['<script>alert("charger")</script>', "65 W 🔌 зарядка テーブル", "a\u202eb charger"])
def test_unusual_printable_queries_remain_literal_json(client, provider, query):
    response = client.get("/api/search", params={"q": query, "mode": "attribute"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.json()["query"] == query
    assert provider.calls == [(query, "attribute", 12)]


@pytest.mark.parametrize("limit", ["1.5", "NaN", "inf", "true", "-1"])
def test_noninteger_or_out_of_range_limits_do_not_reach_engine(client, provider, limit):
    response = client.get("/api/search", params={"q": "charger", "limit": limit})
    assert response.status_code == 400
    assert provider.calls == []
    assert "Traceback" not in response.text


def test_repeated_queries_and_mode_changes_each_reach_provider(client, provider):
    requests = [("charger", "baseline", 1), ("charger", "attribute", 40), ("shelf", "bm25", 12), ("charger", "baseline", 1)]
    for query, mode, limit in requests:
        response = client.get("/api/search", params={"q": query, "mode": mode, "limit": limit})
        assert response.status_code == 200
        assert response.json()["mode"] == mode
    assert provider.calls == requests


def test_static_assets_and_report_do_not_follow_symlinks_outside_root(project, provider):
    secret = project / "private-model.bin"
    (project / "web" / "linked-model.bin").symlink_to(secret)
    report_dir = project / "reports"
    report_dir.mkdir()
    (report_dir / "summary.json").symlink_to(secret)
    with TestClient(api.create_app(project / "artifacts", engine=provider)) as client:
        assert client.get("/linked-model.bin").status_code == 404
        assert client.get("/api/report").status_code == 404
