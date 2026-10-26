"""Exercise the actual release artifacts through the engine and HTTP application."""

import argparse
import json
import math
from pathlib import Path

from fastapi.testclient import TestClient

from catalog_search.api import create_app
from catalog_search.engine import SearchEngine


def verify(artifacts):
    engine = SearchEngine(artifacts)
    metadata = engine.meta()
    assert metadata["catalog_size"] == 30000
    assert metadata["evaluation_available"]
    assert len(metadata["examples"]) == 6
    checked = []
    for query in metadata["examples"]:
        candidate_sets = []
        for mode in ("bm25", "baseline", "attribute"):
            result = engine.search(query, mode=mode, limit=5)
            assert result["query"] == query and result["mode"] == mode
            assert len(result["results"]) == 5
            assert [row["rank"] for row in result["results"]] == list(range(1, 6))
            assert all(row["title"] and math.isfinite(row["score"]) for row in result["results"])
            assert result["diagnostics"]["score_is_probability"] is False
            candidate_sets.append(result["total_candidates"])
            checked.append({"query": query, "mode": mode, "latency_ms": result["latency_ms"]})
        assert len(set(candidate_sets)) == 1
    assert not engine.search("xqzvbnmunfindabletoken", "attribute", 5)["results"]
    with TestClient(create_app(artifact_dir=artifacts, engine=engine)) as client:
        for path in ("/", "/experiment.html", "/styles.css", "/app.js", "/report.js", "/health", "/api/meta", "/api/report"):
            assert client.get(path).status_code == 200, path
        response = client.get("/api/search", params={"q": metadata["examples"][0], "limit": 5})
        assert response.status_code == 200
        product_id = response.json()["results"][0]["id"]
        assert client.get(f"/api/product/{product_id}").status_code == 200
        assert client.get("/api/search", params={"q": " ", "limit": 5}).status_code == 400
        assert client.get("/api/search", params={"q": "a", "limit": 41}).status_code == 400
    return {"status": "passed", "model_version": metadata["model_version"],
            "catalog_size": metadata["catalog_size"], "checks": checked,
            "note": "Functional release smoke. Timings are not a controlled latency benchmark."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    print(json.dumps(verify(args.artifacts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
