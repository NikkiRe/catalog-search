"""Local HTTP interface for the trained catalog search engine.

Run with ``uvicorn catalog_search.api:create_app --factory --no-access-log``.
The application reads one engine per factory invocation and never writes queries.
"""

from __future__ import annotations

import logging
import os
import unicodedata
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
SearchMode = Literal["attribute", "baseline", "bm25"]


def _load_engine(artifact_dir: Path) -> Any:
    # Delay importing the model stack so static/error pages remain available when
    # optional model dependencies or trained artifacts have not been prepared.
    from catalog_search.engine import SearchEngine

    return SearchEngine(artifact_dir)


def _query(value: str) -> str:
    value = value.strip()
    if not value:
        raise HTTPException(400, "Введите непустой поисковый запрос.")
    if len(value) > 256:
        raise HTTPException(400, "Поисковый запрос должен содержать не больше 256 символов.")
    if any(unicodedata.category(character) in {"Cc", "Cs"} for character in value):
        raise HTTPException(400, "Используйте запрос из одной строки без управляющих символов.")
    return value


def _json_call(function: Callable[[], Any], *, operation: str) -> JSONResponse:
    """Expose JSON and short errors, never exception text containing paths/queries."""
    try:
        value = function()
        if not isinstance(value, Mapping):
            raise TypeError("Engine response must be a mapping")
        return JSONResponse(jsonable_encoder(dict(value)))
    except HTTPException:
        raise
    except (FileNotFoundError, ImportError, OSError):
        LOGGER.error("catalog_%s_unavailable", operation)
        raise HTTPException(503, "Поисковая модель недоступна. Проверьте подготовку артефактов и перезапустите сервис.") from None
    except ValueError:
        if operation in {"search", "product"}:
            raise HTTPException(400, "Не удалось обработать запрос. Проверьте его параметры.") from None
        LOGGER.error("catalog_%s_failed", operation)
        raise HTTPException(503, "Сведения о поисковой модели временно недоступны.") from None
    except Exception:
        LOGGER.error("catalog_%s_failed", operation)
        status = 503 if operation in {"meta", "health"} else 500
        raise HTTPException(status, "Не удалось выполнить запрос. Повторите попытку позже.") from None


def create_app(artifact_dir: Path | str | None = None, engine: Any = None) -> FastAPI:
    """Create the API and static frontend, optionally injecting a preloaded engine.

    ``artifact_dir`` overrides ``CATALOG_SEARCH_ARTIFACT_DIR``; the default is the
    project's ``artifacts`` directory. An initialization failure is retained and
    exposed as HTTP 503 without retrying expensive model loads on every request.
    """
    configured_dir = artifact_dir if artifact_dir is not None else os.environ.get("CATALOG_SEARCH_ARTIFACT_DIR")
    artifacts = Path(configured_dir).expanduser().resolve() if configured_dir else PROJECT_ROOT / "artifacts"
    app = FastAPI(title="Catalog search", docs_url=None, redoc_url=None, openapi_url=None, debug=False)
    app.state.engine = engine
    app.state.artifact_dir = artifacts
    app.state.initialization_error = None

    if engine is None:
        try:
            app.state.engine = _load_engine(artifacts)
        except (FileNotFoundError, NotADirectoryError):
            app.state.initialization_error = "Поисковая модель ещё не подготовлена. Создайте артефакты модели и перезапустите сервис."
            LOGGER.warning("catalog_model_artifacts_missing")
        except Exception:
            app.state.initialization_error = "Не удалось загрузить поисковую модель. Проверьте зависимости и артефакты, затем перезапустите сервис."
            LOGGER.error("catalog_model_initialization_failed")

    @app.exception_handler(RequestValidationError)
    async def invalid_parameters(_: Request, error: RequestValidationError) -> JSONResponse:
        fields = sorted({str(item["loc"][-1]) for item in error.errors() if item.get("loc")})
        return JSONResponse(status_code=400, content={"detail": "Некорректные параметры запроса.", "fields": fields})

    @app.middleware("http")
    async def response_headers(request: Request, call_next: Callable) -> Any:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/") or request.url.path == "/health":
            response.headers["Cache-Control"] = "no-store"
        return response

    def ready_engine() -> Any:
        if app.state.engine is None:
            raise HTTPException(503, app.state.initialization_error or "Поисковая модель недоступна.")
        return app.state.engine

    def report_path() -> Path | None:
        # There is no request-controlled filename or arbitrary directory endpoint.
        # The extra locations support an operator-selected artifact directory.
        candidates = (
            artifacts / "reports" / "summary.json",
            artifacts.parent / "reports" / "summary.json",
        )
        for candidate in candidates:
            if candidate.is_file() and not candidate.is_symlink():
                return candidate
        return None

    @app.get("/health")
    def health() -> JSONResponse:
        provider = ready_engine()

        def payload() -> dict[str, Any]:
            metadata = provider.meta()
            return {
                "status": "ok",
                **{key: metadata[key] for key in ("model_version", "catalog_size") if key in metadata},
            }

        return _json_call(payload, operation="health")

    @app.get("/api/meta")
    def metadata() -> JSONResponse:
        provider = ready_engine()

        def payload() -> dict[str, Any]:
            value = dict(provider.meta())
            if report_path() is not None:
                value.setdefault("report_url", "/api/report")
            return value

        return _json_call(payload, operation="meta")

    @app.get("/api/search")
    def search(
        q: str = Query(default=""),
        mode: SearchMode = Query(default="attribute"),
        limit: int = Query(default=12, ge=1, le=40),
    ) -> JSONResponse:
        query = _query(q)
        provider = ready_engine()
        return _json_call(lambda: provider.search(query, mode=mode, limit=limit), operation="search")

    @app.get("/api/product/{product_id}")
    def product(product_id: str) -> JSONResponse:
        if len(product_id) > 256 or any(unicodedata.category(character) in {"Cc", "Cs"} for character in product_id):
            raise HTTPException(400, "Некорректный идентификатор товара.")
        provider = ready_engine()

        def payload() -> Any:
            value = provider.product(product_id)
            if value is None:
                raise HTTPException(404, "Товар не найден в каталоге.")
            return value

        return _json_call(payload, operation="product")

    @app.get("/api/report")
    def report() -> FileResponse:
        path = report_path()
        if path is None:
            raise HTTPException(404, "Отчёт ещё не подготовлен.")
        return FileResponse(path, media_type="application/json", filename="summary.json", content_disposition_type="inline")

    # Mount last: API routes must never be shadowed by the catch-all frontend.
    web_candidates = (PROJECT_ROOT / "web", Path(__file__).resolve().parent / "web")
    web_dir = next((candidate for candidate in web_candidates if candidate.is_dir()), None)
    if web_dir is not None:
        app.mount("/", StaticFiles(directory=web_dir, html=True, follow_symlink=False), name="web")

    return app
