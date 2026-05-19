"""
test_rate_limit.py — Cobertura del middleware RateLimitMiddleware.

Verifica:
  - Bajo el límite, requests pasan con headers correctos
  - Al exceder el límite, devuelve 429 con Retry-After
  - /health y otros paths fuera de /api/* NO se cuentan
  - Whitelist bypass funciona
  - Sliding window se purga correctamente (timestamps viejos no cuentan)
  - X-Forwarded-For es respetado (modo detrás de proxy)
"""

from __future__ import annotations

import os
import time

import pytest


@pytest.fixture
def fastapi_app(monkeypatch):
    """Construye una app FastAPI mínima con solo el middleware bajo test.
    No requiere modelos preloadeados ni dependencias pesadas.
    """
    # Limpiar singletons del rate limiter entre tests
    monkeypatch.setenv("T1_PRELOAD", "false")
    monkeypatch.setenv("T2_PRELOAD", "false")

    from fastapi import FastAPI
    from backend.middleware.rate_limit import RateLimitMiddleware

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limit_per_hour=5)

    @app.get("/api/test")
    def test_endpoint():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


class TestRateLimitBasic:
    def test_under_limit_passes(self, fastapi_app):
        from fastapi.testclient import TestClient
        client = TestClient(fastapi_app)
        for i in range(5):
            r = client.get("/api/test")
            assert r.status_code == 200
            assert r.headers["x-ratelimit-limit"] == "5"
            assert int(r.headers["x-ratelimit-remaining"]) == 4 - i

    def test_over_limit_returns_429(self, fastapi_app):
        from fastapi.testclient import TestClient
        client = TestClient(fastapi_app)
        for _ in range(5):
            client.get("/api/test")
        r = client.get("/api/test")
        assert r.status_code == 429
        body = r.json()
        assert body["error"] == "rate_limit_exceeded"
        assert body["limit"] == 5
        assert body["remaining"] == 0
        assert body["retry_after_seconds"] > 0
        assert "retry-after" in {k.lower() for k in r.headers.keys()}

    def test_non_api_path_not_counted(self, fastapi_app):
        """GET /health no debería consumir slots del bucket."""
        from fastapi.testclient import TestClient
        client = TestClient(fastapi_app)
        # 5 requests /health (no cuentan)
        for _ in range(5):
            r = client.get("/health")
            assert r.status_code == 200
            # No header de rate limit en respuestas non-/api/
            assert "x-ratelimit-remaining" not in r.headers
        # Ahora 5 requests a /api/test deberían pasar
        for _ in range(5):
            r = client.get("/api/test")
            assert r.status_code == 200


class TestRateLimitWhitelist:
    def test_whitelist_bypass(self, monkeypatch):
        """IPs en RATE_LIMIT_WHITELIST nunca son rate limitadas."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.middleware.rate_limit import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_WHITELIST", "testclient")

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, limit_per_hour=2)

        @app.get("/api/x")
        def x():
            return {"ok": True}

        client = TestClient(app)
        # Aunque el limit es 2, debería poder hacer 10 requests
        for _ in range(10):
            r = client.get("/api/x")
            assert r.status_code == 200


class TestRateLimitSlidingWindow:
    def test_sliding_window_purges_old_timestamps(self, monkeypatch):
        """Cuando timestamps son viejos (> window), deben purgarse."""
        from backend.middleware.rate_limit import RateLimitMiddleware
        from collections import deque

        # Instanciar middleware sin app real para testear el purge
        mw = RateLimitMiddleware.__new__(RateLimitMiddleware)
        mw.window = 3600
        mw.limit = 5
        mw._buckets = {}

        # Bucket con 5 timestamps viejos (hace 2 horas)
        old_ts = time.time() - 7200
        bucket = deque([old_ts] * 5)
        mw._buckets["1.2.3.4"] = bucket

        mw._purge_old(bucket, time.time())
        assert len(bucket) == 0, "Timestamps viejos deberían purgarse"


class TestRateLimitForwardedFor:
    def test_x_forwarded_for_respected(self, monkeypatch):
        """Cuando hay X-Forwarded-For, usa esa IP (primera del CSV)."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.middleware.rate_limit import RateLimitMiddleware

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, limit_per_hour=3)

        @app.get("/api/y")
        def y():
            return {"ok": True}

        client = TestClient(app)
        # Una IP agota su cupo
        for _ in range(3):
            r = client.get("/api/y", headers={"x-forwarded-for": "10.0.0.1"})
            assert r.status_code == 200
        r = client.get("/api/y", headers={"x-forwarded-for": "10.0.0.1"})
        assert r.status_code == 429

        # Otra IP tiene cupo propio
        r = client.get("/api/y", headers={"x-forwarded-for": "10.0.0.2"})
        assert r.status_code == 200


class TestRateLimitEnvDefault:
    def test_default_limit_from_env(self, monkeypatch):
        """Sin pasar limit_per_hour, lee RATE_LIMIT_PER_HOUR del env."""
        from fastapi import FastAPI
        from backend.middleware.rate_limit import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "42")
        app = FastAPI()
        mw = RateLimitMiddleware(app)
        assert mw.limit == 42

    def test_default_is_1000(self, monkeypatch):
        """Sin env var, default es 1000."""
        from fastapi import FastAPI
        from backend.middleware.rate_limit import RateLimitMiddleware

        monkeypatch.delenv("RATE_LIMIT_PER_HOUR", raising=False)
        app = FastAPI()
        mw = RateLimitMiddleware(app)
        assert mw.limit == 1000
