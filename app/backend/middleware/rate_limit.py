"""
rate_limit.py — Middleware FastAPI para rate limiting por IP.

Implementa un sliding window de 1 hora con `collections.deque[float]` por IP,
configurable via `RATE_LIMIT_PER_HOUR` (default 1000). Cuando una IP excede,
devuelve HTTP 429 con header `Retry-After` y JSON con detalles.

Diseño:
  - Estado en memoria (dict {ip: deque}). Por simplicidad single-worker.
    Para producción multi-worker se necesitaría Redis o sticky sessions.
  - Limpia timestamps viejos (> 1h) en cada acceso (amortización O(1)).
  - Solo cuenta requests a `/api/*` (no health checks, docs, static).
  - Whitelist de IPs configurable via `RATE_LIMIT_WHITELIST` (CSV).
  - Headers en cada respuesta:
        X-RateLimit-Limit: 1000
        X-RateLimit-Remaining: <N>
        X-RateLimit-Reset: <epoch seconds>
"""

from __future__ import annotations

import os
import time
import logging
from collections import defaultdict, deque

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

log = logging.getLogger("maia.rate_limit")

# Configuración via env vars
DEFAULT_LIMIT_PER_HOUR = 1000
WINDOW_SECONDS = 3600  # 1 hora


def _parse_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        log.warning(f"Env {name} invalido, usando default {default}")
        return default


def _parse_csv_env(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {ip.strip() for ip in raw.split(",") if ip.strip()}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit por IP, sliding window de 1 hora.

    Aplica solo a paths que empiezan con `/api/`. Otros endpoints (health,
    docs, frontend static) no se cuentan.
    """

    def __init__(self, app, limit_per_hour: int | None = None):
        super().__init__(app)
        self.limit = limit_per_hour or _parse_int_env(
            "RATE_LIMIT_PER_HOUR", DEFAULT_LIMIT_PER_HOUR
        )
        self.whitelist = _parse_csv_env("RATE_LIMIT_WHITELIST")
        self.window = WINDOW_SECONDS
        # ip -> deque de timestamps (segundos epoch)
        self._buckets: dict[str, deque] = defaultdict(deque)
        log.info(
            f"RateLimitMiddleware activo: limit={self.limit} req/hora · "
            f"whitelist={self.whitelist if self.whitelist else '(vacía)'}"
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extrae la IP del cliente. Respeta X-Forwarded-For si está detrás
        de un reverse proxy (nginx/ALB), tomando el primer hop. Sin header,
        cae a request.client.host."""
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    def _purge_old(self, bucket: deque, now: float) -> None:
        """Elimina timestamps fuera de la ventana (más viejos que `now - window`)."""
        threshold = now - self.window
        while bucket and bucket[0] < threshold:
            bucket.popleft()

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # No aplicar a paths que no son /api/*
        if not path.startswith("/api/"):
            return await call_next(request)

        ip = self._get_client_ip(request)

        # Whitelist: bypass total
        if ip in self.whitelist:
            return await call_next(request)

        now = time.time()
        bucket = self._buckets[ip]
        self._purge_old(bucket, now)

        n_used = len(bucket)
        remaining = self.limit - n_used

        if n_used >= self.limit:
            # Cuándo se libera el slot más viejo
            oldest = bucket[0]
            reset_at = oldest + self.window
            retry_after = max(int(reset_at - now), 1)
            log.warning(
                f"RATE LIMIT excedido por {ip}: {n_used}/{self.limit} · "
                f"path={path} · retry_after={retry_after}s"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": (
                        f"Limite excedido: {self.limit} requests por hora por IP. "
                        f"Probar de nuevo en {retry_after} segundos."
                    ),
                    "limit": self.limit,
                    "remaining": 0,
                    "retry_after_seconds": retry_after,
                },
                headers={
                    "X-RateLimit-Limit": str(self.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_at)),
                    "Retry-After": str(retry_after),
                },
            )

        # Registrar el request actual y dejarlo pasar
        bucket.append(now)
        response = await call_next(request)

        # Agregar headers informativos a la respuesta
        new_remaining = self.limit - len(bucket)
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(new_remaining)
        # Reset es cuando se liberará el slot más viejo (o ahora + window si bucket vacío)
        reset_at = (bucket[0] + self.window) if bucket else (now + self.window)
        response.headers["X-RateLimit-Reset"] = str(int(reset_at))

        return response
