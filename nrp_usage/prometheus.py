"""Minimal, dependency-free client for a Prometheus-compatible ``/api/v1`` API.

The NRP Grafana exposes its Thanos querier through Grafana's datasource proxy,
so the same code path works for a plain Prometheus URL and for a proxy URL.

Only what this tool needs is implemented: instant vector queries, issued
concurrently, with per-query error isolation so partial data still renders.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import ssl
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

DEFAULT_TIMEOUT = 20.0
USER_AGENT = "nrp-usage"

# Certificate locations that work on a stock macOS "python.org" framework build,
# which ships without a populated OpenSSL CA store.
_SYSTEM_CA_FILES = (
    "/etc/ssl/cert.pem",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/certs/ca-certificates.crt",
)
_MACOS_ROOTS_KEYCHAIN = "/System/Library/Keychains/SystemRootCertificates.keychain"
_VERIFY_HINTS = (
    "CERTIFICATE_VERIFY_FAILED",
    "unable to get local issuer certificate",
)


class PromError(RuntimeError):
    """Base class for transport and query failures.

    ``message`` is the short, user-facing reason; the offending URL is kept
    separately so it only shows up under --verbose.
    """

    def __init__(self, message: str, url: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.url = url

    def __str__(self) -> str:
        return self.message


class ConfigError(PromError):
    """The requested endpoint could not be resolved."""


class AuthError(PromError):
    """The endpoint rejected our credentials (401/403)."""


@dataclass(frozen=True)
class Sample:
    """One element of an instant query result."""

    labels: Mapping[str, str] = field(default_factory=dict)
    value: float = 0.0
    timestamp: float = 0.0

    def label(self, name: str, default: str = "") -> str:
        return self.labels.get(name, default)

    @property
    def is_finite(self) -> bool:
        return self.value == self.value and self.value not in (float("inf"), float("-inf"))


@dataclass(frozen=True)
class _CASource:
    """A named way to build an :class:`ssl.SSLContext`."""

    name: str
    cafile: str | None = None
    cadata: str | None = None

    def build(self) -> ssl.SSLContext:
        if self.cafile:
            ctx = ssl.create_default_context(cafile=self.cafile)
        elif self.cadata:
            ctx = ssl.create_default_context()
            ctx.load_verify_locations(cadata=self.cadata)
        else:
            ctx = ssl.create_default_context()
        return ctx


def _macos_root_pem() -> str:
    """PEM bundle of the macOS system trust store, or "" when unavailable."""
    if not os.path.exists(_MACOS_ROOTS_KEYCHAIN):
        return ""
    try:
        proc = subprocess.run(
            ["security", "find-certificate", "-a", "-p", _MACOS_ROOTS_KEYCHAIN],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if "BEGIN CERTIFICATE" in proc.stdout else ""


@lru_cache(maxsize=1)
def _ca_sources() -> tuple[_CASource, ...]:
    """CA sources to try in order. The first one that verifies wins."""
    sources: list[_CASource] = []
    explicit = os.environ.get("NRP_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if explicit:
        sources.append(_CASource(f"NRP_CA_BUNDLE={explicit}", cafile=explicit))
    sources.append(_CASource("python default store"))
    for path in _SYSTEM_CA_FILES:
        if os.path.exists(path):
            sources.append(_CASource(path, cafile=path))
    try:  # certifi is optional; use it when present
        import certifi

        sources.append(_CASource("certifi", cafile=certifi.where()))
    except (ImportError, OSError):
        pass
    roots = _macos_root_pem()
    if roots:
        sources.append(_CASource("macOS system roots", cadata=roots))

    seen: set[tuple[str, str | None]] = set()
    unique: list[_CASource] = []
    for src in sources:
        key = (src.cafile, src.cadata[:64] if src.cadata else None)
        if key in seen:
            continue
        seen.add(key)
        unique.append(src)
    return tuple(unique)


def _describe_reason(reason: object) -> str:
    """Turn a socket/SSL level failure into something short worth reading."""
    if isinstance(reason, socket.timeout):
        return "timed out"
    if isinstance(reason, OSError) and getattr(reason, "errno", None) in (8, -2, -3):
        return f"could not resolve host ({reason})"
    return str(reason) or type(reason).__name__


def _is_verify_error(exc: BaseException) -> bool:
    text = "".join(f"{type(exc).__name__} {exc} {getattr(exc, 'reason', '')}")
    return any(hint in text for hint in _VERIFY_HINTS)


class PromClient:
    """Executes instant queries against one Prometheus-compatible endpoint."""

    def __init__(
        self,
        api_base: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        headers: Mapping[str, str] | None = None,
        insecure: bool = False,
        max_workers: int = 6,
        recover: Callable[[PromClient], str | None] | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.insecure = insecure
        self.max_workers = max_workers
        #: called with this client when Grafana reports an unknown datasource;
        #: returns a replacement api base, or None to give up.
        self.recover: Callable[[PromClient], str | None] | None = recover
        self._headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if headers:
            self._headers.update({k: v for k, v in headers.items() if v})
        self._tls_index = 0
        self._recover_lock = threading.Lock()

    # -- HTTP -------------------------------------------------------------

    def get_json(self, path: str, params: Mapping[str, str] | None = None) -> dict:
        """GET ``path`` (relative to the api base) and decode JSON."""
        return self._request_json(path, params or {})

    def _context(self) -> ssl.SSLContext:
        if self.insecure:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        sources = _ca_sources()
        self._tls_index = min(self._tls_index, len(sources) - 1)
        return sources[self._tls_index].build()

    def _tls_source_name(self) -> str:
        sources = _ca_sources()
        return sources[min(self._tls_index, len(sources) - 1)].name

    def _request_json(self, path: str, params: Mapping[str, str]) -> dict:
        requested = self.api_base
        try:
            return self._fetch_json(path, params)
        except UnknownDatasource:
            if not self._heal(requested):
                raise
            return self._fetch_json(path, params)

    def _heal(self, failed_base: str) -> bool:
        """Point the client at a working datasource; True when the caller should retry.

        Queries run concurrently, so one winner heals the shared client while the
        others simply retry against the new base instead of each re-discovering.
        """
        if self.recover is None:
            return False
        with self._recover_lock:
            if self.api_base != failed_base:
                return True  # another thread already fixed it
            replacement = self.recover(self)
            if not replacement or replacement.rstrip("/") == failed_base:
                return False
            self.api_base = replacement.rstrip("/")
            return True

    def _fetch_json(self, path: str, params: Mapping[str, str]) -> dict:
        url = f"{self.api_base}{path}?{urllib.parse.urlencode(params)}"
        last_verify_error: ssl.SSLError | None = None
        attempts = 1 if self.insecure else len(_ca_sources())
        for _ in range(attempts):
            request = urllib.request.Request(url, headers=self._headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout, context=self._context()) as resp:
                    payload = resp.read()
                break
            except urllib.error.HTTPError as exc:
                body = ""
                with contextlib.suppress(OSError, UnicodeDecodeError):
                    body = exc.read().decode("utf-8", "replace")[:200].strip()
                if exc.code in (401, 403):
                    raise AuthError(
                        f"HTTP {exc.code}, needs credentials. Pass --token-file, or "
                        "--prom-url to an open Prometheus API.",
                        url=url,
                    ) from exc
                if exc.code == 404 and "unable to find datasource" in body.lower():
                    raise UnknownDatasource(f"HTTP 404: {body}", url=url) from exc
                raise PromError(f"HTTP {exc.code}: {body or exc.reason}", url=url) from exc
            except urllib.error.URLError as exc:
                reason = exc.reason
                if isinstance(reason, ssl.SSLError) and not self.insecure and _is_verify_error(reason):
                    last_verify_error = reason
                    self._tls_index += 1  # try the next CA source
                    continue
                raise PromError(_describe_reason(reason), url=url) from exc
            except OSError as exc:
                # Raw socket failures (connection reset, broken pipe, timeouts) reach
                # us unwrapped when they happen mid-response rather than at connect.
                if _is_verify_error(exc):
                    last_verify_error = exc
                    self._tls_index += 1
                    continue
                raise PromError(_describe_reason(exc), url=url) from exc
        else:
            detail = ", ".join(src.name for src in _ca_sources())
            raise PromError(
                "TLS certificate verification failed "
                f"({last_verify_error}). Tried CA sources: {detail}. "
                "Set NRP_CA_BUNDLE=/path/to/ca.pem, or NRP_INSECURE=1 to skip verification."
            )
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PromError(
                "response was not JSON (a login page or proxy error?): "
                f"{payload[:160]!r}",
                url=url,
            ) from exc

    # -- Queries ----------------------------------------------------------

    def query(self, expr: str, *, time: float | None = None) -> list[Sample]:
        """Run an instant query and return its vector samples."""
        params = {"query": expr}
        if time is not None:
            params["time"] = f"{time:.3f}"
        payload = self._request_json("/query", params)
        if payload.get("status") != "success":
            raise QueryError(expr, str(payload.get("error") or payload))
        return _parse_vector(payload, expr)

    def query_many(self, queries: Mapping[str, str]) -> dict[str, list[Sample] | QueryError]:
        """Run several instant queries concurrently.

        Values are either a list of samples or the exception that query raised,
        so a single broken metric never sinks the whole report.
        """
        if not queries:
            return {}
        if len(queries) == 1:
            name, expr = next(iter(queries.items()))
            return {name: _capture(self.query, expr)}
        workers = max(1, min(self.max_workers, len(queries)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nrp-query") as pool:
            futures = {name: pool.submit(_capture, self.query, expr) for name, expr in queries.items()}
            return {name: fut.result() for name, fut in futures.items()}

    def label_values(self, label: str, matcher: str | None = None, limit: int = 200) -> list[str]:
        params: dict[str, str] = {}
        if matcher:
            params["match[]"] = matcher
        payload = self._request_json(f"/label/{urllib.parse.quote(label, safe='')}/values", params)
        values = payload.get("data") or []
        if isinstance(values, dict):
            values = values.get("values", [])
        return [str(v) for v in values[:limit]]


class QueryError(PromError):
    """A single PromQL expression failed."""

    def __init__(self, expr: str, message: str) -> None:
        super().__init__(f"query failed: {message}")
        self.expr = expr
        self.message = message


class UnknownDatasource(PromError):
    """Grafana rejected the datasource uid."""


def _capture(fn, expr: str):
    """One query's outcome: samples, or a QueryError that report building can survive."""
    try:
        return fn(expr)
    except PromError as exc:
        return QueryError(expr, str(exc))
    except Exception as exc:  # never let one metric take down the whole report
        return QueryError(expr, f"{type(exc).__name__}: {exc}")


def _parse_vector(payload: Mapping, expr: str) -> list[Sample]:
    data = payload.get("data") or {}
    result = data.get("result")
    if result is None:
        raise QueryError(expr, "result was missing from response")
    if not isinstance(result, Iterable) or isinstance(result, (str, bytes, dict)):
        raise QueryError(expr, f"unexpected result type {type(result).__name__}")
    samples: list[Sample] = []
    for entry in result:
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue  # stream or matrix shape we did not ask for
        try:
            timestamp = float(value[0])
            number = float(value[1])
        except (TypeError, ValueError):
            continue
        labels = entry.get("metric") or {}
        samples.append(Sample(labels={str(k): str(v) for k, v in labels.items()}, value=number, timestamp=timestamp))
    return samples
