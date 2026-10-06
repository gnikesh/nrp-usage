"""Shared helpers for building fake metric responses (no network in tests)."""

from __future__ import annotations

from nrp_usage.prometheus import QueryError, Sample


def sample(value: float, **labels) -> Sample:
    return Sample(labels={k: str(v) for k, v in labels.items()}, value=float(value), timestamp=1_700_000_000.0)


def vector(pairs) -> list[Sample]:
    """``[("Qwen/x", 3), ...]`` or ``[(("model_name","x"),("reason","cap"), 1), ...]``."""
    out: list[Sample] = []
    for item in pairs:
        if isinstance(item[0], tuple):
            value = item[-1]
            labels = dict(item[:-1])
            out.append(sample(value, **labels))
        else:
            name, value = item
            out.append(sample(value, model_name=name))
    return out


def error(name: str, message: str = "boom") -> QueryError:
    return QueryError(name, message)


def engine_results(
    *,
    running=None,
    waiting=None,
    replicas=None,
    busy_replicas=None,
    kv_cache=None,
    queue_avg=None,
    queue_rate=None,
    waiting_reason=None,
    failures=(),
) -> dict:
    """A complete raw result set in the shape ``PromClient.query_many`` returns."""
    results = {
        "running": vector(running or []),
        "waiting": vector(waiting or []),
        "replicas": vector(replicas or []),
        "busy_replicas": vector(busy_replicas or []),
        "kv_cache": vector(kv_cache or []),
        "queue_avg": vector(queue_avg or []),
        "queue_rate": vector(queue_rate or []),
        "waiting_reason": waiting_reason if waiting_reason is not None else vector([]),
    }
    for name in failures:
        results[name] = error(name, f"{name} unavailable")
    return results
