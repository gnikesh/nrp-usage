"""PromQL for NRP queue depth, and assembly of a :class:`~nrp_usage.model.Snapshot`.

The queries mirror panel 4 of the "Envoy LLMs" Grafana dashboard, plus the engine
detail metrics that explain *why* a model is busy (KV-cache pressure, queue wait) and
the gateway metrics that show which API keys of a team are driving the traffic.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from .model import (
    AliasUsage,
    Snapshot,
    TeamOverview,
    build_aliases,
    build_models,
    build_team_overview,
    by_label,
    by_model,
    by_model_and_label,
    by_pair,
    compile_pattern,
)
from .prometheus import PromClient, QueryError, Sample
from .trend import Interval, RowPair, SamplePair, TrendTracker

MODEL_LABEL = "model_name"
ALIAS_LABEL = "gen_ai_original_model"

#: a key must average at least this many requests/minute in the window to be listed
MIN_REQ_PER_MIN = 0.05

#: Anything shaped like 30s / 5m / 2h / 1d
WINDOW_RE = re.compile(r"^\d+(ms|[smhdwy])$")

ENGINE_QUERIES: dict[str, str] = {
    "running": 'sum by (model_name) (vllm:num_requests_running)',
    "waiting": 'sum by (model_name) (vllm:num_requests_waiting)',
    "replicas": 'count by (model_name) (vllm:num_requests_running)',
    "busy_replicas": 'count by (model_name) (vllm:num_requests_running > 0)',
    "kv_cache": 'max by (model_name) (vllm:kv_cache_usage_perc)',
    "waiting_reason": 'sum by (model_name, reason) (vllm:num_requests_waiting_by_reason)',
}

#: rate-based queries need the lookback window substituted for {window}
ENGINE_WINDOW_QUERIES: dict[str, str] = {
    # Mean time a finished request spent in the WAITING phase. A percentile is
    # deliberately not used: vLLM's lowest queue-time bucket is le=0.3 and it
    # holds almost every observation, so histogram_quantile would interpolate
    # inside that single bucket and print ~285ms even for an idle model.
    "queue_avg": (
        'sum by (model_name) (rate(vllm:request_queue_time_seconds_sum[{window}])) / '
        'sum by (model_name) (rate(vllm:request_queue_time_seconds_count[{window}]))'
    ),
    # a mean is only meaningful if requests actually left the queue in the window
    "queue_rate": 'sum by (model_name) (rate(vllm:request_queue_time_seconds_count[{window}]))',
    # Recent contention, matching what the dashboard's curve shows. Both are maxima/shares of the
    # SUMMED series (a subquery): sum(max_over_time(...)) would add each replica's own peak and
    # overshoot (measured 81 vs the real 60 for Qwen3.8-27B).
    "waiting_peak": 'max_over_time((sum by (model_name) (vllm:num_requests_waiting))[{window}:15s])',
    # share of the window that had any queue at all: a max carries no frequency, so a one-sample
    # blip and sustained queueing look identical otherwise
    "waiting_share": (
        'sum_over_time((sum by (model_name) (vllm:num_requests_waiting) > bool 0)[{window}:15s]) / '
        'count_over_time((sum by (model_name) (vllm:num_requests_waiting))[{window}:15s])'
    ),
}

#: gateway metrics are labelled by user-facing alias (``glm-5``) instead of the
#: engine's checkpoint name, and carry the team_id / token_alias dashboard vars
GATEWAY_DURATION_SUM = "gen_ai_server_request_duration_seconds_sum"
GATEWAY_DURATION_COUNT = "gen_ai_server_request_duration_seconds_count"
GATEWAY_TOKEN_USAGE = "gen_ai_client_token_usage_sum"

#: per-team breakdown: which API key hit which model, and how long those calls took
TEAM_PAIR = "token_alias, gen_ai_original_model"


class WindowError(ValueError):
    """The requested rate window is not a Prometheus duration."""


def validate_window(window: str) -> str:
    window = window.strip()
    if not WINDOW_RE.match(window):
        raise WindowError(f"invalid window {window!r}; use forms like 30s, 5m, 1h")
    return window


_WINDOW_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0, "y": 31536000.0}
_WINDOW_PART = re.compile(r"(\d+)(ms|[smhdwy])")


def window_seconds(window: str) -> float:
    """Total seconds in a Prometheus duration like ``5m`` (``1h30m`` sums its parts)."""
    return sum(int(amount) * _WINDOW_UNITS[unit] for amount, unit in _WINDOW_PART.findall(window))


#: characters RE2 treats as metacharacters. Note what is *not* here: ``-``, ``#``,
#: ``&``, ``~`` and space are ordinary characters in RE2, and escaping them is a
#: parse error ("unknown escape sequence"), so re.escape() cannot be used here.
_PROM_REGEX_SPECIAL = re.compile(r"([\\.+*?()|[\]{}^$])")

#: Go string literals are what PromQL parses, so backslashes need doubling again
_GO_STRING_ESCAPES = {"\\": "\\\\", '"': '\\"'}


def escape_prom_regex(value: str) -> str:
    """Escape a literal string so RE2 matches it as-is."""
    return _PROM_REGEX_SPECIAL.sub(r"\\\1", value)


def prom_string_literal(text: str) -> str:
    """Quote already-escaped regex text for embedding inside a PromQL ``"..."``."""
    return "".join(_GO_STRING_ESCAPES.get(ch, ch) for ch in text)


def regex_matcher(values: Sequence[str]) -> str:
    """Regex label matcher for the given values (empty selection matches everything)."""
    values = [v for v in values if v not in ("", "$__all", ".*")]
    if not values:
        return ".*"
    return "^(" + "|".join(escape_prom_regex(v) for v in values) + ")$"


def prom_matcher(label: str, values: Sequence[str]) -> str:
    """A complete ``label=~"..."`` matcher, safe to paste into a query."""
    return f'{label}=~"{prom_string_literal(regex_matcher(values))}"'


#: per-team token breakdown, mapped to the result suffix each one feeds
TEAM_TOKEN_TYPES = {"tin": "input", "tout": "output", "treason": "reasoning"}

#: raw cumulative counter, differenced between --watch frames to get true production
TEAM_COUNTER_ALL = "team_cum[{}]"


def build_queries(
    *,
    window: str = "15m",
    gateway: bool = False,
    team_ids: Sequence[str] = (),
    token_aliases: Sequence[str] = (),
    contention: bool = True,
    overview_teams: Sequence[str] = (),
) -> dict[str, str]:
    """All instant queries for one report, keyed by a stable short name."""
    window = validate_window(window)
    queries = dict(ENGINE_QUERIES)
    for key, template in ENGINE_WINDOW_QUERIES.items():
        if not contention and (key.endswith("_peak") or key == "waiting_share"):
            continue  # subqueries are the priciest part of a refresh; skip them when hidden
        queries[key] = template.format(window=window)

    for team in overview_teams:
        sel = prom_matcher("team_id", [team])
        by_pair = f"sum by ({TEAM_PAIR})"
        count = f"{by_pair} (rate({GATEWAY_DURATION_COUNT}{{{sel}}}[{window}]))"
        total = f"{by_pair} (rate({GATEWAY_DURATION_SUM}{{{sel}}}[{window}]))"
        queries[f"team_rpm[{team}]"] = f"{count} * 60"
        queries[f"team_mean[{team}]"] = f"{total} / {count}"
        for suffix, token_type in TEAM_TOKEN_TYPES.items():
            matchers = f'gen_ai_token_type="{token_type}",{sel}'
            queries[f"team_{suffix}[{team}]"] = (
                f"{by_pair} (increase({GATEWAY_TOKEN_USAGE}{{{matchers}}}[{window}]))"
            )
        # one query carrying all three types as cumulative counters, for frame deltas
        queries[TEAM_COUNTER_ALL.format(team)] = (
            f"sum by ({TEAM_PAIR}, gen_ai_token_type) ({GATEWAY_TOKEN_USAGE}{{{sel}}})"
        )
    if gateway:
        extra = []
        if team_ids:
            extra.append(prom_matcher("team_id", team_ids))
        if token_aliases:
            extra.append(prom_matcher("token_alias", token_aliases))

        def selector(*base: str) -> str:
            matchers = [m for m in (*base, *extra) if m]
            return "{" + ",".join(matchers) + "}"

        def rate_sum(metric: str, *base: str) -> str:
            return f"sum by ({ALIAS_LABEL}) (rate({metric}{selector(*base)}[{window}]))"

        queries["alias_concurrency"] = rate_sum(GATEWAY_DURATION_SUM)
        queries["alias_rpm"] = f"{rate_sum(GATEWAY_DURATION_COUNT)} * 60"
        queries["alias_out_tps"] = rate_sum(GATEWAY_TOKEN_USAGE, 'gen_ai_token_type="output"')
    return queries


def _results_to_maps(
    results: Mapping[str, Sequence[Sample] | QueryError],
) -> tuple[dict, dict, list[str], dict[str, dict[str, dict]]]:
    """Split raw results into engine maps, gateway maps, errors and per-team maps."""
    engine: dict[str, object] = {}
    gateway: dict[str, object] = {}
    teams: dict[str, dict[str, dict]] = {}
    errors: list[str] = []
    for key, value in results.items():
        if isinstance(value, QueryError):
            errors.append(f"{key}: {value.message}")
            continue
        if key.startswith("team_") and "[" in key:
            kind, _, team = key.partition("[")
            team = team.rstrip("]")
            slot = teams.setdefault(team, {})
            if kind == "team_cum":
                slot["cum"] = by_pair(
                    value, "token_alias", "gen_ai_original_model", "gen_ai_token_type"
                )
            else:
                slot[kind.removeprefix("team_")] = by_pair(value, "token_alias", "gen_ai_original_model")
        elif key == "waiting_reason":
            engine["waiting_by_reason"] = by_model_and_label(value, "reason")
        elif key.startswith("alias_"):
            gateway[key] = by_label(value, ALIAS_LABEL)
        else:
            engine[key] = by_model(value)
    return engine, gateway, errors, teams


#: a percentile needs at least this many observations in the window to be shown
MIN_QUEUE_SAMPLES = 3.0


def _confident_means(avg: Mapping[str, float], queue_rate: Mapping[str, float], window: str) -> dict[str, float]:
    """Drop averages computed from too few completed requests."""
    seconds = window_seconds(window)
    return {
        name: value
        for name, value in avg.items()
        if queue_rate.get(name, 0.0) * seconds >= MIN_QUEUE_SAMPLES
    }


def make_snapshot(
    results: Mapping[str, Sequence[Sample] | QueryError],
    *,
    source: str,
    window: str = "15m",
    fetched_at: datetime | None = None,
    detail: str = "",
) -> Snapshot:
    """Turn query results into a snapshot (no I/O; used by tests too)."""
    engine, gateway, errors, team_maps = _results_to_maps(results)
    models = build_models(
        running=engine.get("running", {}),  # type: ignore[arg-type]
        waiting=engine.get("waiting", {}),  # type: ignore[arg-type]
        replicas=engine.get("replicas", {}),  # type: ignore[arg-type]
        busy_replicas=engine.get("busy_replicas", {}),  # type: ignore[arg-type]
        kv_cache=engine.get("kv_cache", {}),  # type: ignore[arg-type]
        queue_avg=_confident_means(engine.get("queue_avg", {}), engine.get("queue_rate", {}), window),
        waiting_by_reason=engine.get("waiting_by_reason", {}),  # type: ignore[arg-type]
        waiting_peak=engine.get("waiting_peak", {}),  # type: ignore[arg-type]
        waiting_share=engine.get("waiting_share", {}),  # type: ignore[arg-type]
    )
    aliases: list[AliasUsage] | None = None
    if gateway:
        aliases = build_aliases(
            concurrency=gateway.get("alias_concurrency", {}),  # type: ignore[arg-type]
            rpm=gateway.get("alias_rpm", {}),  # type: ignore[arg-type]
            tokens_per_sec=gateway.get("alias_out_tps", {}),  # type: ignore[arg-type]
        )
    teams: list[TeamOverview] = []
    for team, slot in team_maps.items():
        teams.append(
            build_team_overview(
                team,
                slot.get("rpm", {}),
                slot.get("mean", {}),
                window,
                min_req_per_min=MIN_REQ_PER_MIN,
                in_tokens=slot.get("tin", {}),
                out_tokens=slot.get("tout", {}),
                reasoning_tokens=slot.get("treason", {}),
            )
        )
    teams.sort(key=lambda t: t.team_id.casefold())
    return Snapshot(
        source=source,
        fetched_at=fetched_at or datetime.now(timezone.utc),
        models=models,
        aliases=aliases,
        teams=teams,
        errors=errors,
        window=window,
        detail=detail,
    )


def collect_counters(team_maps: Mapping[str, Mapping[str, Mapping]]) -> dict[SamplePair, float]:
    """Flatten every team's cumulative counters into tracker-ready samples.

    ``increase()`` over a window answers "how much in the last 15m", which is dominated by
    which samples fell off the left edge; two frames 5s apart can differ on an idle model.
    Production *right now* has to come from the raw counter, so it is sampled separately.
    """
    samples: dict[SamplePair, float] = {}
    for team, slot in team_maps.items():
        for (alias, model, token_type), value in slot.get("cum", {}).items():
            samples[(team, alias, model, token_type)] = value
    return samples


def attach_trends(
    snapshot: Snapshot,
    histories: Mapping[str, Mapping[RowPair, list[Interval]]],
    slots: int,
) -> None:
    """Copy the rolling per-interval production onto the matching key/model rows.

    ``histories`` maps team id to {token_type: {row: intervals}}. Each row keeps the last
    ``slots`` intervals oldest-first, which is what the team table renders as its strip.
    """
    for team in snapshot.teams:
        per_type = histories.get(team.team_id, {})
        ins, outs = per_type.get("input", {}), per_type.get("output", {})
        for key in team.keys:
            row = (team.team_id, key.token_alias, key.model)
            key.in_history = list(ins.get(row, []))[-slots:]
            key.out_history = list(outs.get(row, []))[-slots:]


def fetch_snapshot(
    client: PromClient,
    *,
    source: str,
    window: str = "15m",
    gateway: bool = False,
    team_ids: Sequence[str] = (),
    token_aliases: Sequence[str] = (),
    detail: str = "",
    contention: bool = True,
    overview_teams: Sequence[str] = (),
    tracker: TrendTracker | None = None,
    history_slots: int = 3,
) -> Snapshot:
    """Run every query for one report and build the snapshot.

    Pass a ``tracker`` (one instance reused across frames) to get rolling per-interval token
    production on ``snapshot.teams``; without it the counts are shown but never arrowed.
    """
    queries = build_queries(
        window=window,
        gateway=gateway,
        team_ids=team_ids,
        token_aliases=token_aliases,
        contention=contention,
        overview_teams=overview_teams,
    )
    # stamp before querying: the footer time and the trend clock must describe the
    # samples actually read, not when assembly finished
    fetched_at = datetime.now(timezone.utc)
    stamp = time.monotonic()
    results = client.query_many(queries)
    snapshot = make_snapshot(
        results, source=source, window=window, detail=detail, fetched_at=fetched_at
    )
    if tracker is not None and overview_teams:
        _, _, _, team_maps = _results_to_maps(results)
        tracker.observe(collect_counters(team_maps), stamp)
        histories = {
            team: {kind: tracker.intervals(history_slots, kind) for kind in ("input", "output")}
            for team in overview_teams
        }
        attach_trends(snapshot, histories, history_slots)
    return snapshot


def team_traffic_query(window: str) -> str:
    """Every team with requests in the window, as requests/minute."""
    window = validate_window(window)
    return f"sum by (team_id) (rate({GATEWAY_DURATION_COUNT}[{window}])) * 60"


def list_teams(client: PromClient, *, window: str = "15m", patterns: Sequence[str] = ()) -> list[tuple[str, float]]:
    """Discover team ids that are actually in use, busiest first.

    Uses live traffic rather than the ``team_id`` label index, which still lists every team
    that has ever existed (348 at the time of writing) and says nothing about who runs now.
    """
    samples = client.query(team_traffic_query(window))
    rows = [(sample.label("team_id"), sample.value) for sample in samples if sample.label("team_id")]
    if patterns:
        matchers = [compile_pattern(p) for p in patterns]
        rows = [row for row in rows if any(m.search(row[0]) for m in matchers)]
    rows.sort(key=lambda row: (-row[1], row[0].casefold()))
    return [(team, rate) for team, rate in rows if rate > 0]


def describe_queries(
    *,
    window: str = "15m",
    gateway: bool = False,
    team_ids: Sequence[str] = (),
    token_aliases: Sequence[str] = (),
    contention: bool = True,
    overview_teams: Sequence[str] = (),
) -> str:
    """The PromQL behind a report, for --explain debugging."""
    queries = build_queries(
        window=window,
        gateway=gateway,
        team_ids=team_ids,
        token_aliases=token_aliases,
        contention=contention,
        overview_teams=overview_teams,
    )
    return "\n\n".join(f"{name}\n  {expr}" for name, expr in sorted(queries.items()))
