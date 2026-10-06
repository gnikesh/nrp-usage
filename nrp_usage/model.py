"""Data structures for a queue-depth snapshot, plus filtering and aggregation.

Pure logic: no I/O, so it is unit-testable without a network.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .prometheus import Sample


def by_label(samples: Sequence[Sample], label: str) -> dict[str, float]:
    """Collapse an instant-vector result to ``{label: value}``, summing duplicates."""
    out: dict[str, float] = {}
    for sample in samples:
        key = sample.label(label)
        if not key or not sample.is_finite:
            continue
        out[key] = out.get(key, 0.0) + sample.value
    return out


def by_pair(samples: Sequence[Sample], *labels: str) -> dict[tuple[str, ...], float]:
    """Collapse an instant-vector result to ``{(label1, label2): value}``."""
    out: dict[tuple[str, ...], float] = {}
    for sample in samples:
        key = tuple(sample.label(name) for name in labels)
        if not all(key) or not sample.is_finite:
            continue
        out[key] = out.get(key, 0.0) + sample.value
    return out


def by_model(samples: Sequence[Sample]) -> dict[str, float]:
    """Collapse an instant-vector result to ``{model_name: value}``."""
    return by_label(samples, "model_name")


def by_model_and_label(samples: Sequence[Sample], label: str) -> dict[str, dict[str, float]]:
    """Collapse a result to ``{model_name: {label_value: value}}``."""
    out: dict[str, dict[str, float]] = {}
    for sample in samples:
        name = sample.label("model_name")
        if not name or not sample.is_finite:
            continue
        bucket = out.setdefault(name, {})
        key = sample.label(label, "other")
        bucket[key] = bucket.get(key, 0.0) + sample.value
    return out


class Level(Enum):
    """Severity of a model's queue state."""

    IDLE = "idle"
    OK = "ok"
    WARN = "warn"
    CRIT = "crit"

    def __lt__(self, other: Level) -> bool:  # ordering used for sorting
        order = (Level.IDLE, Level.OK, Level.WARN, Level.CRIT)
        return order.index(self) < order.index(other)


@dataclass(frozen=True)
class Thresholds:
    """Where a model goes from "in use" to "you will wait"."""

    waiting_warn: int = 1
    waiting_crit: int = 8
    kv_warn: float = 0.75
    kv_crit: float = 0.9

    @classmethod
    def from_options(cls, opts: Mapping[str, object]) -> Thresholds:
        def num(key: str, default):
            value = opts.get(key, default)
            try:
                return type(default)(value)
            except (TypeError, ValueError):
                return default

        return cls(
            waiting_warn=int(num("waiting_warn", 1)),
            waiting_crit=int(num("waiting_crit", 8)),
            kv_warn=float(num("kv_warn", 0.75)),
            kv_crit=float(num("kv_crit", 0.9)),
        )


#: shared immutable default, so call sites do not build one per invocation
DEFAULT_THRESHOLDS = Thresholds()


#: leading alphabetic run of a short name, used as its family ("Qwen3.8-27B" -> "Qwen")
_FAMILY_RE = re.compile(r"^[A-Za-z]+")


@dataclass
class ModelUsage:
    """Instantaneous vLLM engine state for one served model."""

    name: str
    running: int = 0
    waiting: int = 0
    replicas: int = 0
    busy_replicas: int = 0
    kv_cache: float | None = None  # 0..1, worst replica
    queue_avg: float | None = None  # mean seconds spent queued over the window
    waiting_peak: int = 0  # highest queue depth seen in the window (queues drain in seconds,
    queued_share: float | None = None  # so this says how often there was one, 0..1
    waiting_capacity: int = 0
    waiting_deferred: int = 0

    @property
    def load(self) -> int:
        return self.running + self.waiting

    @property
    def is_active(self) -> bool:
        return self.load > 0

    @property
    def queued(self) -> bool:
        return self.waiting > 0

    @property
    def short_name(self) -> str:
        """Drop the org prefix, e.g. ``Qwen/Qwen3.8-27B`` -> ``Qwen3.8-27B``."""
        return self.name.rsplit("/", 1)[-1] if "/" in self.name else self.name

    @property
    def family(self) -> str:
        """Model family: the leading word of the short name.

        ``Qwen/Qwen3.8-27B`` -> ``Qwen``, ``google/gemma-4-31B-it-qat-w4a16-ct`` -> ``gemma``,
        ``Inferact/GLM-5.3-NVFP4`` -> ``GLM``. A name that starts with a digit or symbol keeps its
        whole short name, so unrelated models never merge into one group.
        """
        match = _FAMILY_RE.match(self.short_name)
        return match.group(0) if match else self.short_name

    def level(self, th: Thresholds = DEFAULT_THRESHOLDS) -> Level:
        if self.waiting >= th.waiting_crit:
            return Level.CRIT
        if self.kv_cache is not None and self.kv_cache >= th.kv_crit and self.waiting > 0:
            return Level.CRIT
        if self.waiting >= th.waiting_warn:
            return Level.WARN
        if self.kv_cache is not None and self.kv_cache >= th.kv_warn:
            return Level.WARN
        if not self.is_active:
            return Level.IDLE
        return Level.OK

    def verdict(self, th: Thresholds = DEFAULT_THRESHOLDS) -> str:
        level = self.level(th)
        if level is Level.CRIT:
            return "congested"
        if level is Level.WARN:
            return "queued" if self.queued else "saturated"
        if level is Level.IDLE:
            return "idle"
        return "serving"

    def merge(self, values: Mapping[str, object]) -> None:  # pragma: no cover - convenience
        for key, value in values.items():
            setattr(self, key, value)


@dataclass
class AliasUsage:
    """Gateway-side view, keyed by the user-facing model alias (``glm-5``)."""

    name: str
    concurrency: float = 0.0  # mean in-flight requests over the window
    requests_per_min: float = 0.0
    output_tokens_per_sec: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.concurrency > 0 or self.requests_per_min > 0


@dataclass
class KeyUsage:
    """One API key's traffic against one model, for a single team."""

    token_alias: str
    model: str
    req_per_min: float = 0.0
    mean_seconds: float | None = None  # mean request duration; None with no completed requests

    @property
    def label(self) -> str:
        return self.token_alias


@dataclass
class TeamOverview:
    """Recent traffic for one team, broken down by API key and model."""

    team_id: str
    window: str = "15m"
    keys: list[KeyUsage] = field(default_factory=list)

    @property
    def req_per_min(self) -> float:
        return sum(k.req_per_min for k in self.keys)

    @property
    def token_aliases(self) -> int:
        return len({k.token_alias for k in self.keys})

    @property
    def models(self) -> int:
        return len({k.model for k in self.keys})

    def summary(self) -> str:
        if not self.keys:
            return f"no traffic in the last {self.window}"
        return (
            f"{self.req_per_min:.1f} req/min across {self.token_aliases} "
            f"key{'s' if self.token_aliases != 1 else ''} and {self.models} "
            f"model{'s' if self.models != 1 else ''}"
        )


def build_team_overview(
    team_id: str,
    rpm: Mapping[tuple[str, str], float],
    mean: Mapping[tuple[str, str], float],
    window: str = "15m",
    *,
    min_req_per_min: float = 0.0,
) -> TeamOverview:
    """Merge the per-(key, model) rate and duration results into one overview.

    Series with no traffic in the window still carry a zero-valued counter, so idle
    pairs are dropped rather than filling the report with 0.0 lines.
    """
    pairs: list[KeyUsage] = []
    for (token_alias, model), value in rpm.items():
        if value <= min_req_per_min:
            continue
        pairs.append(
            KeyUsage(
                token_alias=token_alias,
                model=model,
                req_per_min=value,
                mean_seconds=mean.get((token_alias, model)),
            )
        )
    pairs.sort(key=lambda p: (-p.req_per_min, p.token_alias.casefold(), p.model.casefold()))
    return TeamOverview(team_id=team_id, window=window, keys=pairs)


@dataclass
class Totals:
    models: int = 0
    active_models: int = 0
    queued_models: int = 0
    replicas: int = 0
    busy_replicas: int = 0
    running: int = 0
    waiting: int = 0

    @property
    def load(self) -> int:
        return self.running + self.waiting


@dataclass
class Snapshot:
    """Everything one report renders from."""

    source: str
    fetched_at: datetime
    models: list[ModelUsage] = field(default_factory=list)
    aliases: list[AliasUsage] | None = None
    teams: list[TeamOverview] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    window: str = "15m"
    detail: str = ""

    @property
    def totals(self) -> Totals:
        totals = Totals(models=len(self.models))
        for model in self.models:
            totals.running += model.running
            totals.waiting += model.waiting
            totals.replicas += model.replicas
            totals.busy_replicas += model.busy_replicas
            totals.active_models += int(model.is_active)
            totals.queued_models += int(model.queued)
        return totals

    def headline(self, models: Sequence[ModelUsage] | None = None, *, label: str = "", width: int = 0) -> str:
        """Summary line for the given selection (defaults to the whole snapshot)."""
        return headline(self.models if models is None else models, label=label, width=width, window=self.window)

    def local_time(self, fmt: str = "%H:%M:%S") -> str:
        return self.fetched_at.astimezone().strftime(fmt)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "fetched_at": self.fetched_at.astimezone(timezone.utc).isoformat(),
            "window": self.window,
            "totals": asdict(self.totals),
            "families": [
                {
                    "name": group.label,
                    "models": len(group.members),
                    "running": group.summary().running,
                    "waiting": group.summary().waiting,
                    "members": [model.short_name for model in group.members],
                }
                for group in group_by_family(self.models)
            ],
            "models": [{**asdict(m), "short_name": m.short_name, "family": m.family} for m in self.models],
            "aliases": [asdict(a) for a in self.aliases] if self.aliases is not None else None,
            "teams": [
                {
                    "team_id": team.team_id,
                    "window": team.window,
                    "req_per_min": round(team.req_per_min, 3),
                    "token_aliases": team.token_aliases,
                    "models": team.models,
                    "keys": [asdict(k) for k in team.keys],
                }
                for team in self.teams
            ],
            "errors": self.errors,
            "source_detail": self.detail,
        }


def headline(models: Sequence[ModelUsage], *, label: str = "", width: int = 0, window: str = "") -> str:
    """One-line summary of a set of models, naming whatever has a queue.

    Parts are appended in descending importance, so when ``width`` is tight the
    least important clauses are dropped whole instead of being elided mid-word.
    """
    totals = summarise(models)
    if not models:
        return "no data"  # never claim "no queues" for a fleet we could not read
    queued = sorted((m for m in models if m.queued), key=lambda m: -m.waiting)
    peaked = sorted((m for m in models if m.waiting_peak), key=lambda m: -m.waiting_peak)
    parts = [f"{totals.running} running", f"{totals.waiting} waiting"]
    if queued:
        parts.append("queued on " + ", ".join(f"{m.short_name} +{m.waiting}" for m in queued[:3]))
    elif peaked:
        # a waiting gauge drains in seconds, so "no queues" alone misleads. This rides in the same
        # clause because trailing clauses are the first thing dropped on a narrow terminal.
        span = f" in the last {window}" if window else ""
        parts.append(f"no queues now, up to {peaked[0].waiting_peak} queued{span}")
    else:
        parts.append("no queues")
    if totals.waiting:
        parts.append(f"{totals.queued_models} model{'s' if totals.queued_models != 1 else ''} queued")
    parts.append(f"{totals.active_models}/{totals.models} models active")
    if label:
        parts.append(label)

    def join(kept: Sequence[str]) -> str:
        return " · ".join(kept)

    while width and len(parts) > 1 and len(join(parts)) > width:
        parts.pop()  # sacrifice the least important clause
    text = join(parts)
    return text if not width else text[:width]


def build_models(
    running: Mapping[str, float],
    waiting: Mapping[str, float],
    replicas: Mapping[str, float],
    busy_replicas: Mapping[str, float],
    kv_cache: Mapping[str, float],
    queue_avg: Mapping[str, float],
    waiting_by_reason: Mapping[str, Mapping[str, float]],
    waiting_peak: Mapping[str, float] | None = None,
    waiting_share: Mapping[str, float] | None = None,
) -> list[ModelUsage]:
    """Combine the individual query results into one row per model.

    A model appears when any of the engine metrics report it, so a model that is
    up but idle is still listed - that is the useful answer to "what can I run now".
    """
    waiting_peak = waiting_peak or {}
    waiting_share = waiting_share or {}
    names = set(running) | set(waiting) | set(replicas) | set(kv_cache)
    models: list[ModelUsage] = []
    for name in sorted(names):
        reasons = waiting_by_reason.get(name, {})
        models.append(
            ModelUsage(
                name=name,
                running=round(running.get(name, 0.0)),
                waiting=round(waiting.get(name, 0.0)),
                replicas=round(replicas.get(name, 0.0)),
                busy_replicas=round(busy_replicas.get(name, 0.0)),
                kv_cache=kv_cache.get(name),
                queue_avg=queue_avg.get(name),
                waiting_peak=round(waiting_peak.get(name, 0.0)),
                queued_share=waiting_share.get(name),
                waiting_capacity=round(reasons.get("capacity", 0.0)),
                waiting_deferred=round(reasons.get("deferred", 0.0)),
            )
        )
    return models


def build_aliases(
    concurrency: Mapping[str, float],
    rpm: Mapping[str, float],
    tokens_per_sec: Mapping[str, float],
) -> list[AliasUsage]:
    names = set(concurrency) | set(rpm) | set(tokens_per_sec)
    return [
        AliasUsage(
            name=name,
            concurrency=concurrency.get(name, 0.0),
            requests_per_min=rpm.get(name, 0.0),
            output_tokens_per_sec=tokens_per_sec.get(name, 0.0),
        )
        for name in sorted(names)
    ]


#: CLI-facing names; "wait" sorts by the AVG WAIT column, "load" by run+wait
SORT_KEYS = ("load", "running", "waiting", "name", "kv", "wait")


def _display_order(model: ModelUsage) -> tuple[str, str]:
    """Tie-break that matches what the table shows: short name, then full checkpoint name."""
    return (model.short_name.casefold(), model.name.casefold())


def sort_models(models: Sequence[ModelUsage], key: str = "load") -> list[ModelUsage]:
    """Sort rows; anything with a queue always floats up regardless of key.

    Ordering by name uses the short name because that is what the table displays, with the full
    checkpoint name as a stable tie-break for same-named models from different orgs.
    """
    models = list(models)
    if key == "name":
        return sorted(models, key=_display_order)
    if key == "kv":
        return sorted(models, key=lambda m: (-(m.kv_cache or 0.0), -m.load, _display_order(m)))
    if key == "wait":
        return sorted(models, key=lambda m: (-(m.queue_avg or 0.0), -m.load, _display_order(m)))
    field_name = {"running": "running", "waiting": "waiting"}.get(key, "load")
    return sorted(
        models,
        key=lambda m: (
            -getattr(m, field_name),
            -m.waiting,
            -m.running,
            _display_order(m),
        ),
    )


@dataclass
class FamilyGroup:
    """Models that share a family, plus the aggregate shown as its subtotal row."""

    label: str
    members: list[ModelUsage]

    @property
    def totals(self) -> Totals:
        return summarise(self.members)

    def summary(self) -> ModelUsage:
        """A synthetic row carrying the group's aggregates, for the subtotal line.

        Counts add up and KV-cache takes the worst replica in the group. Mean wait, the queue peak
        and the queued share stay blank: none of them can be derived from per-model values, since
        the members' extremes need not occur at the same moment.
        """
        kv_values = [m.kv_cache for m in self.members if m.kv_cache is not None]
        return ModelUsage(
            name=self.label,
            running=sum(m.running for m in self.members),
            waiting=sum(m.waiting for m in self.members),
            replicas=sum(m.replicas for m in self.members),
            busy_replicas=sum(m.busy_replicas for m in self.members),
            kv_cache=max(kv_values) if kv_values else None,
            queue_avg=None,
            waiting_capacity=sum(m.waiting_capacity for m in self.members),
            waiting_deferred=sum(m.waiting_deferred for m in self.members),
        )


def _pick_label(variants: Sequence[str]) -> str:
    """Deterministic spelling for a family: most common, then alphabetically.

    Picking the first one seen would let load ordering decide the label, so a family could
    rename itself between refreshes (``GLM`` vs ``glm``).
    """
    counts = Counter(variants)
    top = max(counts.values())
    return min(value for value, count in counts.items() if count == top)


def group_by_family(models: Sequence[ModelUsage], key: str = "load") -> list[FamilyGroup]:
    """Cluster models by family: groups in a fixed alphabetical block order, members by sort key.

    Grouping is case-insensitive (``GLM`` and ``glm`` are one family). Family blocks never
    reorder with traffic, so a refreshing view stays put; ``key`` only orders the models inside
    each block (use ``-s name`` for a completely static layout).
    """
    buckets: dict[str, list[ModelUsage]] = {}
    spellings: dict[str, list[str]] = {}
    for model in models:
        bucket = model.family.casefold()
        buckets.setdefault(bucket, []).append(model)
        spellings.setdefault(bucket, []).append(model.family)

    groups = [
        FamilyGroup(label=_pick_label(spellings[bucket]), members=sort_models(members, key))
        for bucket, members in buckets.items()
    ]
    groups.sort(key=lambda g: (g.label.casefold(), g.label))
    return groups


def _matches(model: ModelUsage, patterns: Sequence[re.Pattern]) -> bool:
    return any(p.search(model.name) for p in patterns)


def filter_models(
    models: Sequence[ModelUsage],
    patterns: Sequence[str],
    *,
    busy_only: bool = False,
) -> list[ModelUsage]:
    """Keep models whose name matches any pattern (case-insensitive regex, falls back to substring)."""
    compiled = [compile_pattern(p) for p in patterns]
    out = [m for m in models if not compiled or _matches(m, compiled)]
    if busy_only:
        out = [m for m in out if m.is_active]
    return out


def compile_pattern(pattern: str) -> re.Pattern:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE)


def summarise(models: Iterable[ModelUsage]) -> Totals:
    snap = Snapshot(source="", fetched_at=datetime.now(timezone.utc), models=list(models))
    return snap.totals
