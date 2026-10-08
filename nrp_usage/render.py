"""Terminal rendering: an aligned queue-depth table with colour that degrades cleanly.

Layout strategy: group rows by model family with a subtotal per family, build the widest
column set the terminal can hold (dropping optional columns), then middle-elide names to
absorb whatever is left over.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .model import (
    DEFAULT_THRESHOLDS,
    AliasUsage,
    KeyUsage,
    Level,
    ModelUsage,
    Snapshot,
    TeamOverview,
    Thresholds,
    group_by_family,
    summarise,
)
from .trend import Interval, Trend, combine

CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "cyan": "36",
    "grey": "90",
}
LEVEL_STYLE: dict[Level, str] = {
    Level.IDLE: "grey",
    Level.OK: "green",
    Level.WARN: "yellow",
    Level.CRIT: "red",
}
GAP = "   "
#: indent for models listed under a family subtotal
GROUP_GAP = "  "


def clamp(text: str, width: int, theme: Theme) -> str:
    """Force a prose line onto one terminal row, keeping its tail if it must shrink."""
    if width <= 0 or len(text) <= width:
        return text
    return elide(text, width, theme)


def strip_ansi(text: str) -> str:
    """Visible characters only, for width maths."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\x1b":
            end = text.find("m", i)
            if end == -1:
                break
            i = end + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def supports_color(stream=None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def unicode_ok(stream=None) -> bool:
    encoding = (getattr(stream or sys.stdout, "encoding", None) or "").lower()
    return "utf" in encoding


@dataclass
class Theme:
    """ANSI styling that turns itself off when it would be noise."""

    color: bool = True
    unicode: bool = True

    @classmethod
    def detect(cls, stream=None, *, force_color: bool = False, no_color: bool = False) -> Theme:
        color = supports_color(stream) or bool(force_color)
        if no_color:
            color = False
        return cls(color=color, unicode=unicode_ok(stream))

    @classmethod
    def plain(cls) -> Theme:
        return cls(color=False, unicode=True)

    def paint(self, text: str, *styles: str | None) -> str:
        chosen = [s for s in styles if s]
        if not self.color or not chosen:
            return text
        return "".join(f"\x1b[{CODES[s]}m" for s in chosen) + text + "\x1b[0m"

    def gauge(self, ratio: float, cells: int = 8) -> str:
        ratio = max(0.0, min(1.0, ratio))
        filled = round(ratio * cells)
        mark = "█" if self.unicode else "#"
        empty = "·" if self.unicode else "."
        return mark * filled + empty * (cells - filled)


@dataclass
class RenderOptions:
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    show_gateway: bool = False
    show_reason: bool = False
    full_names: bool = False
    group: bool = True
    show_org: bool = False
    show_contention: bool = True
    sort_key: str = "load"
    window: str = "15m"
    max_keys: int = 8
    history_slots: int = 3
    history_reserved: bool = False  # watching: hold the strip's columns even before data exists
    width: int = 0

    def resolved_width(self) -> int:
        if self.width:
            return max(60, int(self.width))
        return max(72, min(shutil.get_terminal_size((110, 24)).columns, 220))


@dataclass
class RowSpec:
    """One table line: a model row, or a family subtotal backed by an aggregate row.

    ``ends_group`` marks the last line of a family block, which is where a blank line goes.
    """

    model: ModelUsage
    label: str
    indent: str = ""
    bold: bool = False
    is_group: bool = False
    ends_group: bool = False
    starts_group: bool = False

    @property
    def display(self) -> str:
        return self.indent + self.label


def display_label(model: ModelUsage, opts: RenderOptions) -> str:
    """Name shown in the MODEL column: the checkpoint without its org by default."""
    return model.name if opts.show_org else model.short_name


def plan_rows(models: Sequence[ModelUsage], opts: RenderOptions) -> list[RowSpec]:
    """Turn the selected models into table lines, grouped by family with a subtotal each.

    Families are emitted in a fixed alphabetical block order so the layout does not move while
    traffic changes; ``sort_key`` only orders the models inside each block. A family with one
    model gets no subtotal line, since a header identical to its only row is just noise.
    """
    if not opts.group:
        return [RowSpec(model, display_label(model, opts)) for model in models]

    rows: list[RowSpec] = []
    for group in group_by_family(models, opts.sort_key):
        if len(group.members) == 1:
            member = group.members[0]
            rows.append(RowSpec(member, display_label(member, opts), starts_group=True, ends_group=True))
            continue
        summary = group.summary()
        label = f"{group.label} ({len(group.members)} models)"
        rows.append(RowSpec(summary, label, bold=True, is_group=True, starts_group=True))
        rows.extend(
            RowSpec(
                member,
                display_label(member, opts),
                indent=GROUP_GAP,
                ends_group=index == len(group.members) - 1,
            )
            for index, member in enumerate(group.members)
        )
    return rows


TOTALS_MARK = "rule"


def interleave_rows(
    table: Sequence[str],
    marks: Sequence[str],
    *,
    header_rows: int,
    rule: str = "",
) -> list[str]:
    """Decorate formatted rows: "" keeps the row, "blank" and "rule" insert a line before it."""
    lines = list(table[:header_rows])
    for index, line in enumerate(table[header_rows:]):
        mark = marks[index] if index < len(marks) else ""
        if mark == "rule" and rule:
            lines.append(rule)
        elif mark == "blank":
            lines.append("")
        lines.append(line)
    return lines


# "=" would read as an equation between two quantities, so steady flow gets an arrow too
ARROWS = {Trend.UP: "↑", Trend.DOWN: "↓", Trend.FLAT: "→", Trend.UNKNOWN: ""}
ASCII_ARROWS = {Trend.UP: "^", Trend.DOWN: "v", Trend.FLAT: ">", Trend.UNKNOWN: ""}
TREND_STYLE = {Trend.UP: "green", Trend.DOWN: "red", Trend.FLAT: "grey", Trend.UNKNOWN: "grey"}


def half_cell(item: Interval | None, theme: Theme, unicode: bool) -> str:
    """One side of a pair: ``↑3.0K``. Blank when no interval has closed yet."""
    if item is None:
        return theme.paint("·", "dim")
    arrow = (ARROWS if unicode else ASCII_ARROWS)[item.trend]
    # the magnitude always rides along: a lone "↓" cannot say whether that is "a bit slower"
    # or "produced nothing", and those want very different reactions
    return theme.paint(f"{arrow}{format_tokens(item.tokens)}", TREND_STYLE[item.trend])


def pair_cell(pair: tuple[Interval | None, Interval | None], theme: Theme, unicode: bool) -> str:
    """One frame's reading, as ``IN/OUT`` with an arrow on each side."""
    return f"{half_cell(pair[0], theme, unicode)}/{half_cell(pair[1], theme, unicode)}"


def format_tokens(value: float | None) -> str:
    """Compact token counts as K / M / B, one decimal place.

    15 minutes of traffic runs to seven figures, so raw counts do not fit a column.
    Below 1000 the exact count is shown, because "0.0K" tells you less than "812".
    """
    if value is None:
        return "-"
    number = float(value)
    if number < 0:
        return "-"
    if number < 1000:
        return f"{number:.0f}"
    for threshold, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        scaled = number / threshold
        if scaled >= 0.9995:  # only a unit this value actually rounds into
            if scaled >= 999.95:  # "1000.0M" is really "1.0B", so promote
                continue
            return f"{scaled:.1f}{unit}"
    return f"{number:.0f}"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds <= 0:
        return "0ms"
    if seconds < 0.001:
        return "<1ms"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 120:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}m{secs:02d}s"


def duration_style(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    if seconds >= 30:
        return "red"
    if seconds >= 3:
        return "yellow"
    return None


def visible_len(text: str) -> int:
    return len(strip_ansi(text))


def pad(text: str, size: int, align: str) -> str:
    """Pad to a *visible* width: str.ljust() would count ANSI bytes as content."""
    fill = max(0, size - visible_len(text))
    return text + " " * fill if align == "left" else " " * fill + text


def elide(text: str, width: int, theme: Theme) -> str:
    """Middle-elide so both the org prefix and the version suffix stay visible."""
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    mark = "…" if theme.unicode else "~"
    if width <= len(mark):
        return mark[:width]
    keep = width - len(mark)
    head = keep // 2
    tail = keep - head
    return text[:head] + mark + (text[len(text) - tail :] if tail else "")


@dataclass
class Column:
    key: str
    header: str
    align: str = "right"  # left|right
    optional: bool = False  # droppable when the terminal is narrow
    fixed: int | None = None  # hard width, otherwise measured from content

    def width(self, cells: Sequence[str]) -> int:
        if self.fixed is not None:
            return self.fixed
        return max([len(self.header)] + [len(strip_ansi(c)) for c in cells])


KV_WIDTH = 14  # '████····  44%'


def drop_to_fit(
    columns: list[Column],
    primary: str,
    primary_floor: int,
    width: int,
    content_widths: Mapping[str, int] | None = None,
) -> list[Column]:
    """Remove optional columns, right-most first, until the table fits ``width``.

    Runs before rendering so a header is never squeezed into elided mush: losing a whole
    column is far easier to read than "RE…SON". ``content_widths`` gives the measured width
    of columns sized from their data, so the estimate is not a guess that only fails later.
    """
    content = dict(content_widths or {})

    def estimate(col: Column) -> int:
        # planned before cells exist, so approximate what format_table will measure
        if col.fixed is not None:
            return col.fixed
        if col.key == primary:
            return primary_floor
        return max(len(col.header), content.get(col.key, 6))

    def natural(cols: list[Column]) -> int:
        return sum(estimate(c) for c in cols) + len(GAP) * (len(cols) - 1)

    while natural(columns) > width and any(c.optional for c in columns):
        for candidate in reversed(columns):
            if candidate.optional:
                columns.remove(candidate)
                break
    return columns


def plan_columns(rows: Sequence[RowSpec], opts: RenderOptions, width: int) -> list[Column]:
    columns = [
        Column("name", "MODEL", align="left"),
        Column("runwait", "RUN/WAIT"),
    ]
    if opts.show_contention:
        # recent contention, because RUN/WAIT is a snapshot and the dashboard shows a curve
        columns.append(Column("waitmax", "WAIT MAX", fixed=9, optional=True))
        columns.append(Column("queued", "QUEUED", fixed=7, optional=True))
    if opts.show_reason and any(row.model.waiting for row in rows):
        # ask for --reason and the column appears whenever anything is queued;
        # waiting_by_reason can lag num_requests_waiting by a scrape interval
        columns.append(Column("reason", "cap/def", fixed=8, optional=True))
    columns.append(Column("nodes", "NODES", fixed=6, optional=True))
    columns.append(Column("kv", "KV CACHE", fixed=KV_WIDTH, optional=True))
    columns.append(Column("queue", "AVG WAIT", fixed=9, optional=True))
    columns.append(Column("status", "STATUS", align="left", fixed=12))

    def name_floor() -> int:
        # reserve the widest thing the name column will ever have to show
        floor = max((len(row.display) for row in rows), default=10)
        if not opts.full_names:
            floor = min(floor, 34)
        return max(floor, len("MODEL"), 12)

    return drop_to_fit(columns, primary="name", primary_floor=name_floor(), width=width,
                       content_widths={"status": 12})


def cell_for(row: RowSpec, key: str, theme: Theme, opts: RenderOptions) -> str:
    """One styled cell. ``row.model`` may be a real model or a family aggregate."""
    model = row.model
    level = model.level(opts.thresholds)
    style = LEVEL_STYLE[level]
    if key == "name":
        name_style = None if level is Level.OK else style
        return theme.paint(row.display, name_style, "bold" if row.bold else None)
    if key == "runwait":
        run = theme.paint(str(model.running), "bold") if model.running else theme.paint("0", "dim")
        wait = (
            theme.paint(str(model.waiting), style, "bold")
            if model.waiting
            else theme.paint("0", "dim")
        )
        return f"{run}{theme.paint('/', 'grey')}{wait}"
    if key == "waitmax":
        if row.is_group:
            return theme.paint("-", "dim")
        if not model.waiting_peak:
            return theme.paint("0", "dim")
        return theme.paint(str(model.waiting_peak), "yellow")
    if key == "queued":
        if row.is_group or model.queued_share is None:
            return theme.paint("-", "dim")
        percent = model.queued_share * 100
        text = f"{percent:3.0f}%"
        style = "red" if percent >= 50 else "yellow" if percent >= 20 else "green" if percent > 0 else "dim"
        return theme.paint(text, style)
    if key == "reason":
        if not (model.waiting_capacity or model.waiting_deferred):
            return theme.paint("-", "dim")
        return f"{model.waiting_capacity}/{model.waiting_deferred}"
    if key == "nodes":
        text = f"{model.busy_replicas}/{model.replicas}"
        return theme.paint(text, "cyan" if model.busy_replicas else "grey")
    if key == "kv":
        if model.kv_cache is None:
            return theme.paint("n/a", "dim")
        percent = model.kv_cache * 100
        gauge_style = "red" if percent >= 90 else "yellow" if percent >= 75 else "green" if percent >= 20 else "grey"
        return theme.paint(theme.gauge(model.kv_cache), gauge_style) + f" {percent:3.0f}%"
    if key == "queue":
        text = format_duration(model.queue_avg)
        return theme.paint(text, duration_style(model.queue_avg)) if text != "-" else theme.paint(text, "dim")
    if key == "status":
        return theme.paint(model.verdict(opts.thresholds), style)
    raise KeyError(key)  # pragma: no cover


def format_table(
    columns: Sequence[Column],
    rows: Sequence[Sequence[str]],
    *,
    width: int,
    theme: Theme,
    primary: str = "name",
    primary_floor: int = 12,
    separator: bool = False,
) -> list[str]:
    """Render styled cells into aligned lines; the primary column absorbs slack."""
    index = {col.key: i for i, col in enumerate(columns)}
    sizes: dict[str, int] = {}
    for col in columns:
        if col.key == primary:
            continue
        cells = [row[index[col.key]] for row in rows]
        sizes[col.key] = col.width(cells)

    primary_col = next(col for col in columns if col.key == primary)
    name_needed = max(
        len(primary_col.header),
        max((len(strip_ansi(row[index[primary]])) for row in rows), default=8),
    )
    gaps = len(GAP) * (len(columns) - 1)
    others = sum(sizes.values())

    if name_needed + others + gaps <= width:
        sizes[primary] = name_needed
    else:
        # Shrink in the order that loses meaning last: names elide legibly
        # ("deepseek-v4-fl…"), a truncated number does not ("124.1" from "124.1K").
        deficit = name_needed + others + gaps - width
        floor = min(name_needed, primary_floor)
        cut = min(deficit, name_needed - floor)
        sizes[primary] = name_needed - cut
        deficit -= cut
        named = [col for col in reversed(columns) if col.key != primary and col.align == "left"]
        numeric = [col for col in reversed(columns) if col not in named and col.key != primary]
        for col in named + numeric:
            if deficit <= 0:
                break
            room = sizes[col.key] - 5
            if room <= 0:
                continue
            take = min(room, deficit)
            sizes[col.key] -= take
            deficit -= take

    def paint_header(col: Column) -> str:
        text = col.header
        size = sizes[col.key]
        text = text if len(text) <= size else elide(text, size, theme)
        # pad outside the escape codes so trailing spaces can be stripped
        return pad(theme.paint(text, "bold", "grey"), size, col.align)

    lines = [GAP.join(paint_header(col) for col in columns).rstrip()]
    if separator:
        lines.append(theme.paint("-" * len(strip_ansi(lines[0])), "dim"))

    for row in rows:
        parts = []
        for col in columns:
            size = sizes[col.key]
            text = row[index[col.key]]
            visible = strip_ansi(text)
            if len(visible) > size:
                # left-aligned cells elide with a marker; silently chopping a name into
                # "gemma-4-31B-i" leaves something that reads like a different real model
                text = elide(visible, size, theme) if col.align == "left" else visible[:size]
            parts.append(pad(text, size, col.align))
        lines.append(GAP.join(parts).rstrip())
    return lines


def render_header(snapshot: Snapshot, models: Sequence[ModelUsage], theme: Theme, opts: RenderOptions) -> list[str]:
    """Title + one summary line, always describing what is actually on screen."""
    width = opts.resolved_width()
    lines = [theme.paint(clamp("NRP model usage", width, theme), "bold", "cyan")]
    scope = ""
    if len(models) != len(snapshot.models):
        scope = f"showing {len(models)} of {len(snapshot.models)} models"
    lines.append(theme.paint(snapshot.headline(models, label=scope, width=width), "grey"))
    if opts.show_contention:
        # WAIT MAX / QUEUED / AVG WAIT no longer carry the window in their header, so say it once
        lines.append(
            theme.paint(
                clamp(f"WAIT MAX, QUEUED and AVG WAIT cover the last {opts.window}", width, theme),
                "dim",
            )
        )
    queued = [m for m in models if m.queued]
    if queued:
        worst = max(queued, key=lambda m: m.waiting)
        # A queue that just formed has no completed requests to average yet, and "( -, mean wait)"
        # reads like a broken number rather than an unknown one, so the clause is dropped.
        timing = f", {format_duration(worst.queue_avg)} mean wait so far" if worst.queue_avg is not None else ""
        lines.append(
            theme.paint(
                clamp(f"busiest queue: {display_label(worst, opts)} ({worst.waiting} waiting{timing})", width, theme),
                "yellow",
            )
        )
    else:
        # Always emitted, never conditionally: omitting it lifts the whole table a row whenever
        # a queue drains and drops it back when one forms. The placeholder also confirms the
        # absence is real rather than a missing line.
        idle = [m for m in models if not m.queued and m.waiting_peak]
        if idle:
            quiet = max(idle, key=lambda m: m.waiting_peak)
            detail = (
                f"busiest queue: none now, up to {quiet.waiting_peak} in the last "
                f"{snapshot.window} on {display_label(quiet, opts)}"
            )
        else:
            detail = "busiest queue: none"
        lines.append(theme.paint(clamp(detail, width, theme), "dim"))
    return lines


def totals_row(models: Sequence[ModelUsage], columns: Sequence[Column], theme: Theme) -> list[str]:
    totals = summarise(models)
    cells = []
    for col in columns:
        if col.key == "name":
            cells.append(theme.paint("TOTAL", "bold"))
        elif col.key == "runwait":
            run = theme.paint(str(totals.running), "bold")
            wait = theme.paint(str(totals.waiting), "bold" if totals.waiting else "dim")
            cells.append(f"{run}{theme.paint('/', 'grey')}{wait}")
        elif col.key == "reason":
            capacity = sum(m.waiting_capacity for m in models)
            deferred = sum(m.waiting_deferred for m in models)
            if not (capacity or deferred):
                cells.append(theme.paint("-", "dim"))
            else:
                cells.append(theme.paint(f"{capacity}/{deferred}", "bold"))
        elif col.key == "nodes":
            cells.append(theme.paint(f"{totals.busy_replicas}/{totals.replicas}", "bold"))
        elif col.key == "status":
            cells.append(theme.paint(f"{totals.active_models}/{totals.models} active", "bold"))
        else:
            # KV / AVG WAIT / WAIT MAX / QUEUED do not aggregate onto a fleet total
            cells.append(theme.paint("-", "dim"))
    return cells


def render_gateway(aliases: Sequence[AliasUsage], theme: Theme, opts: RenderOptions) -> list[str]:
    header = theme.paint("Gateway traffic by model alias", "bold")
    if not aliases:
        return [header, theme.paint("  no traffic in window", "grey")]
    columns = [
        Column("name", "ALIAS", align="left"),
        Column("conc", "CONC", fixed=7),
        Column("rpm", "REQ/MIN", fixed=9),
        Column("tps", "OUT TOK/S", fixed=11),
    ]
    rows = []
    for alias in aliases:
        style = "cyan" if alias.is_active else "grey"
        rows.append(
            [
                theme.paint(alias.name, style),
                theme.paint(f"{alias.concurrency:.2f}", style),
                theme.paint(f"{alias.requests_per_min:.1f}", style),
                theme.paint(f"{alias.output_tokens_per_sec:.0f}", style),
            ]
        )
    return [header, *format_table(columns, rows, width=opts.resolved_width(), theme=theme, separator=True)]


# Fixed, not content-derived, so the table cannot reflow between --watch frames: sizing
# these from the data shifts every column sideways whenever a delta appears or disappears.
# 15 is the widest legitimate cell, e.g. "999.9M ↑999.9K".
TOKEN_WIDTH = 15


PAIR_WIDTH = 14  # widest honest cell: two 5-char counts plus an arrow each, e.g. "↑99.9K/↑99.9K"


def slot_item(key: KeyUsage, stream: str, position: int, total: int) -> Interval | None:
    """The interval belonging to strip column ``position`` for one row and one stream.

    Histories are aligned to their newest entry, so a row that appeared only recently still
    puts its latest reading under the newest column instead of shifting it left.
    """
    history = getattr(key, f"{stream}_history", None) or []
    if not history:
        return None
    index = len(history) - total + position
    return history[index] if 0 <= index < len(history) else None


def fold_slot(keys: Sequence[KeyUsage], stream: str, position: int, total: int) -> Interval | None:
    """Aggregate one strip column across several rows."""
    parts = [item for key in keys if (item := slot_item(key, stream, position, total)) is not None]
    return combine(*parts) if parts else None


def render_team(team: TeamOverview, theme: Theme, opts: RenderOptions) -> list[str]:
    """Which API keys of a team are hitting which models, and what each frame produced.

    The strip is one ``IN/OUT`` column per recent frame, oldest on the left and the newest
    reading on the right, each side arrowed against its own predecessor. Rows are capped at
    ``opts.max_keys``; the totals line and --json still cover every pair.
    """
    slots_hint = max(1, opts.history_slots)
    title = theme.paint(f"team {team.team_id}", "bold")
    subtitle = theme.paint(team.summary(), "grey")
    if not team.keys:
        # still three lines, so a team that starts or stops producing traffic does not lift the
        # blocks below it by a row
        return [
            title,
            subtitle,
            theme.paint(f"run with --watch for a {slots_hint}-frame IN/OUT history strip", "dim"),
        ]

    shown, hidden = team.keys[: opts.max_keys], team.keys[opts.max_keys :]
    slots = slots_hint
    # short enough that a narrow terminal cannot elide it into "2 key/… pairs"
    total_label = f"{len(team.keys)} pair{'s' if len(team.keys) != 1 else ''}"
    key_floor = max(12, len(total_label))

    def mean_cell(seconds: float | None) -> str:
        if seconds is None:
            return theme.paint("-", "dim")
        return theme.paint(format_duration(seconds), duration_style(seconds))

    def reason_cell(value: float) -> str:
        return theme.paint(format_tokens(value), "grey") if value > 0 else theme.paint("-", "dim")

    def window_cell(in_tokens: float, out_tokens: float, bold: bool = False) -> str:
        if not in_tokens and not out_tokens:
            return theme.paint("-/-", "dim")
        text = f"{format_tokens(in_tokens)}/{format_tokens(out_tokens)}"
        return theme.paint(text, "grey" if not bold else None)

    want_reason = any(k.reasoning_tokens > 0 for k in team.keys)
    # With no history at all (one-shot run) the strip would be pure placeholders, so it is
    # omitted. While watching, the columns are reserved from the first frame so the table
    # never changes shape mid-session as the cells fill in.
    has_history = any(k.in_history or k.out_history for k in team.keys)
    show_strip = has_history or opts.history_reserved
    columns: list[Column] = [
        Column("key", "API KEY", align="left"),
        Column("model", "MODEL", align="left"),
        Column("rpm", "REQ/MIN", fixed=8, optional=True),
        Column("mean", "MEAN SEC", fixed=9, optional=True),
    ]
    if show_strip:
        # oldest left, newest right; never dropped, since the strip is the point of the section.
        # Age-labelled because three identical headers do not say which side is newest.
        for index in range(slots):
            age = index - (slots - 1)
            header = "IN/OUT now" if age == 0 else f"IN/OUT {age}"
            columns.append(Column(f"slot{index}", header, fixed=PAIR_WIDTH))
    # only droppable once the strip is competing for width; otherwise it is the sole token view
    columns.append(Column("window", f"{team.window} IN/OUT", fixed=PAIR_WIDTH, optional=show_strip))
    if want_reason:
        columns.append(Column("reason", "REASON", fixed=8, optional=True))
    columns = drop_to_fit(
        columns,
        primary="key",
        primary_floor=key_floor,
        width=opts.resolved_width(),
        content_widths={"model": max([len("MODEL")] + [len(k.model) for k in team.keys])},
    )

    def make_row(values: dict[str, str]) -> list[str]:
        """Emit cells in surviving-column order; positional rows break when one is dropped."""
        return [values.get(col.key, "") for col in columns]

    def strip_cells(key: KeyUsage) -> dict[str, str]:
        return {
            f"slot{position}": pair_cell(
                (slot_item(key, "in", position, slots), slot_item(key, "out", position, slots)),
                theme,
                theme.unicode,
            )
            for position in range(slots)
        }

    def strip_fold(keys: Sequence[KeyUsage]) -> dict[str, str]:
        return {
            f"slot{position}": pair_cell(
                (fold_slot(keys, "in", position, slots), fold_slot(keys, "out", position, slots)),
                theme,
                theme.unicode,
            )
            for position in range(slots)
        }

    rows = [
        make_row(
            {
                "key": theme.paint(key.token_alias, "cyan"),
                "model": theme.paint(key.model, "blue"),
                "rpm": theme.paint(f"{key.req_per_min:.1f}", "bold"),
                "mean": mean_cell(key.mean_seconds),
                "window": window_cell(key.in_tokens, key.out_tokens),
                "reason": reason_cell(key.reasoning_tokens),
                **strip_cells(key),
            }
        )
        for key in shown
    ]
    if hidden:
        rows.append(
            make_row(
                {
                    "key": theme.paint(f"... +{len(hidden)}", "grey"),
                    "rpm": theme.paint(f"{sum(k.req_per_min for k in hidden):.1f}", "grey"),
                    "mean": theme.paint("-", "dim"),
                    "window": window_cell(
                        sum(k.in_tokens for k in hidden), sum(k.out_tokens for k in hidden)
                    ),
                    "reason": reason_cell(sum(k.reasoning_tokens for k in hidden)),
                    **strip_fold(hidden),
                }
            )
        )
    rows.append(
        make_row(
            {
                "key": theme.paint(total_label, "bold"),
                "rpm": theme.paint(f"{team.req_per_min:.1f}", "bold"),
                "mean": theme.paint("-", "dim"),  # no weighted mean available across keys
                "window": window_cell(team.in_tokens, team.out_tokens, bold=True),
                "reason": reason_cell(team.reasoning_tokens),
                **strip_fold(team.keys),
            }
        )
    )

    clauses = []
    if show_strip:
        clauses.append(f"the {slots} IN/OUT columns are the last {slots} frames, oldest left / newest right")
        clauses.append("arrows compare each frame's rate with the one before")
    else:
        clauses.append(f"run with --watch for a {slots}-frame IN/OUT history strip")
    clauses.append(f"{team.window} IN/OUT covers the whole window; REASON is part of its OUT")
    while len(clauses) > 1 and len(" · ".join(clauses)) > opts.resolved_width():
        clauses.pop()
    lines = [title, subtitle, theme.paint(" · ".join(clauses), "dim")]
    lines += format_table(
        columns,
        rows,
        width=opts.resolved_width(),
        theme=theme,
        primary="key",
        primary_floor=key_floor,
        separator=True,
    )
    return lines


def render_report(
    snapshot: Snapshot,
    models: Sequence[ModelUsage],
    theme: Theme,
    opts: RenderOptions,
) -> str:
    width = opts.resolved_width()
    lines = render_header(snapshot, models, theme, opts)
    lines.append("")
    if not models:
        # distinguish "your filter matched nothing" from "the fleet reported nothing"
        message = (
            "no models matched"
            if snapshot.models
            else "no models reporting data"
        )
        lines.append(theme.paint(message, "yellow"))
    else:
        specs = plan_rows(models, opts)
        columns = plan_columns(specs, opts, width)
        rows = [[cell_for(spec, col.key, theme, opts) for col in columns] for spec in specs]
        rows.append(totals_row(models, columns, theme))
        longest = max([len(spec.display) for spec in specs] + [len("MODEL")])
        table = format_table(
            columns,
            rows,
            width=width,
            theme=theme,
            primary_floor=longest if opts.full_names else 12,
            separator=True,
        )
        # a blank line separates family blocks; a dashed rule separates the grand total
        marks = ["blank" if index and specs[index - 1].ends_group else "" for index in range(len(specs))]
        marks.append(TOTALS_MARK)
        rule = theme.paint("-" * visible_len(table[0]), "dim")
        lines += interleave_rows(table, marks, header_rows=2, rule=rule)
        if snapshot.teams:
            # a rule of the same weight divides the fleet view from this team's keys
            lines.append(rule)
            for index, team in enumerate(snapshot.teams):
                if index:
                    lines.append("")
                lines += render_team(team, theme, opts)
    if opts.show_gateway and snapshot.aliases is not None:
        lines.append("")
        lines += render_gateway(snapshot.aliases, theme, opts)
    lines.append("")
    if snapshot.errors:
        shown = "; ".join(snapshot.errors[:2])
        if len(snapshot.errors) > 2:
            shown += f" (+{len(snapshot.errors) - 2} more)"
        lines.append(theme.paint(clamp("! partial data: " + shown, width, theme), "yellow"))
    footer = f"{snapshot.source}  ·  {snapshot.local_time()}"
    lines.append(theme.paint(clamp(footer, width, theme), "grey"))
    return "\n".join(lines)


def render_quiet(models: Sequence[ModelUsage]) -> str:
    """Two numbers for scripts: total running and total waiting."""
    return f"{sum(m.running for m in models)} {sum(m.waiting for m in models)}"
