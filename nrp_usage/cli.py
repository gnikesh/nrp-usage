"""``nrp-usage`` command line interface."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import replace

from . import __version__
from .model import SORT_KEYS, Snapshot, Thresholds, filter_models, sort_models
from .prometheus import AuthError, PromError
from .render import RenderOptions, Theme, render_quiet, render_report
from .source import DATASOURCE_UID, GRAFANA_URL, SourceError, resolve_source, token_from_file
from .trend import TrendTracker
from .usage import describe_queries, fetch_snapshot, list_teams

EPILOG = f"""\
examples:
  nrp-usage                     all models, busiest first
  nrp-usage glm qwen            only models matching glm or qwen
  nrp-usage --watch 5           live view, refreshed every 5s
  nrp-usage --gateway           add gateway traffic per model alias (glm-5, qwen3, ...)
  nrp-usage --busy-only         hide idle models
  nrp-usage --json | jq .totals
  nrp-usage -q glm-5            just "<running> <waiting>", for scripts
  nrp-usage -x glm              exit 1 when that model has a queue

environment:
  NRP_TEAMS         comma separated team ids to show an API-key overview for (default: none)
  NRP_PROM_URL        query a Prometheus/Thanos API directly instead of via Grafana
  NRP_GRAFANA_URL     override the Grafana host (default {GRAFANA_URL})
  NRP_DATASOURCE_UID  override the datasource uid (default {DATASOURCE_UID}, or "auto" to discover)
  NRP_TOKEN_FILE      file holding a bearer token for a private datasource
  NRP_TIMEOUT         HTTP timeout in seconds
  NRP_INSECURE=1      skip TLS verification (not recommended)
  NO_COLOR=1          disable colour (also set automatically when stdout is not a tty)
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nrp-usage",
        description="Show live vLLM queue depth on the NRP LLM platform: how many requests each "
        "model is running and how many are waiting.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("patterns", nargs="*", metavar="PATTERN",
                        help="case-insensitive model-name filters")
    parser.add_argument("-V", "--version", action="version", version=f"nrp-usage {__version__}")

    view = parser.add_argument_group("view")
    view.add_argument("-w", "--watch", nargs="?", const=5.0, type=float, metavar="SECONDS",
                      help="refresh continuously (default every 5s)")
    view.add_argument("-b", "--busy-only", action="store_true", help="hide idle models")
    view.add_argument("-s", "--sort", choices=SORT_KEYS, default="load",
                      help="order models within each family (default: load); families "
                           "themselves always hold a fixed alphabetical block order")
    view.add_argument("--no-group", dest="group", action="store_false",
                      help="one flat list, no family subtotals")
    view.add_argument("--org", action="store_true",
                      help="keep the org prefix in model names (Qwen/Qwen3.8-27B)")
    view.add_argument("-g", "--gateway", action="store_true",
                      help="also show gateway traffic per model alias")
    view.add_argument("--reason", action="store_true",
                      help="split the queue by vLLM wait reason (capacity/deferred)")
    view.add_argument("--no-contention", dest="contention", action="store_false",
                      help="hide the WAIT MAX and QUEUED columns (recent contention; RUN/WAIT "
                           "alone is a snapshot and reads quieter than the dashboard curve)")
    view.add_argument("--full-names", action="store_true", help="never elide model names")
    view.add_argument("--width", type=int, default=0, help="force output width")
    view.add_argument("--color", dest="color", action="store_true", default=None, help="force colour on")
    view.add_argument("--no-color", dest="color", action="store_false", help="force colour off")
    view.add_argument("--json", action="store_true", help="machine-readable output")
    view.add_argument("-q", "--quiet", action="store_true", help="print only '<running> <waiting>'")

    data = parser.add_argument_group("data")
    data.add_argument("--window", default=os.environ.get("NRP_WINDOW", "15m"), metavar="DURATION",
                      help="lookback for windowed figures: WAIT MAX and QUEUED cover it, "
                           "AVG WAIT the mean over it; RUN/WAIT, NODES and KV CACHE stay "
                           "instantaneous (default: 15m, which matches the dashboard's "
                           "now-15m view; use 5m for a tighter recent look)")
    data.add_argument("--list-teams", action="store_true",
                      help="list team ids with traffic in --window, busiest first, then exit "
                           "(positional PATTERNs filter the list)")
    data.add_argument("--keys", type=int, default=8, metavar="N", dest="max_keys",
                      help="rows of API keys to list per team in the overview (default: 8)")
    data.add_argument("--history", type=int, default=3, metavar="N", dest="history_slots",
                      help="IN/OUT columns per team row, one per --watch frame, oldest on the "
                           "left and newest on the right (default: 3)")
    data.add_argument("--team", action="append", default=[], metavar="TEAM_ID",
                      help="add a per-team API-key overview below the table, and scope --gateway "
                           "figures to it (repeatable; falls back to $NRP_TEAMS, and to nothing if "
                           "that is unset; use --list-teams to find your id)")
    data.add_argument("--no-teams", dest="teams", action="store_false",
                      help="omit the per-team API-key overview even if $NRP_TEAMS is set")
    data.add_argument("--token", action="append", default=[], metavar="ALIAS",
                      help="restrict gateway figures to a token_alias (repeatable)")
    data.add_argument("--warn-waiting", type=int, default=1, metavar="N",
                      help="queue depth that turns a row yellow (default: 1)")
    data.add_argument("--crit-waiting", type=int, default=8, metavar="N",
                      help="queue depth that turns a row red (default: 8)")

    conn = parser.add_argument_group("connection")
    conn.add_argument("--prom-url", help="Prometheus/Thanos API base URL, bypassing Grafana")
    conn.add_argument("--grafana-url", help=f"Grafana host (default: {GRAFANA_URL})")
    conn.add_argument("--dashboard-uid", help="dashboard to discover the datasource from")
    conn.add_argument("--ds-uid", help="datasource uid, or 'auto' to read it from the dashboard")
    conn.add_argument("--token-file", help="read a bearer token from this file")
    conn.add_argument("--timeout", type=float, default=float(os.environ.get("NRP_TIMEOUT", "20")),
                      help="HTTP timeout in seconds (default: 20)")
    conn.add_argument("--insecure", action="store_true",
                      default=os.environ.get("NRP_INSECURE") == "1",
                      help="skip TLS certificate verification")
    conn.add_argument("--explain", action="store_true", help="print the PromQL that would run, then exit")
    conn.add_argument("-v", "--verbose", action="count", default=0, help="report the resolved endpoint")

    script = parser.add_argument_group("scripting")
    script.add_argument("-x", "--exit-if-busy", action="store_true",
                        help="exit 1 if anything matching is queued")
    return parser


def _teams_of_interest(args: argparse.Namespace) -> list[str]:
    """Teams to summarise below the table, or none.

    Empty by default: the fleet view stands on its own, and showing a stranger someone
    else's API keys would be wrong. Set ``NRP_TEAMS`` or pass ``--team`` to opt in.
    """
    if not args.teams:
        return []
    if args.team:
        return list(args.team)
    env = os.environ.get("NRP_TEAMS", "")
    return [t.strip() for t in env.split(",") if t.strip()]


def _read_token(args: argparse.Namespace) -> str | None:
    path = args.token_file or os.environ.get("NRP_TOKEN_FILE")
    return token_from_file(path) if path else None


def _fetch(args: argparse.Namespace, tracker: TrendTracker | None = None) -> Snapshot:
    source = resolve_source(
        prom_url=args.prom_url,
        grafana_url=args.grafana_url,
        datasource_uid=args.ds_uid,
        dashboard_uid=args.dashboard_uid,
        token=_read_token(args),
        timeout=args.timeout,
        insecure=args.insecure,
    )
    if args.verbose:
        print(f"source: {source.description}", file=sys.stderr)
    client = source.make_client(timeout=args.timeout, insecure=args.insecure)
    return fetch_snapshot(
        client,
        source=source.label,
        detail=source.description,
        window=args.window,
        gateway=args.gateway,
        team_ids=args.team,
        token_aliases=args.token,
        contention=args.contention,
        overview_teams=_teams_of_interest(args),
        tracker=tracker,
        history_slots=max(1, args.history_slots),
    )


def _select(snapshot: Snapshot, args: argparse.Namespace) -> list:
    return sort_models(
        filter_models(snapshot.models, args.patterns, busy_only=args.busy_only),
        args.sort,
    )


def _total_failure(snapshot: Snapshot) -> str | None:
    """The message to report when the report cannot answer the question at all."""
    if snapshot.models:
        return None
    if snapshot.errors:
        return f"could not read any metrics from {snapshot.source}: {snapshot.errors[0]}"
    # An empty-but-successful response is the dangerous case: rendering it as
    # "0 running, 0 waiting, no queues" would read as an idle fleet during an outage.
    return (
        f"{snapshot.source} answered but returned no vLLM series at all. "
        "Either the serving fleet is scaled to zero or the metrics backend is not answering "
        "- this is not the same as the models being idle."
    )


def _frame(snapshot: Snapshot, models, args: argparse.Namespace, theme: Theme, opts: RenderOptions) -> str:
    if args.json:
        payload = replace(snapshot, models=list(models))
        return json.dumps(payload.to_dict(), indent=None if args.quiet else 2)
    if args.quiet:
        return render_quiet(models)
    report = render_report(snapshot, models, theme, opts)
    if args.watch:
        report += "\n" + theme.paint(f"refreshing every {args.watch:g}s · ctrl-c to quit", "grey")
    return report


def _print(frame: str) -> None:
    sys.stdout.write(frame + "\n")
    sys.stdout.flush()


def _run_watch(args: argparse.Namespace, theme: Theme, opts: RenderOptions) -> int:
    interval = float(args.watch)
    tty = sys.stdout.isatty()
    first = True
    tracker = TrendTracker(depth=max(1, args.history_slots) + 2)
    # +2 frames: n columns need n intervals, and an interval needs two frames, so the leftmost
    # column still has a predecessor to compare against
    while True:
        started = time.monotonic()
        try:
            snapshot = _fetch(args, tracker)
            failure = _total_failure(snapshot)
            if failure:
                tracker.reset()  # nothing useful was sampled; do not stretch the baseline
                frame = theme.paint(f"nrp-usage: {failure}", "red")
            else:
                frame = _frame(snapshot, _select(snapshot, args), args, theme, opts)
        except (PromError, SourceError) as exc:
            # Keep the display alive: a blip should not end a watch session.
            tracker.reset()  # the gap is unknown, so the next frame restarts the baseline
            frame = theme.paint(f"nrp-usage: {exc}", "red")
        if tty and not first:
            sys.stdout.write("\x1b[H\x1b[2J")
        _print(frame)
        first = False
        time.sleep(max(0.1, interval - (time.monotonic() - started)))


def _cmd_list_teams(args: argparse.Namespace) -> int:
    """Print the team ids that are actually running traffic, for --team discovery."""
    source = resolve_source(
        prom_url=args.prom_url,
        grafana_url=args.grafana_url,
        datasource_uid=args.ds_uid,
        dashboard_uid=args.dashboard_uid,
        token=_read_token(args),
        timeout=args.timeout,
        insecure=args.insecure,
    )
    client = source.make_client(timeout=args.timeout, insecure=args.insecure)
    try:
        teams = list_teams(client, window=args.window, patterns=args.patterns)
    except (PromError, SourceError) as exc:
        print(f"nrp-usage: {exc}", file=sys.stderr)
        return 2
    theme = Theme.detect(sys.stdout, force_color=args.color is True, no_color=args.color is False)
    if not teams:
        hint = f" matching {' '.join(args.patterns)}" if args.patterns else ""
        print(theme.paint(f"no teams with traffic in the last {args.window}{hint}", "yellow"))
        return 0
    width = max(len(team) for team, _ in teams)
    print(theme.paint(f"Teams with traffic in the last {args.window}", "bold", "cyan"))
    for team, rate in teams:
        print(f"  {team.ljust(width)}   {theme.paint(f'{rate:8.1f}', 'bold')} req/min")
    print()
    print(theme.paint(f"use one of them:  nrp-usage --team {teams[0][0]}", "grey"))
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `nrp-usage | head` stays quiet

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.watch is not None and args.watch <= 0:
        parser.error("--watch needs a positive number of seconds")
    if args.warn_waiting > args.crit_waiting:
        parser.error("--warn-waiting must not exceed --crit-waiting")

    if args.list_teams:
        return _cmd_list_teams(args)

    if args.explain:
        try:
            _print(describe_queries(
                window=args.window,
                gateway=args.gateway,
                team_ids=args.team,
                token_aliases=args.token,
                contention=args.contention,
                overview_teams=_teams_of_interest(args),
            ))
        except ValueError as exc:
            parser.error(str(exc))
        return 0

    theme = Theme.detect(sys.stdout, force_color=args.color is True, no_color=args.color is False)
    opts = RenderOptions(
        thresholds=Thresholds(
            waiting_warn=args.warn_waiting,
            waiting_crit=max(args.crit_waiting, args.warn_waiting),
        ),
        show_gateway=args.gateway,
        show_reason=args.reason,
        full_names=args.full_names,
        group=args.group,
        show_org=args.org,
        show_contention=args.contention,
        sort_key=args.sort,
        window=args.window,
        max_keys=max(1, args.max_keys),
        history_slots=max(1, args.history_slots),
        history_reserved=bool(args.watch),
        width=args.width,
    )

    try:
        if args.watch:
            return _run_watch(args, theme, opts)
        snapshot = _fetch(args)
        failure = _total_failure(snapshot)
        if failure:
            print(f"nrp-usage: {failure}", file=sys.stderr)
            if args.verbose and snapshot.detail:
                print(f"  endpoint: {snapshot.detail}", file=sys.stderr)
            return 2
        models = _select(snapshot, args)
        _print(_frame(snapshot, models, args, theme, opts))
        if args.exit_if_busy:
            return 1 if sum(m.waiting for m in models) > 0 else 0
        return 0
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return 130
    except (PromError, SourceError, AuthError) as exc:
        print(f"nrp-usage: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        os._exit(0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
