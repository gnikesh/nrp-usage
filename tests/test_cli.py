"""Unit tests for argument handling, output modes and exit codes.

The metrics fetch is replaced with a canned snapshot, so these exercise the CLI
end to end without a network.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
from datetime import datetime, timezone
from unittest import mock

from nrp_usage import cli as cli_module
from nrp_usage.cli import build_parser
from nrp_usage.model import KeyUsage, ModelUsage, Snapshot, TeamOverview
from nrp_usage.prometheus import PromError
from nrp_usage.render import RenderOptions
from nrp_usage.source import SourceError
from nrp_usage.usage import build_queries

ROWS = [
    ModelUsage("Qwen/Qwen3.8-27B", running=40, waiting=0, replicas=3, busy_replicas=3, kv_cache=0.44,
               queue_avg=0.05, waiting_peak=2, queued_share=0.15),
    ModelUsage("Inferact/GLM-5.3-NVFP4", running=10, waiting=6, replicas=2, busy_replicas=2,
               kv_cache=0.9, queue_avg=4.0, waiting_capacity=6),
    ModelUsage(
        "google/gemma-4-12B-it-qat-w4a16-ct", running=0, waiting=0, replicas=5, busy_replicas=0, kv_cache=0.0
    ),
]


def canned_snapshot(**overrides) -> Snapshot:
    base = {
        "source": "unit test source",
        "fetched_at": datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc),
        "models": [ModelUsage(**vars(m)) for m in ROWS],
        "aliases": None,
        "errors": [],
        "window": "5m",
    }
    base.update(overrides)
    return Snapshot(**base)


@contextlib.contextmanager
def _silent():
    """Swallow stray prints from argparse while tests exercise error paths."""
    with contextlib.redirect_stderr(io.StringIO()):
        yield


def run(argv) -> tuple[int, str, str]:
    """Run the CLI with the network fetch stubbed out."""

    def fake_fetch(args):
        return canned_snapshot()

    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(cli_module, "_fetch", side_effect=fake_fetch), \
         contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli_module.main(argv)
    return code, out.getvalue(), err.getvalue()


def run_error(argv, exc: Exception) -> tuple[int, str, str]:
    """Run the CLI when the fetch raises."""
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(cli_module, "_fetch", side_effect=exc), \
         contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli_module.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestArguments(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(
            (args.patterns, args.sort, args.window, args.gateway, args.watch),
            ([], "load", "15m", False, None),
        )
        # 15m so the windowed figures cover the same span as the dashboard's now-15m view
        self.assertTrue(args.group)
        self.assertTrue(args.contention)
        self.assertFalse(args.org)

    def test_contention_and_group_can_be_turned_off(self):
        args = build_parser().parse_args(["--no-contention", "--no-group"])
        self.assertFalse(args.contention)
        self.assertFalse(args.group)

    def test_patterns_are_positional(self):
        self.assertEqual(build_parser().parse_args(["glm", "qwen"]).patterns, ["glm", "qwen"])

    def test_watch_default_interval(self):
        self.assertEqual(build_parser().parse_args(["--watch"]).watch, 5.0)
        self.assertEqual(build_parser().parse_args(["-w", "2"]).watch, 2.0)

    def test_bad_sort_choice(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            build_parser().parse_args(["--sort", "vibes"])

    def test_reversed_thresholds_are_rejected(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as ctx:
            cli_module.main(["--warn-waiting", "9", "--crit-waiting", "2"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("must not exceed", err.getvalue())

    def test_non_positive_watch_is_rejected(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as ctx:
            cli_module.main(["--watch", "0"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("positive", err.getvalue())

    def test_help_lists_examples(self):
        parser = build_parser()
        text = parser.format_help()
        self.assertIn("nrp-usage --watch 5", text)
        self.assertIn("NRP_PROM_URL", text)


class TestOutputModes(unittest.TestCase):
    def test_default_table(self):
        code, out, _ = run([])
        self.assertEqual(code, 0)
        self.assertIn("NRP model usage", out)
        self.assertIn("Qwen3.8-27B", out)  # org prefix is dropped by default
        self.assertIn("40/0", out)  # running and waiting share a column
        self.assertIn("WAIT MAX", out)  # recent contention, so dashboard figures reconcile
        self.assertIn("2", out)
        self.assertIn("15%", out)  # queued share
        self.assertIn("congested", out)  # GLM has 6 waiting
        self.assertIn("TOTAL", out)

    def test_colour_off_by_default_when_piped(self):
        _, out, _ = run([])
        self.assertNotIn("\x1b[", out)

    def test_json(self):
        code, out, _ = run(["--json"])
        payload = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(payload["totals"]["running"], 50)
        self.assertEqual(len(payload["models"]), 3)
        self.assertEqual(payload["window"], "5m")
        self.assertIn("Qwen/Qwen3.8-27B", [m["name"] for m in payload["models"]])

    def test_json_honours_filters(self):
        _, out, _ = run(["--json", "glm"])
        payload = json.loads(out)
        self.assertEqual([m["name"] for m in payload["models"]], ["Inferact/GLM-5.3-NVFP4"])
        self.assertEqual(payload["totals"]["waiting"], 6)

    def test_quiet(self):
        code, out, _ = run(["-q"])
        self.assertEqual((code, out.strip()), (0, "50 6"))

    def test_quiet_json_is_compact(self):
        _, out, _ = run(["--json", "-q"])
        self.assertEqual(len(out.strip().splitlines()), 1)

    def test_busy_only_hides_idle_models(self):
        _, out, _ = run(["--json", "-b"])
        names = [m["name"] for m in json.loads(out)["models"]]
        self.assertNotIn("google/gemma-4-12B-it-qat-w4a16-ct", names)

    def test_sort_by_waiting(self):
        _, out, _ = run(["--json", "-s", "waiting"])
        names = [m["name"] for m in json.loads(out)["models"]]
        self.assertEqual(names[0], "Inferact/GLM-5.3-NVFP4")

    def test_gateway_flag_reaches_the_query(self):
        with mock.patch.object(cli_module, "_fetch", return_value=canned_snapshot()) as fetch:
            with contextlib.redirect_stdout(io.StringIO()):
                cli_module.main(["--gateway", "--team", "t1", "--window", "30s"])
        # _fetch is called with the parsed namespace; confirm the options landed there
        args = fetch.call_args.args[0]
        self.assertTrue(args.gateway)
        self.assertEqual(args.team, ["t1"])
        self.assertEqual(args.window, "30s")

    def test_explain_does_not_fetch(self):
        with mock.patch.object(cli_module, "_fetch", side_effect=AssertionError("should not fetch")):
            code, out, _ = run(["--explain"])
        self.assertEqual(code, 0)
        self.assertIn("vllm:num_requests_running", out)
        self.assertNotIn("alias_rpm", out)

    def test_explain_with_gateway(self):
        _, out, _ = run(["--explain", "--gateway"])
        self.assertIn("gen_ai_original_model", out)


class TestExitCodes(unittest.TestCase):
    def test_zero_when_nothing_queued(self):
        quiet = [ModelUsage("a", running=1), ModelUsage("b", running=2)]
        with mock.patch.object(cli_module, "_fetch", return_value=canned_snapshot(models=quiet)):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_module.main(["-x"]), 0)

    def test_one_when_queued(self):
        code, _, _ = run(["-x"])
        self.assertEqual(code, 1)

    def test_filter_scopes_the_exit_code(self):
        code, _, _ = run(["-x", "gemma"])  # that model is idle
        self.assertEqual(code, 0)

    def test_prom_error_exits_two(self):
        code, _, err = run_error([], PromError("no route to host"))
        self.assertEqual(code, 2)
        self.assertIn("no route to host", err)

    def test_source_error_exits_two(self):
        code, _, err = run_error([], SourceError("could not discover"))
        self.assertEqual(code, 2)
        self.assertIn("could not discover", err)

    def test_every_query_failing_is_a_failure_not_an_empty_report(self):
        empty = canned_snapshot(models=[], errors=["running: HTTP 502", "waiting: HTTP 502"])
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(cli_module, "_fetch", return_value=empty), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_module.main([])
        self.assertEqual(code, 2)
        self.assertIn("could not read any metrics", err.getvalue())
        self.assertEqual(out.getvalue(), "")

    def test_no_series_at_all_is_not_reported_as_idle(self):
        # a 200-with-empty-result (backend hiccup) must never print "no queues"
        silent = canned_snapshot(models=[], errors=[])
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(cli_module, "_fetch", return_value=silent), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_module.main([])
        self.assertEqual(code, 2)
        message = err.getvalue()
        self.assertIn("no vLLM series", message)
        self.assertIn("not the same as the models being idle", message)
        self.assertNotIn("no queues", message)

    def test_quiet_mode_also_refuses_to_claim_zero(self):
        silent = canned_snapshot(models=[], errors=[])
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(cli_module, "_fetch", return_value=silent), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_module.main(["-q"])
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue(), "")

    def test_partial_data_still_succeeds(self):
        partial = canned_snapshot(errors=["kv_cache: HTTP 502"])
        with mock.patch.object(cli_module, "_fetch", return_value=partial):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = cli_module.main([])
        self.assertEqual(code, 0)
        self.assertIn("partial data", out.getvalue())

    def test_filters_that_match_nothing_are_not_an_error(self):
        with mock.patch.object(cli_module, "_fetch", return_value=canned_snapshot()):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = cli_module.main(["no-such-model"])
        self.assertEqual(code, 0)
        self.assertIn("no models matched", out.getvalue())

    def test_keyboard_interrupt_exits_130(self):
        code, _out, _err = run_error([], KeyboardInterrupt)
        self.assertEqual(code, 130)

    def test_help_exits_zero(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            cli_module.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_version(self):
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
            cli_module.main(["--version"])
        self.assertIn("nrp-usage", out.getvalue())


class TestTeamSelection(unittest.TestCase):
    def args(self, argv):
        return build_parser().parse_args(argv)

    def test_default_shows_no_team_at_all(self):
        # a fresh install must not surface someone else's API keys
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cli_module._teams_of_interest(self.args([])), [])

    def test_default_issues_no_team_queries(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            queries = build_queries(window="15m", overview_teams=cli_module._teams_of_interest(self.args([])))
        self.assertFalse([k for k in queries if k.startswith("team_")])

    def test_env_var_enables_the_section(self):
        with mock.patch.dict(os.environ, {"NRP_TEAMS": "acme-lab"}, clear=True):
            self.assertEqual(cli_module._teams_of_interest(self.args([])), ["acme-lab"])

    def test_explicit_team_wins(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                cli_module._teams_of_interest(self.args(["--team", "other-team"])),
                ["other-team"],
            )

    def test_env_var_is_honoured_and_trimmed(self):
        with mock.patch.dict(os.environ, {"NRP_TEAMS": "a-team, b-team"}, clear=True):
            self.assertEqual(cli_module._teams_of_interest(self.args([])), ["a-team", "b-team"])

    def test_blank_env_var_means_no_team(self):
        with mock.patch.dict(os.environ, {"NRP_TEAMS": "  , , "}, clear=True):
            self.assertEqual(cli_module._teams_of_interest(self.args([])), [])

    def test_flag_beats_env(self):
        with mock.patch.dict(os.environ, {"NRP_TEAMS": "env-team"}, clear=True):
            self.assertEqual(cli_module._teams_of_interest(self.args(["--team", "cli-team"])), ["cli-team"])

    def test_no_teams_disables_the_section(self):
        with mock.patch.dict(os.environ, {"NRP_TEAMS": "acme-lab"}, clear=True):
            self.assertEqual(cli_module._teams_of_interest(self.args(["--no-teams"])), [])

    def test_teams_reach_the_queries(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            teams = cli_module._teams_of_interest(self.args(["--team", "acme-lab"]))
        queries = build_queries(window="15m", overview_teams=teams)
        self.assertIn("team_rpm[acme-lab]", queries)
        self.assertIn("team_mean[acme-lab]", queries)

    def test_no_team_queries_when_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            teams = cli_module._teams_of_interest(self.args(["--no-teams"]))
        queries = build_queries(window="15m", overview_teams=teams)
        self.assertFalse([k for k in queries if k.startswith("team_")])

    def test_keys_option_flows_to_render_options(self):
        args = self.args(["--keys", "3"])
        self.assertEqual(args.max_keys, 3)

    def test_zero_keys_is_clamped_to_one(self):
        # --keys 0 would hide the whole point of the section
        self.assertEqual(RenderOptions(max_keys=max(1, 0)).max_keys, 1)

    def test_json_output_carries_teams_when_requested(self):
        teams = [TeamOverview("acme-lab", "15m", [KeyUsage("alpha-key", "qwen3", 12.5, 12.58)])]
        with mock.patch.object(cli_module, "_fetch", return_value=canned_snapshot(teams=teams)):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = cli_module.main(["--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["teams"][0]["team_id"], "acme-lab")
        self.assertEqual(payload["teams"][0]["keys"][0]["token_alias"], "alpha-key")

    def test_table_shows_the_team_section_when_a_team_is_set(self):
        teams = [TeamOverview("acme-lab", "15m", [KeyUsage("alpha-key", "qwen3", 12.5, 12.58)])]
        with mock.patch.object(cli_module, "_fetch", return_value=canned_snapshot(teams=teams)):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                cli_module.main(["--team", "acme-lab", "--width", "110"])
        text = out.getvalue()
        self.assertIn("team acme-lab", text)
        self.assertIn("API KEY", text)

    def test_no_team_section_when_no_team_is_configured(self):
        # the fresh-install case: no NRP_TEAMS, no --team, so no keys to show
        with mock.patch.object(cli_module, "_fetch", return_value=canned_snapshot(teams=[])):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = cli_module.main(["--width", "110"])
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("TOTAL", text)
        self.assertNotIn("API KEY", text)
        self.assertNotIn("team acme-lab", text)


class TestListTeamsCli(unittest.TestCase):
    def test_prints_teams_and_a_usable_hint(self):
        with mock.patch.object(cli_module, "list_teams", return_value=[("gamma-lab", 152.8), ("acme-lab", 10.8)]):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = cli_module.main(["--list-teams"])
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("gamma-lab", text)
        self.assertIn("152.8 req/min", text)
        self.assertIn("nrp-usage --team gamma-lab", text)

    def test_filters_are_passed_through(self):
        with mock.patch.object(cli_module, "list_teams", return_value=[]) as listed:
            with contextlib.redirect_stdout(io.StringIO()):
                cli_module.main(["--list-teams", "ksu", "--window", "30s"])
        self.assertEqual(listed.call_args.kwargs["patterns"], ["ksu"])
        self.assertEqual(listed.call_args.kwargs["window"], "30s")

    def test_empty_result_is_not_an_error(self):
        with mock.patch.object(cli_module, "list_teams", return_value=[]):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = cli_module.main(["--list-teams"])
        self.assertEqual(code, 0)
        self.assertIn("no teams with traffic", out.getvalue())

    def test_backend_failure_exits_two(self):
        with mock.patch.object(cli_module, "list_teams", side_effect=PromError("connection refused")):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = cli_module.main(["--list-teams"])
        self.assertEqual(code, 2)
        self.assertIn("connection refused", err.getvalue())

    def test_does_not_render_the_model_table(self):
        with mock.patch.object(cli_module, "list_teams", return_value=[("t", 1.0)]):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                cli_module.main(["--list-teams"])
        self.assertNotIn("RUN/WAIT", out.getvalue())


class TestStripOnlyInWatch(unittest.TestCase):
    """Plain runs get one IN/OUT column; -w gets the multi-frame strip."""

    def team(self, **over):
        base = {"req_per_min": 40.0, "mean_seconds": 0.7, "in_tokens": 62_800, "out_tokens": 9_700}
        base.update(over)
        return KeyUsage("main", "qwen3", **base)

    def test_one_shot_has_a_single_in_out_column(self):
        teams = [TeamOverview("acme-lab", "15m", [self.team()])]
        with mock.patch.object(cli_module, "_fetch", return_value=canned_snapshot(teams=teams)):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                cli_module.main(["--team", "acme-lab", "--width", "120"])
        header = next(line for line in out.getvalue().splitlines() if line.startswith("API KEY"))
        assert "15m IN/OUT" in header, header
        assert "IN/OUT now" not in header, header
        assert header.count("IN/OUT") == 1, header

    def test_watch_reserves_the_frame_columns_from_the_first_tick(self):
        teams = [TeamOverview("acme-lab", "15m", [self.team()])]
        frames = iter([canned_snapshot(teams=teams), KeyboardInterrupt()])

        def fake_fetch(args, tracker=None):
            value = next(frames)
            if not isinstance(value, Snapshot):
                raise value
            return value

        out = io.StringIO()
        with mock.patch.object(cli_module, "_fetch", side_effect=fake_fetch), \
             mock.patch.object(cli_module.time, "sleep"), contextlib.redirect_stdout(out):
            cli_module.main(["--watch", "1", "--team", "acme-lab", "--width", "120"])
        header = next(line for line in out.getvalue().splitlines() if line.startswith("API KEY"))
        assert "IN/OUT -2" in header and "IN/OUT -1" in header and "IN/OUT now" in header, header

    def test_history_flag_widens_the_strip(self):
        teams = [TeamOverview("acme-lab", "15m", [self.team()])]
        frames = iter([canned_snapshot(teams=teams), KeyboardInterrupt()])

        def fake_fetch(args, tracker=None):
            value = next(frames)
            if not isinstance(value, Snapshot):
                raise value
            return value

        out = io.StringIO()
        with mock.patch.object(cli_module, "_fetch", side_effect=fake_fetch), \
             mock.patch.object(cli_module.time, "sleep"), contextlib.redirect_stdout(out):
            cli_module.main(["--watch", "1", "--history", "5", "--team", "acme-lab", "--width", "160"])
        header = next(line for line in out.getvalue().splitlines() if line.startswith("API KEY"))
        assert "IN/OUT -4" in header, header


class TestWatchMode(unittest.TestCase):
    def frames(self, values):
        """A _fetch stand-in that replays snapshots and exceptions in order."""
        pending = iter(values)

        def fake_fetch(args, tracker=None):
            value = next(pending)
            if not isinstance(value, Snapshot):
                raise value
            return value

        return fake_fetch

    def run_watch(self, values, argv=("--watch", "1", "--no-color")):
        out = io.StringIO()
        with mock.patch.object(cli_module, "_fetch", side_effect=self.frames(values)), \
             mock.patch.object(cli_module.time, "sleep"), contextlib.redirect_stdout(out):
            code = cli_module.main(list(argv))
        return code, out.getvalue()

    def test_renders_each_frame_then_stops_on_interrupt(self):
        code, rendered = self.run_watch([canned_snapshot(), canned_snapshot(), KeyboardInterrupt()])
        self.assertEqual(code, 130)
        self.assertEqual(rendered.count("NRP model usage"), 2)
        self.assertIn("refreshing every 1s", rendered)

    def test_a_blip_does_not_end_the_session(self):
        _code, rendered = self.run_watch([PromError("timeout"), canned_snapshot(), KeyboardInterrupt()])
        self.assertIn("timeout", rendered)               # the failure was shown
        self.assertEqual(rendered.count("NRP model usage"), 1)  # and the next frame still arrived


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
