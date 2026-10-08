"""Unit tests for the table layout, colour handling and width fitting."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from nrp_usage.model import KeyUsage, ModelUsage, Snapshot, TeamOverview
from nrp_usage.render import (
    Column,
    RenderOptions,
    RowSpec,
    Theme,
    cell_for,
    elide,
    format_duration,
    format_table,
    plan_columns,
    plan_rows,
    render_gateway,
    render_quiet,
    render_report,
    render_team,
    strip_ansi,
)
from nrp_usage.trend import Interval, Trend
from nrp_usage.usage import make_snapshot
from tests.fakes import vector


def model(
    name="a/b",
    running=0,
    waiting=0,
    replicas=1,
    busy=0,
    kv=None,
    queue=None,
    cap=0,
    deferred=0,
    waiting_peak=0,
    queued_share=None,
):
    # keyword construction: ModelUsage gains fields over time, positional build silently shifts
    return ModelUsage(
        name=name,
        running=running,
        waiting=waiting,
        replicas=replicas,
        busy_replicas=busy,
        kv_cache=kv,
        queue_avg=queue,
        waiting_peak=waiting_peak,
        queued_share=queued_share,
        waiting_capacity=cap,
        waiting_deferred=deferred,
    )


def snapshot(*models, window="15m", errors=(), teams=None):
    return Snapshot(
        source="test source",
        fetched_at=datetime(2026, 10, 6, 9, 30, tzinfo=timezone.utc),
        models=list(models),
        teams=list(teams) if teams is not None else [],
        errors=list(errors),
        window=window,
    )


class TestTheme(unittest.TestCase):
    def test_plain_theme_does_not_emit_escapes(self):
        theme = Theme(color=False)
        self.assertEqual(theme.paint("x", "red"), "x")

    def test_colour_theme_wraps(self):
        theme = Theme(color=True)
        self.assertEqual(theme.paint("x", "red"), "\x1b[31mx\x1b[0m")

    def test_none_styles_are_skipped(self):
        theme = Theme(color=True)
        self.assertEqual(theme.paint("x", None), "x")

    def test_gauge(self):
        self.assertEqual(Theme(color=False, unicode=True).gauge(0.5, 8), "████····")
        self.assertEqual(Theme(color=False, unicode=False).gauge(0.5, 8), "####....")
        self.assertEqual(Theme(color=False).gauge(2.0, 4), "████")
        self.assertEqual(Theme(color=False).gauge(-1.0, 4), "····")

    def test_strip_ansi(self):
        self.assertEqual(strip_ansi("\x1b[31mred\x1b[0m plain"), "red plain")
        self.assertEqual(strip_ansi("no codes"), "no codes")
        self.assertEqual(strip_ansi("\x1b[broken"), "")


class TestFormatters(unittest.TestCase):
    def test_durations(self):
        cases = [
            (None, "-"),
            (0.0, "0ms"),
            (0.0005, "<1ms"),
            (0.054, "54ms"),
            (0.999, "999ms"),
            (2.5, "2.50s"),
            (119.0, "119.00s"),
            (120, "2m00s"),
            (3700, "61m40s"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(format_duration(value), expected)

    def test_elide_middle(self):
        self.assertEqual(elide("Qwen/Qwen3.8-27B", 20, Theme(unicode=True)), "Qwen/Qwen3.8-27B")
        short = elide("deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", 20, Theme(unicode=True))
        self.assertEqual(len(short), 20)
        self.assertTrue(short.startswith("deep"))
        self.assertTrue(short.endswith("Exp"))
        self.assertIn("…", short)

    def test_elide_ascii_fallback(self):
        self.assertIn("~", elide("abcdefghijklmnop", 8, Theme(unicode=False)))

    def test_elide_edge_sizes(self):
        self.assertEqual(elide("abc", 0, Theme()), "")
        self.assertEqual(elide("abc", 1, Theme()), "…")
        # at two columns the tail (the part that identifies the model) wins
        self.assertEqual(elide("abcd", 2, Theme()), "…d")


class TestColumnPlanning(unittest.TestCase):
    def cols(self, models, opts=None, width=160):
        opts = opts or RenderOptions()
        return [c.key for c in plan_columns(plan_rows(models, opts), opts, width)]

    def test_wide_terminal_keeps_everything(self):
        models = [model("deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", kv=0.5)]
        self.assertEqual(self.cols(models), ["name", "runwait", "waitmax", "queued", "nodes", "kv", "queue", "status"])

    def test_narrow_terminal_drops_optional_columns_right_to_left(self):
        models = [model("deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", kv=0.5)]
        self.assertEqual(
            self.cols(models, width=120),
            ["name", "runwait", "waitmax", "queued", "nodes", "kv", "queue", "status"],
        )
        self.assertEqual(
            self.cols(models, width=104), ["name", "runwait", "waitmax", "queued", "nodes", "kv", "status"]
        )
        self.assertEqual(self.cols(models, width=88), ["name", "runwait", "waitmax", "queued", "nodes", "status"])
        self.assertEqual(self.cols(models, width=80), ["name", "runwait", "waitmax", "queued", "status"])
        cramped = self.cols(models, width=60)
        self.assertEqual(cramped, ["name", "runwait", "status"])
        # what always survives is the answer itself
        self.assertTrue({"name", "runwait", "status"} <= set(cramped))

    def test_contention_outlives_the_cosmetic_columns_when_squeezing(self):
        # WAIT MAX / QUEUED sit left of NODES/KV so they are dropped last: they are what
        # reconciles the snapshot with the dashboard, so they survive a narrow terminal
        models = [model("deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", kv=0.5)]
        self.assertEqual(self.cols(models, width=88), ["name", "runwait", "waitmax", "queued", "nodes", "status"])

    def test_run_and_wait_share_one_column(self):
        cols = self.cols([model("a/b", running=3, waiting=1)])
        self.assertIn("runwait", cols)
        self.assertNotIn("running", cols)
        self.assertNotIn("waiting", cols)

    def test_group_labels_count_toward_the_name_column(self):
        # a subtotal header like "Qwen (12 models)" must not force other columns out
        models = [model(f"org/Qwen3.8-{i}") for i in range(12)]
        self.assertEqual(self.cols(models), ["name", "runwait", "waitmax", "queued", "nodes", "kv", "queue", "status"])

    def test_contention_columns_can_be_hidden(self):
        models = [model("Qwen/Qwen3.8-27B", running=1)]
        self.assertIn("waitmax", self.cols(models))
        self.assertIn("queued", self.cols(models))
        hidden = self.cols(models, RenderOptions(show_contention=False))
        self.assertNotIn("waitmax", hidden)
        self.assertNotIn("queued", hidden)
        queued = [model(waiting=2, cap=2)]
        self.assertNotIn("reason", self.cols(queued))
        self.assertIn("reason", self.cols(queued, RenderOptions(show_reason=True)))

    def test_reason_column_appears_for_any_queue(self):
        # waiting_by_reason can lag num_requests_waiting, so do not gate on it
        self.assertIn("reason", self.cols([model(waiting=1)], RenderOptions(show_reason=True)))

    def test_reason_column_hidden_when_nothing_queued(self):
        self.assertNotIn("reason", self.cols([model(running=1)], RenderOptions(show_reason=True)))


class TestCells(unittest.TestCase):
    theme = Theme(color=False)

    def cell(self, row, key, theme=None):
        return cell_for(RowSpec(row, row.short_name), key, theme or self.theme, RenderOptions())

    def test_idle_row(self):
        row = model("a/b", replicas=3)
        self.assertEqual(self.cell(row, "status"), "idle")
        self.assertEqual(self.cell(row, "runwait"), "0/0")
        self.assertEqual(self.cell(row, "kv"), "n/a")
        self.assertEqual(self.cell(row, "queue"), "-")
        self.assertEqual(self.cell(row, "nodes"), "0/3")

    def test_run_wait_pair_format(self):
        self.assertEqual(self.cell(model("a/b", running=12, waiting=3), "runwait"), "12/3")

    def test_run_wait_queue_part_is_coloured(self):
        colored = self.cell(model("a/b", running=12, waiting=3), "runwait", Theme(color=True))
        self.assertEqual(strip_ansi(colored), "12/3")
        # the waiting count carries the warning colour, the running count stays plain bold
        self.assertIn("\x1b[1m12\x1b[0m", colored)
        self.assertIn("\x1b[33m", colored.split("12")[1])

    def test_kv_gauge_appears_in_cell(self):
        text = self.cell(model(kv=0.5), "kv")
        self.assertIn("████····", text)
        self.assertIn("50%", text)

    def test_queue_reason_split(self):
        self.assertEqual(self.cell(model("a/b", waiting=5, cap=4, deferred=1), "reason"), "4/1")

    def test_wait_max_cell(self):
        self.assertEqual(self.cell(model("a/b", waiting_peak=49), "waitmax"), "49")
        self.assertEqual(self.cell(model("a/b", waiting_peak=0), "waitmax"), "0")

    def test_queued_cell_is_a_percentage(self):
        self.assertEqual(self.cell(model("a/b", queued_share=0.333), "queued"), " 33%")
        self.assertEqual(self.cell(model("a/b", queued_share=0.0), "queued"), "  0%")
        self.assertEqual(self.cell(model("a/b", queued_share=None), "queued"), "-")

    def test_contention_is_blank_on_group_rows_because_it_cannot_be_derived(self):
        member = model("Qwen/Qwen3.8-27B", waiting_peak=49, queued_share=0.33)
        spec = RowSpec(model("Qwen/Qwen3.8-27B"), "Qwen (1 models)", bold=True, is_group=True)
        member_spec = RowSpec(member, member.short_name)
        self.assertEqual(cell_for(spec, "waitmax", self.theme, RenderOptions()), "-")
        self.assertEqual(cell_for(spec, "queued", self.theme, RenderOptions()), "-")
        self.assertEqual(cell_for(member_spec, "waitmax", self.theme, RenderOptions()), "49")
        self.assertEqual(cell_for(member_spec, "queued", self.theme, RenderOptions()), " 33%")

    def test_name_uses_the_given_label(self):
        row = model("Qwen/Qwen3.8-27B", running=3)
        spec = RowSpec(row, "Qwen3.8-27B")
        self.assertEqual(strip_ansi(cell_for(spec, "name", self.theme, RenderOptions())), "Qwen3.8-27B")

    def test_verdict_colours_the_name(self):
        colored = Theme(color=True)
        self.assertIn("\x1b[90m", self.cell(model("idle/model"), "name", colored))
        self.assertNotIn("\x1b[", self.cell(model("busy/model", running=3), "name", colored))


class TestTable(unittest.TestCase):
    def test_columns_align_without_colour(self):
        columns = [Column("name", "MODEL", align="left"), Column("running", "RUN", fixed=5)]
        rows = [["short", "1"], ["a-much-longer-model", "22"]]
        lines = format_table(columns, rows, width=40, theme=Theme(color=False))
        header, first, second = lines

        def right_edge(line: str, token: str) -> int:
            return line.index(token) + len(token)

        self.assertEqual(right_edge(header, "RUN"), right_edge(first, "1"))
        self.assertEqual(right_edge(header, "RUN"), right_edge(second, "22"))
        self.assertTrue(header.startswith("MODEL"))
        self.assertTrue(all(len(line) <= 40 for line in lines))

    def test_coloured_output_lays_out_exactly_like_plain(self):
        # guards against padding that counts ANSI bytes as width
        models = [
            model("deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", running=4, waiting=9,
                  replicas=2, busy=2, kv=0.93, queue=12.0),
            model("Qwen/Qwen3.8-27B", running=40, replicas=3, busy=3, kv=0.44, queue=0.05),
            model("google/gemma-4-12B-it-qat-w4a16-ct", replicas=5),
        ]
        opts = RenderOptions(width=120)
        plain = render_report(snapshot(*models), models, Theme(color=False), opts)
        colored = render_report(snapshot(*models), models, Theme(color=True), opts)
        self.assertEqual(
            [strip_ansi(line) for line in plain.splitlines()],
            [strip_ansi(line) for line in colored.splitlines()],
        )
        self.assertIn("\x1b[", colored)

    def test_lines_have_no_trailing_whitespace(self):
        models = [model("a/b", running=1), model("c/d")]
        rendered = render_report(snapshot(*models), models, Theme(color=False), RenderOptions(width=100))
        for line in rendered.splitlines():
            self.assertEqual(line, line.rstrip(), f"trailing space: {line!r}")

    def test_output_fits_the_requested_width(self):
        theme = Theme(color=False)
        long_name = "very-long-organisation/Model-Name-With-A-Rather-Long-Tail-v12345"
        models = [model(long_name, running=3, waiting=1, kv=0.8)]
        for width in (72, 90, 120, 200):
            rendered = render_report(snapshot(*models), models, theme, RenderOptions(width=width))
            for line in rendered.splitlines():
                self.assertLessEqual(len(strip_ansi(line)), width, f"width={width} line={line!r}")

    def test_long_names_are_elided_not_wrapped(self):
        theme = Theme(color=False)
        long_name = "very-long-organisation/Model-Name-With-A-Rather-Long-Tail-v12345"
        models = [model(long_name, running=3, kv=0.8)]
        rendered = render_report(snapshot(*models), models, theme, RenderOptions(width=80))
        body = [line for line in rendered.splitlines() if "Tail" in line or "…" in line]
        self.assertTrue(body)
        self.assertNotIn(long_name, rendered)

    def test_names_shrink_before_the_status_word_is_cut(self):
        # the name column is the one that elides cleanly, so it takes the strain
        theme = Theme(color=False)
        models = [model("deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", running=4, waiting=9, queue=1.0)]
        for width in (60, 66, 72, 80):
            rendered = render_report(snapshot(*models), models, theme, RenderOptions(width=width))
            self.assertNotIn("act\n", rendered, f"status truncated at width={width}")
            self.assertIn("congested", rendered, f"status lost at width={width}")
            for line in rendered.splitlines():
                self.assertLessEqual(len(strip_ansi(line)), width, f"width={width} line={line!r}")

    def test_full_names_option_never_elides(self):
        theme = Theme(color=False)
        name = "org/Model-With-A-Long-Name-For-Sure"
        models = [model(name, running=1)]
        rendered = render_report(snapshot(*models), models, theme, RenderOptions(width=70, full_names=True))
        # --full-names is about not truncating; the org prefix is --org's business
        self.assertIn("Model-With-A-Long-Name-For-Sure", rendered)
        self.assertNotIn("…", rendered)

    def test_names_are_short_by_default(self):
        theme = Theme(color=False)
        models = [model("Qwen/Qwen3.8-27B", running=1), model("google/gemma-4-31B", running=2)]
        rendered = render_report(snapshot(*models), models, theme, RenderOptions(width=110))
        self.assertIn("Qwen3.8-27B", rendered)
        self.assertNotIn("Qwen/", rendered)
        self.assertNotIn("google/", rendered)

    def test_org_option_restores_prefixes(self):
        theme = Theme(color=False)
        models = [model("Qwen/Qwen3.8-27B", running=1)]
        rendered = render_report(snapshot(*models), models, theme, RenderOptions(width=110, show_org=True))
        self.assertIn("Qwen/Qwen3.8-27B", rendered)

    def test_totals_row_present(self):
        theme = Theme(color=False)
        models = [model("a/b", running=3, waiting=2, replicas=2, busy=1), model("c/d", replicas=1)]
        rendered = render_report(snapshot(*models), models, theme, RenderOptions(width=110))
        totals = [line for line in rendered.splitlines() if line.startswith("TOTAL")]
        self.assertEqual(len(totals), 1)
        self.assertIn("3", totals[0])
        self.assertIn("2", totals[0])
        self.assertIn("1/3", totals[0])

    def test_empty_selection_says_why(self):
        # "no models matched" implies a filter; an empty fleet must not read that way
        no_data = render_report(snapshot(), [], Theme(color=False), RenderOptions(width=100))
        self.assertIn("no models reporting data", no_data)
        models = [model("a/b", running=1)]
        filtered = render_report(snapshot(*models), [], Theme(color=False), RenderOptions(width=100))
        self.assertIn("no models matched", filtered)

    def test_errors_are_surfaced(self):
        theme = Theme(color=False)
        models = [model("a/b", running=1)]
        rendered = render_report(snapshot(*models, errors=["kv_cache: 502"]), models, theme, RenderOptions(width=100))
        self.assertIn("partial data", rendered)
        self.assertIn("kv_cache", rendered)

    def test_more_than_two_errors_are_collapsed(self):
        rendered = render_report(
            snapshot(model("a"), errors=["e1", "e2", "e3", "e4"]),
            [model("a")],
            Theme(color=False),
            RenderOptions(width=100),
        )
        self.assertIn("+2 more", rendered)


class TestGrouping(unittest.TestCase):
    theme = Theme(color=False)

    def fleet(self):
        return [
            model("Qwen/Qwen3.8-27B", running=55, waiting=4, replicas=3, busy=3, kv=0.44, queue=0.6),
            model("Qwen/Qwen3.8-Flash-Next-FP8", running=12, replicas=2, busy=2, kv=0.70, queue=0.05),
            model("Qwen/Qwen3-VL-Embedding-8B", replicas=2),
            model("Inferact/GLM-5.3-NVFP4", running=14, waiting=23, replicas=2, busy=2, kv=0.97, queue=41.5),
            model("google/gemma-4-31B", running=6, replicas=3, busy=2, kv=0.16),
            model("google/gemma-4-12B", replicas=5),
        ]

    def rendered(self, models=None, **overrides):
        opts = RenderOptions(width=120, **overrides)
        models = self.fleet() if models is None else models
        return render_report(snapshot(*models), models, self.theme, opts)

    def lines(self, text):
        return [line for line in text.splitlines() if line.strip()]

    def test_family_headers_and_subtotals(self):
        body = self.rendered()
        self.assertIn("Qwen (3 models)", body)
        self.assertIn("gemma (2 models)", body)
        self.assertIn("  Qwen3.8-27B", body)  # members are indented under their family
        self.assertNotIn("GLM (1 models)", body)  # singletons get no header

    def test_subtotals_sum_running_waiting_and_nodes(self):
        row = next(line for line in self.lines(self.rendered()) if line.startswith("Qwen (3 models)"))
        self.assertIn("67/4", row)  # 55+12+0 running, 4+0+0 waiting
        self.assertIn("5/7", row)  # busy/total replicas

    def test_subtotal_status_reflects_the_worst_member(self):
        row = next(line for line in self.lines(self.rendered()) if line.startswith("Qwen (3 models)"))
        self.assertIn("queued", row)
        kv = next(line for line in self.lines(self.rendered()) if line.startswith("gemma (2 models)"))
        self.assertIn("serving", kv)

    def test_families_hold_a_fixed_alphabetical_order(self):
        first = [line.split()[0] for line in self.lines(self.rendered()) if "(" in line and "models)" in line]
        self.assertEqual(first, sorted(first, key=str.casefold))

    def test_group_order_does_not_move_when_traffic_changes(self):
        quiet = [
            model("Qwen/Qwen3.8-27B", running=1, replicas=1),
            model("Qwen/Qwen3.8-Flash", running=1, replicas=1),
            model("Inferact/GLM-5.3-NVFP4", running=99, waiting=40, replicas=1),
        ]
        busy = [
            model("Inferact/GLM-5.3-NVFP4", running=1, replicas=1),
            model("Qwen/Qwen3.8-Flash", running=80, replicas=1),
            model("Qwen/Qwen3.8-27B", running=80, replicas=1),
        ]
        def headers(text: str) -> list[str]:
            return [line.split()[0] for line in self.lines(text) if "(2 models)" in line]
        self.assertEqual(headers(self.rendered(quiet)), headers(self.rendered(busy)))

    def test_member_order_follows_the_sort_key(self):
        # "name" is a plain lexicographic sort of the displayed label, so '-' (in Qwen3-VL)
        # orders before '.' (in Qwen3.8); it is stable and predictable, not version-aware
        flat = self.lines(self.rendered(sort_key="name"))
        members = [line.strip().split()[0] for line in flat if line.startswith("  Qwen")]
        self.assertEqual(members, ["Qwen3-VL-Embedding-8B", "Qwen3.8-27B", "Qwen3.8-Flash-Next-FP8"])
        heavy = self.lines(self.rendered(sort_key="running"))
        members = [line.strip().split()[0] for line in heavy if line.startswith("  Qwen")]
        self.assertEqual(members[0], "Qwen3.8-27B")

    def test_exact_row_sequence_and_separators(self):
        # golden layout: family header, indented members, blank line, next family,
        # then a dashed rule before the grand total
        models = [
            model("Qwen/Qwen3.8-27B", running=5, replicas=1),
            model("Qwen/Qwen3.8-Flash", running=2, replicas=1),
            model("Inferact/GLM-5.3-NVFP4", running=1, replicas=1),
        ]
        body = render_report(snapshot(*models), models, self.theme, RenderOptions(width=90)).splitlines()
        table = body[body.index("") + 1 :]  # drop the title block

        def kind(line: str) -> str:
            if line.startswith("MODEL"):
                return "header"
            if line.startswith("-"):
                return "rule"
            if "(2 models)" in line:
                return "group"
            if line.startswith("  "):
                return "member"
            if line.startswith("GLM"):
                return "single"
            if line.startswith("TOTAL"):
                return "total"
            return "blank" if not line.strip() else "other"

        self.assertEqual(
            [kind(line) for line in table],
            # GLM sorts first alphabetically, then the Qwen block; TOTAL gets the rule
            ["header", "rule", "single", "blank", "group", "member", "member", "rule", "total", "blank", "other"],
        )

    def test_no_group_is_a_flat_list(self):
        body = self.rendered(group=False)
        self.assertNotIn("models)", body)
        self.assertNotIn("  Qwen3.8-27B", body)
        self.assertIn("Qwen3.8-27B", body)
        lines = body.splitlines()
        header = next(i for i, line in enumerate(lines) if line.startswith("MODEL"))
        total = next(i for i, line in enumerate(lines) if line.startswith("TOTAL"))
        between = lines[header + 2 : total]  # the separator follows the header
        self.assertNotIn("", between, "flat mode should not insert family separators")

    def test_grouped_mode_inserts_one_blank_per_family(self):
        lines = self.rendered().splitlines()
        header = next(i for i, line in enumerate(lines) if line.startswith("MODEL"))
        total = next(i for i, line in enumerate(lines) if line.startswith("TOTAL"))
        between = lines[header + 2 : total - 1]  # exclude the rule before TOTAL
        self.assertEqual(between.count(""), 2)  # three families -> gaps after the first two blocks


class TestHeaderAndFooter(unittest.TestCase):
    def test_headline_mentions_queues(self):
        models = [model("org/GLM-5", running=2, waiting=7)]
        rendered = render_report(snapshot(*models), models, Theme(color=False), RenderOptions(width=120))
        self.assertIn("7 waiting", rendered)
        self.assertIn("GLM-5 +7", rendered)
        self.assertIn("busiest queue", rendered)

    def test_headline_stays_about_the_present_moment(self):
        # recent-queue context lives on the reserved "busiest queue" line, which has its own
        # line and so survives narrow terminals far better than a trailing clause
        models = [model("a/b", running=2)]
        rendered = render_report(snapshot(*models), models, Theme(color=False), RenderOptions(width=120))
        self.assertIn("no queues", rendered.splitlines()[1])
        self.assertNotIn("busiest queue", rendered.splitlines()[1])

    def test_footer_shows_source_and_time(self):
        rendered = render_report(snapshot(model("a")), [model("a")], Theme(color=False), RenderOptions(width=100))
        footer = rendered.splitlines()[-1]
        self.assertIn("test source", footer)
        self.assertRegex(footer, r"\d{2}:\d{2}:\d{2}")  # local time, whatever the zone


class TestGateway(unittest.TestCase):
    def results(self):
        return {
            "running": vector([]),
            "waiting": vector([]),
            "replicas": vector([]),
            "busy_replicas": vector([]),
            "kv_cache": vector([]),
            "queue_avg": vector([]),
            "queue_rate": vector([]),
            "waiting_reason": vector([]),
            "alias_concurrency": vector([
                (("gen_ai_original_model", "glm-5"), 4.0),
                (("gen_ai_original_model", "idle-alias"), 0.0),
            ]),
            "alias_rpm": vector([(("gen_ai_original_model", "glm-5"), 90.0)]),
            "alias_out_tps": vector([(("gen_ai_original_model", "glm-5"), 1200.0)]),
        }

    def test_gateway_section_is_appended(self):
        snap = make_snapshot(self.results(), source="s", window="5m")
        rendered = render_report(snap, snap.models, Theme(color=False), RenderOptions(width=120, show_gateway=True))
        self.assertIn("Gateway traffic by model alias", rendered)
        self.assertIn("glm-5", rendered)
        self.assertIn("1200", rendered)

    def test_gateway_hidden_by_default(self):
        snap = make_snapshot(self.results(), source="s", window="5m")
        rendered = render_report(snap, snap.models, Theme(color=False), RenderOptions(width=120))
        self.assertNotIn("Gateway traffic", rendered)

    def test_empty_gateway(self):
        lines = render_gateway([], Theme(color=False), RenderOptions(width=100))
        self.assertIn("no traffic", "\n".join(lines))


class TestLayoutDoesNotShift(unittest.TestCase):
    """The fleet table must start on the same line whether or not anything is queued."""

    theme = Theme(color=False)

    def body(self, rows, **over):
        snap = snapshot(*rows)
        return render_report(snap, rows, self.theme, RenderOptions(width=104, **over)).splitlines()

    def table_row(self, lines):
        return next(i for i, line in enumerate(lines) if line.startswith("MODEL"))

    def test_busiest_queue_line_is_always_present(self):
        quiet = [model("a/b", running=9, replicas=1)]
        busy = [model("a/b", running=9, waiting=12, replicas=1)]
        assert self.body(quiet)[3].startswith("busiest queue:")
        assert self.body(busy)[3].startswith("busiest queue:")

    def test_table_starts_at_the_same_line_in_every_state(self):
        states = {
            "idle, never queued": [model("a/b", running=9, replicas=1)],
            "idle, queued earlier": [model("a/b", running=9, replicas=1, waiting_peak=31)],
            "queued, no timing": [model("a/b", running=9, waiting=12, replicas=1)],
            "queued, with timing": [model("a/b", running=9, waiting=12, replicas=1, queue=4.2)],
            "nothing at all": [model("a/b", replicas=1)],
        }
        positions = {name: self.table_row(self.body(rows)) for name, rows in states.items()}
        assert len(set(positions.values())) == 1, positions

    def test_unknown_mean_wait_is_omitted_not_dashed(self):
        # a fresh queue has no completed requests to average; "(- mean wait)" reads as broken
        line = self.body([model("a/b", running=9, waiting=12, replicas=1)])[3]
        assert "mean wait" not in line, line
        assert "12 waiting" in line, line
        known = self.body([model("a/b", running=9, waiting=12, replicas=1, queue=4.2)])[3]
        assert "4.20s mean wait" in known, known

    def test_recent_queue_names_the_window_and_the_model(self):
        line = self.body([model("Inferact/GLM-5.3-NVFP4", running=9, replicas=1, waiting_peak=31)])[3]
        assert line == "busiest queue: none now, up to 31 in the last 15m on GLM-5.3-NVFP4", line


class TestTeamSection(unittest.TestCase):
    theme = Theme(color=False)

    def team(self, **overrides):
        keys = overrides.pop("keys", [
            KeyUsage("alpha-key", "qwen3", 12.5, 12.58),
            KeyUsage("alpha-key", "glm-5", 3.1, 41.2),
            KeyUsage("beta-key", "qwen3", 0.4, 0.031),
        ])
        return TeamOverview(team_id="acme-lab", window="15m", keys=keys, **overrides)

    def snap(self, team=None):
        models = [model("Qwen/Qwen3.8-27B", running=5, replicas=1)]
        return snapshot(*models, teams=[team if team is not None else self.team()])

    def render(self, snap=None, **overrides):
        opts = RenderOptions(width=110, **overrides)
        models = (snap or self.snap()).models
        return render_report(snap or self.snap(), models, self.theme, opts)

    def test_section_sits_below_total_behind_a_rule(self):
        lines = self.render().splitlines()
        total = next(i for i, line in enumerate(lines) if line.startswith("TOTAL"))
        team = next(i for i, line in enumerate(lines) if line.startswith("team "))
        between = [line for line in lines[total + 1 : team] if line.strip()]
        self.assertTrue(all(set(line) == {"-"} for line in between), between)
        self.assertGreater(team, total)

    def test_columns_show_key_model_rate_and_duration(self):
        body = self.render()
        self.assertIn("API KEY", body)
        self.assertIn("REQ/MIN", body)
        self.assertIn("MEAN SEC", body)
        self.assertIn("alpha-key", body)
        self.assertIn("12.5", body)
        self.assertIn("12.58s", body)
        self.assertIn("31ms", body)  # sub-second means stay readable

    def test_slow_calls_are_highlighted(self):
        colored = render_report(
            self.snap(self.team(keys=[KeyUsage("k", "m", 5.0, 41.2)])),
            self.snap().models,
            Theme(color=True),
            RenderOptions(width=110),
        )
        self.assertIn("\x1b[31m", colored)  # >=30s mean is red

    def test_summary_line_counts_keys_and_models(self):
        self.assertIn("16.0 req/min across 2 keys and 2 models", self.render())

    def test_long_lists_are_capped_with_the_remainder_accounted(self):
        keys = [KeyUsage(f"key-{i}", "qwen3", float(i), 1.0) for i in range(1, 13)]
        body = self.render(self.snap(self.team(keys=keys)), max_keys=8)
        self.assertIn("... +4", body)
        self.assertIn("12 pairs", body)
        self.assertEqual(body.count("key-"), 8)  # only the first 8 are listed

    def test_no_truncation_when_within_the_cap(self):
        keys = [KeyUsage(f"key-{i}", "qwen3", 1.0, 1.0) for i in range(3)]
        body = self.render(self.snap(self.team(keys=keys)), max_keys=8)
        self.assertNotIn("more pairs", body)

    def test_idle_team_says_so_without_a_table(self):
        body = self.render(self.snap(TeamOverview(team_id="quiet-team", window="15m")))
        self.assertIn("team quiet-team", body)
        self.assertIn("no traffic in the last 15m", body)
        self.assertNotIn("API KEY", body)

    def test_hidden_entirely_when_no_teams_requested(self):
        body = self.render(self.snap(TeamOverview(team_id="", keys=[])))
        self.assertNotIn("API KEY", body)

    def test_json_and_quiet_are_unaffected_by_the_cap(self):
        # the cap is display-only; to_dict keeps every pair
        keys = [KeyUsage(f"key-{i}", "qwen3", 1.0, 1.0) for i in range(12)]
        payload = self.snap(self.team(keys=keys)).to_dict()
        self.assertEqual(len(payload["teams"][0]["keys"]), 12)

    def test_coloured_layout_matches_plain(self):
        plain = self.render()
        colored = render_report(self.snap(), self.snap().models, Theme(color=True), RenderOptions(width=110))
        self.assertEqual(
            [strip_ansi(line) for line in plain.splitlines()],
            [strip_ansi(line) for line in colored.splitlines()],
        )


def next_line_starting_with(prefix: str):
    """A finder for a rendered line by its prefix, so assertions stay readable."""

    def find(text: str) -> str:
        return next(line for line in text.splitlines() if line.startswith(prefix))

    return find


class TestTokenStrip(unittest.TestCase):
    """The per-frame IN/OUT history strip in the team section."""

    theme = Theme(color=False)

    def iv(self, tokens, trend=Trend.FLAT):
        return Interval(tokens, tokens / 5.0, 5.0, trend)

    def key(self, **over):
        base = {
            "token_alias": "main", "model": "qwen3", "req_per_min": 41.2, "mean_seconds": 9.6,
            "in_tokens": 42_300_000, "out_tokens": 52_600, "reasoning_tokens": 25_700,
            "in_history": [self.iv(1000, Trend.FLAT), self.iv(3_050, Trend.UP), self.iv(2_900, Trend.FLAT)],
            "out_history": [self.iv(80, Trend.FLAT), self.iv(210, Trend.UP), self.iv(600, Trend.UP)],
        }
        base.update(over)
        return KeyUsage(**base)

    def render(self, *keys, **over):
        team = TeamOverview("acme-lab", "15m", list(keys) or [self.key()])
        return "\n".join(render_team(team, self.theme, RenderOptions(**{"width": 140, **over})))

    def header_line(self, body):
        return next(line for line in body.splitlines() if line.startswith("API KEY"))

    def test_three_frame_columns_oldest_left_newest_right(self):
        # age-labelled so the reader knows which side is current
        assert "IN/OUT -2" in self.header_line(self.render())

    def test_cell_shows_input_slash_output_each_arrowed(self):
        body = self.render()
        assert "\u21921.0K/\u219280" in body, body      # oldest slot
        assert "\u21913.0K/\u2191210" in body, body      # middle slot
        assert "\u21922.9K/\u2191600" in body, body      # newest slot

    def test_newest_reading_is_the_rightmost_frame_column(self):
        single = self.key(in_history=[self.iv(7, Trend.UP)], out_history=[self.iv(3, Trend.UP)])
        body = self.render(single)
        row = next(line for line in body.splitlines() if line.startswith("main"))
        assert row.rstrip().endswith("25.7K"), row      # REASON sits after the strip
        strip = [part for part in row.split() if "/" in part and any(a in part for a in "\u2191\u2193\u2192")]
        assert strip[-1] == "\u21917/\u21913", strip

    def test_no_history_shows_no_strip(self):
        # one-shot mode has nothing to compare, so it must not reserve 42 columns of placeholders
        fresh = self.key(in_history=[], out_history=[])
        body = self.render(fresh)
        assert "15m IN/OUT" in body                    # window column remains
        assert "IN/OUT now" not in body                # no frame columns
        assert "\u00b7/\u00b7" not in body, body      # no empty frame cells
        assert "--watch" in body, body                 # and it says how to get them

    def test_watching_reserves_the_strip_so_the_table_never_changes_shape(self):
        # without reserved columns the table gains three columns between frame 1 and frame 2
        fresh = self.key(in_history=[], out_history=[])
        body = self.render(fresh, history_reserved=True)
        assert "IN/OUT now" in body and "IN/OUT -2" in body
        assert body.count("\u00b7/\u00b7") == 6, body  # 3 slots x (data row + totals row)
        head = next_line_starting_with("API KEY")
        assert head(body) == head(self.render()), (head(body), head(self.render()))

    def test_pair_cells_are_never_truncated(self):
        # at 12 wide a 14-char pair silently lost its unit: "810.0K/270.0"
        huge = self.key(
            in_tokens=810_000, out_tokens=270_000,
            in_history=[self.iv(99_900, Trend.UP)] * 3, out_history=[self.iv(99_900, Trend.UP)] * 3,
        )
        body = self.render(huge)
        assert "810.0K/270.0K" in body, body
        assert "\u219199.9K/\u219199.9K" in body, body

    def test_rendering_preserves_the_order_it_is_given(self):
        # reordering on volume happens in build_team_overview; the renderer must not shuffle
        loud = self.key(token_alias="batch", model="glm-5", req_per_min=500.0)
        quiet = self.key(token_alias="alpha", model="qwen3", req_per_min=1.0)
        body = self.render(loud, quiet)
        names = [line.split()[0] for line in body.splitlines() if line.startswith(("alpha", "batch"))]
        assert names == ["batch", "alpha"], names

    def test_partial_history_pads_on_the_left(self):
        partial = self.key(
            in_history=[self.iv(500, Trend.FLAT), self.iv(900, Trend.UP)],
            out_history=[self.iv(50, Trend.FLAT), self.iv(90, Trend.UP)],
        )
        body = self.render(partial)
        row = next(line for line in body.splitlines() if line.startswith("main"))
        assert "\u2191900/\u219190" in row, row     # newest still lands in the right-hand column
        assert row.count("\u00b7/\u00b7") == 1, row  # only the leftmost slot is empty

    def test_history_length_is_configurable(self):
        five = self.key(
            in_history=[self.iv(i * 100, Trend.FLAT) for i in range(1, 6)],
            out_history=[self.iv(i * 10, Trend.FLAT) for i in range(1, 6)],
        )
        assert self.header_line(self.render(five)).count("IN/OUT -") == 2       # -2 and -1
        assert "IN/OUT now" in self.header_line(self.render(five))
        assert "IN/OUT -4" in self.header_line(self.render(five, history_slots=5))

    def test_totals_row_folds_the_strip_across_rows(self):
        second = self.key(
            token_alias="batch", model="glm-5",
            in_history=[self.iv(1_000, Trend.UP)] * 3, out_history=[self.iv(100, Trend.UP)] * 3,
        )
        body = self.render(self.key(), second)
        totals = next(line for line in body.splitlines() if line.startswith("2 pairs"))
        # each column folds the same slot across rows: newest = 2.9K+1.0K in / 600+100 out
        assert "\u21913.9K/\u2191700" in totals, totals

    def test_reason_is_still_explained_as_a_subset(self):
        body = self.render(history_slots=1, width=200)
        assert "REASON is part of its OUT" in body, body

    def test_coloured_strip_matches_plain_layout(self):
        plain = self.render()
        colored = render_team(
            TeamOverview("acme-lab", "15m", [self.key()]), Theme(color=True), RenderOptions(width=140)
        )
        assert [strip_ansi(x) for x in plain.splitlines()] == [strip_ansi(x) for x in colored]

    def test_arrows_survive_narrow_widths_without_mangling(self):
        for width in (78, 92, 108, 126):
            body = self.render(width=width)
            for line in body.splitlines():
                assert len(strip_ansi(line)) <= width, (width, line)
            row = next(line for line in body.splitlines() if line.startswith("main"))
            assert "\u21913.0K" in row and "\u2191600" in row, (width, row)


class TestQuiet(unittest.TestCase):
    def test_totals_only(self):
        models = [model("a", running=3, waiting=1), model("b", running=2)]
        self.assertEqual(render_quiet(models), "5 1")

    def test_respects_selection(self):
        all_models = [model("a", running=3, waiting=1), model("b", running=2)]
        self.assertEqual(render_quiet(all_models[:1]), "3 1")

    def test_empty_selection(self):
        self.assertEqual(render_quiet([]), "0 0")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
