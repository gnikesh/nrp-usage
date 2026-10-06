"""Unit tests for the pure logic in nrp_usage.model."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timezone

from nrp_usage.model import (
    ModelUsage,
    Snapshot,
    Thresholds,
    build_aliases,
    build_models,
    build_team_overview,
    by_label,
    by_model,
    by_model_and_label,
    filter_models,
    group_by_family,
    sort_models,
    summarise,
)
from tests.fakes import vector


def model(name, running=0, waiting=0, replicas=1, busy=0, kv=None, queue=None, cap=0, deferred=0) -> ModelUsage:
    # keyword construction: ModelUsage gains fields over time, positional build silently shifts
    return ModelUsage(
        name=name,
        running=running,
        waiting=waiting,
        replicas=replicas,
        busy_replicas=busy,
        kv_cache=kv,
        queue_avg=queue,
        waiting_capacity=cap,
        waiting_deferred=deferred,
    )


class TestSampleCollapsing(unittest.TestCase):
    def test_by_model_sums_duplicate_series(self):
        samples = vector([("a", 1.0), ("a", 2.5), ("b", 4.0)])
        self.assertEqual(by_model(samples), {"a": 3.5, "b": 4.0})

    def test_nan_and_missing_labels_are_dropped(self):
        samples = [
            vector([("a", 1.0)])[0],
            vector([("b", float("nan"))])[0],
            vector([("c", math.inf)])[0],
            vector([("", 5.0)])[0],
        ]
        self.assertEqual(by_model(samples), {"a": 1.0})

    def test_by_label_uses_any_label(self):
        samples = vector([(("gen_ai_original_model", "glm-5"), 3.0)])
        self.assertEqual(by_label(samples, "gen_ai_original_model"), {"glm-5": 3.0})

    def test_by_model_and_label(self):
        samples = vector([
            (("model_name", "a"), ("reason", "capacity"), 2.0),
            (("model_name", "a"), ("reason", "deferred"), 1.0),
            (("model_name", "b"), ("reason", "capacity"), 4.0),
        ])
        self.assertEqual(
            by_model_and_label(samples, "reason"),
            {"a": {"capacity": 2.0, "deferred": 1.0}, "b": {"capacity": 4.0}},
        )


class TestBuildModels(unittest.TestCase):
    def test_union_of_names_keeps_idle_models(self):
        models = build_models(
            running={"a": 3.0},
            waiting={},
            replicas={"a": 2.0, "b": 5.0},
            busy_replicas={"a": 1.0},
            kv_cache={"a": 0.5, "b": 0.0},
            queue_avg={"b": 0.2},
            waiting_by_reason={},
        )
        self.assertEqual([m.name for m in models], ["a", "b"])
        self.assertEqual((models[0].running, models[0].busy_replicas, models[0].replicas), (3, 1, 2))
        self.assertEqual(models[1].running, 0)
        self.assertEqual(models[1].queue_avg, 0.2)

    def test_wait_reasons_are_split(self):
        models = build_models(
            running={},
            waiting={"a": 7.0},
            replicas={},
            busy_replicas={},
            kv_cache={},
            queue_avg={},
            waiting_by_reason={"a": {"capacity": 5.0, "deferred": 2.0}},
        )
        self.assertEqual((models[0].waiting, models[0].waiting_capacity, models[0].waiting_deferred), (7, 5, 2))

    def test_absent_kv_stays_none(self):
        models = build_models(
            running={"a": 1.0}, waiting={}, replicas={}, busy_replicas={},
            kv_cache={}, queue_avg={}, waiting_by_reason={},
        )
        self.assertIsNone(models[0].kv_cache)


class TestThresholds(unittest.TestCase):
    def test_verdicts(self):
        th = Thresholds(waiting_warn=1, waiting_crit=8)
        cases = [
            (model("a", replicas=1), "idle"),
            (model("a", running=4, busy=1), "serving"),
            (model("a", running=4, waiting=3, busy=1), "queued"),
            (model("a", running=4, waiting=9, busy=1), "congested"),
            (model("a", running=4, kv=0.95, busy=1), "saturated"),
        ]
        for row, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(row.verdict(th), expected)

    def test_level_ordering(self):
        self.assertTrue(model("a", waiting=9).level() > model("b", waiting=1).level())
        self.assertTrue(model("c").level() < model("d", waiting=1).level())

    def test_custom_thresholds_move_the_line(self):
        row = model("a", running=1, waiting=4, busy=1)
        self.assertEqual(row.verdict(Thresholds(waiting_crit=2)), "congested")
        self.assertEqual(row.verdict(Thresholds(waiting_warn=5, waiting_crit=9)), "serving")


class TestFilteringAndSorting(unittest.TestCase):
    def setUp(self):
        self.rows = [
            model("Qwen/Qwen3.8-27B", running=40, waiting=0),
            model("Inferact/GLM-5.3-NVFP4", running=10, waiting=6),
            model("google/gemma-4-12B", running=0, waiting=0),
        ]

    def test_substring_filter_is_case_insensitive(self):
        got = [m.name for m in filter_models(self.rows, ["glm"])]
        self.assertEqual(got, ["Inferact/GLM-5.3-NVFP4"])

    def test_regex_filter(self):
        got = [m.name for m in filter_models(self.rows, [r"^Qwen/"])]
        self.assertEqual(got, ["Qwen/Qwen3.8-27B"])

    def test_invalid_regex_falls_back_to_literal(self):
        rows = [*self.rows, model("meta/Llama-3.1(8B-Instruct")]  # unbalanced paren
        got = [m.name for m in filter_models(rows, ["3.1(8B"])]
        self.assertEqual(got, ["meta/Llama-3.1(8B-Instruct"])

    def test_no_patterns_keeps_everything(self):
        self.assertEqual(len(filter_models(self.rows, [])), 3)

    def test_busy_only_hides_idle(self):
        got = [m.name for m in filter_models(self.rows, [], busy_only=True)]
        self.assertEqual(got, ["Qwen/Qwen3.8-27B", "Inferact/GLM-5.3-NVFP4"])

    def test_sort_by_load_then_queue(self):
        got = [m.name for m in sort_models(list(reversed(self.rows)), "load")]
        self.assertEqual(got, ["Qwen/Qwen3.8-27B", "Inferact/GLM-5.3-NVFP4", "google/gemma-4-12B"])

    def test_sort_by_name_is_case_insensitive(self):
        got = [m.name for m in sort_models(self.rows, "name")]
        self.assertEqual(got, ["google/gemma-4-12B", "Inferact/GLM-5.3-NVFP4", "Qwen/Qwen3.8-27B"])

    def test_sort_by_waiting_prefers_queues(self):
        rows = [model("a", running=50), model("b", running=1, waiting=2)]
        self.assertEqual([m.name for m in sort_models(rows, "waiting")], ["b", "a"])

    def test_sort_by_queue_uses_wait_time(self):
        rows = [model("a", running=5, queue=0.01), model("b", running=1, queue=9.0)]
        self.assertEqual([m.name for m in sort_models(rows, "wait")], ["b", "a"])


class TestSnapshot(unittest.TestCase):
    def snapshot(self):
        return Snapshot(
            source="test",
            fetched_at=datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc),
            models=[
                model("a/x", running=3, waiting=2, replicas=2, busy=1, cap=2),
                model("b/y", running=0, waiting=0, replicas=1),
            ],
            errors=["kv_cache: unavailable"],
        )

    def test_totals(self):
        totals = self.snapshot().totals
        self.assertEqual((totals.running, totals.waiting, totals.replicas, totals.busy_replicas), (3, 2, 3, 1))
        self.assertEqual((totals.models, totals.active_models, totals.queued_models), (2, 1, 1))

    def test_headline_names_the_queued_model(self):
        # the headline uses the short name so it stays one readable line
        self.assertIn("x +2", self.snapshot().headline())
        self.assertIn("2 waiting", self.snapshot().headline())

    def test_headline_when_nothing_waits(self):
        snap = Snapshot(source="t", fetched_at=datetime.now(timezone.utc), models=[model("a", running=1)])
        self.assertIn("no queues", snap.headline())

    def test_summarise_matches_snapshot_totals(self):
        rows = [model("a", running=2, waiting=1), model("b", running=5)]
        self.assertEqual(summarise(rows).running, 7)
        self.assertEqual(summarise(rows).waiting, 1)

    def test_to_dict_shape(self):
        payload = self.snapshot().to_dict()
        self.assertEqual(payload["source"], "test")
        self.assertEqual(payload["totals"]["running"], 3)
        self.assertEqual(payload["models"][0]["name"], "a/x")
        self.assertEqual(payload["errors"], ["kv_cache: unavailable"])
        self.assertIsNone(payload["aliases"])

    def test_local_time_renders(self):
        self.assertRegex(self.snapshot().local_time(), r"\d{2}:\d{2}:\d{2}")


class TestFamilies(unittest.TestCase):
    def test_family_is_the_leading_word_of_the_short_name(self):
        cases = {
            "Qwen/Qwen3.8-27B": "Qwen",
            "Qwen/Qwen3.8-Flash-Next-FP8": "Qwen",
            "Qwen/Qwen3-VL-Embedding-8B": "Qwen",
            "google/gemma-4-31B-it-qat-w4a16-ct": "gemma",
            "Inferact/GLM-5.3-NVFP4": "GLM",
            "MiniMaxAI/MiniMax-M2.7": "MiniMax",
            "moonshotai/Kimi-K2.7-Code": "Kimi",
            "openai/gpt-oss-120b": "gpt",
            "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp": "DeepSeek",
            "vendor/plainmodel": "plainmodel",
        }
        for name, family in cases.items():
            with self.subTest(name=name):
                self.assertEqual(model(name).family, family)

    def test_names_starting_with_a_digit_keep_their_short_name(self):
        # otherwise every numeric-prefixed checkpoint would merge into one bogus family
        self.assertEqual(model("x/7B-instruct").family, "7B-instruct")
        self.assertEqual(model("x/7B-instruct").short_name, "7B-instruct")

    def test_groups_are_alphabetical_and_ignore_load(self):
        models = [
            model("Inferact/GLM-5.3-NVFP4", running=1),
            model("Qwen/Qwen3.8-27B", running=999),
            model("google/gemma-4.31B", running=5),
        ]
        for key in ("load", "name", "waiting", "kv", "wait"):
            with self.subTest(key=key):
                # 'gemma' < 'GLM' < 'Qwen' caselessly, whatever the traffic looks like
                self.assertEqual([g.label for g in group_by_family(models, key)], ["gemma", "GLM", "Qwen"])

    def test_group_membership_is_case_insensitive(self):
        groups = group_by_family([model("a/GLM-5.1"), model("b/glm-4.2")])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].label, "GLM")  # most common spelling, then alphabetical

    def test_label_spelling_does_not_depend_on_input_order(self):
        rows = [model("a/GLM-5.1"), model("b/glm-5.2"), model("c/Glm-5.3"), model("d/GLM-5.4")]
        forward = group_by_family(rows).pop().label
        reverse = group_by_family(list(reversed(rows))).pop().label
        self.assertEqual(forward, reverse)

    def test_summary_aggregates_and_worst_kv(self):
        group = group_by_family([
            model("Qwen/Qwen3.8-27B", running=5, waiting=1, replicas=2, busy=1, kv=0.4, cap=1),
            model("Qwen/Qwen3.8-Flash", running=2, replicas=3, busy=0, kv=0.9, deferred=2),
            model("Qwen/Qwen3-VL", replicas=1),
        ]).pop()
        summary = group.summary()
        self.assertEqual((summary.running, summary.waiting), (7, 1))
        self.assertEqual((summary.replicas, summary.busy_replicas), (6, 1))
        self.assertEqual((summary.waiting_capacity, summary.waiting_deferred), (1, 2))
        self.assertAlmostEqual(summary.kv_cache, 0.9)  # worst replica in the family
        self.assertIsNone(summary.queue_avg)  # averaging means without rates would be invented

    def test_group_ordering_ignores_the_sort_key_but_members_follow_it(self):
        group = group_by_family([
            model("Qwen/Qwen3.8-27B", running=1, queue=3.5),
            model("Qwen/Qwen3.8-Flash", running=1, queue=0.1),
        ], "wait").pop()
        self.assertIsNone(group.summary().queue_avg)  # the subtotal never shows an invented mean
        self.assertEqual([m.short_name for m in group.members], ["Qwen3.8-27B", "Qwen3.8-Flash"])

    def test_members_within_a_group_follow_the_sort_key(self):
        models = [
            model("Qwen/Qwen3-b", running=1),
            model("Qwen/Qwen3-a", running=9),
            model("Qwen/Qwen3-c", running=5),
        ]
        self.assertEqual(
            [m.short_name for m in group_by_family(models, "running").pop().members],
            ["Qwen3-a", "Qwen3-c", "Qwen3-b"],
        )

    def test_single_model_families_are_still_groups(self):
        groups = group_by_family([model("openai/gpt-oss-120b", running=2)])
        self.assertEqual([(g.label, len(g.members)) for g in groups], [("gpt", 1)])


class TestTeamOverview(unittest.TestCase):
    def test_pairs_merge_on_key_and_model(self):
        team = build_team_overview(
            "acme-lab",
            {("alpha-key", "qwen3"): 12.5, ("alpha-key", "glm-5"): 3.1, ("beta-key", "qwen3"): 0.4},
            {("alpha-key", "qwen3"): 12.58, ("alpha-key", "glm-5"): 41.2, ("beta-key", "qwen3"): 0.031},
            "15m",
        )
        self.assertEqual(
            [(k.token_alias, k.model) for k in team.keys][:2],
            [("alpha-key", "qwen3"), ("alpha-key", "glm-5")],
        )
        self.assertAlmostEqual(team.req_per_min, 16.0)
        self.assertEqual((team.token_aliases, team.models), (2, 2))
        self.assertIn("16.0 req/min across 2 keys and 2 models", team.summary())

    def test_zero_traffic_pairs_are_dropped(self):
        # the counter series outlives the traffic; listing 0.0 rows buries the signal
        team = build_team_overview(
            "t",
            {("live", "qwen3"): 4.0, ("idle", "gemma"): 0.0},
            {("live", "qwen3"): 1.0, ("idle", "gemma"): 0.0},
        )
        self.assertEqual([k.token_alias for k in team.keys], ["live"])

    def test_missing_mean_stays_none(self):
        team = build_team_overview("t", {("k", "m"): 5.0}, {})
        self.assertIsNone(team.keys[0].mean_seconds)

    def test_empty_team(self):
        team = build_team_overview("quiet-team", {}, {}, "15m")
        self.assertEqual(team.summary(), "no traffic in the last 15m")
        self.assertEqual(team.keys, [])

    def test_sorted_by_traffic(self):
        team = build_team_overview(
            "t",
            {("small", "m"): 1.0, ("big", "m"): 50.0, ("mid", "m"): 10.0},
            {},
        )
        self.assertEqual([k.token_alias for k in team.keys], ["big", "mid", "small"])

    def test_singular_wording(self):
        team = build_team_overview("t", {("only", "qwen3"): 2.0}, {("only", "qwen3"): 1.0})
        self.assertEqual(team.summary(), "2.0 req/min across 1 key and 1 model")


class TestBuildAliases(unittest.TestCase):
    def test_merges_and_flags_activity(self):
        aliases = build_aliases({"glm-5": 2.5, "qwen3": 0.0}, {"glm-5": 60.0}, {"glm-5": 500.0, "idle": 1.0})
        self.assertEqual([a.name for a in aliases], ["glm-5", "idle", "qwen3"])
        self.assertTrue(aliases[0].is_active)
        self.assertFalse(next(a for a in aliases if a.name == "qwen3").is_active)
        self.assertAlmostEqual(aliases[0].output_tokens_per_sec, 500.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
