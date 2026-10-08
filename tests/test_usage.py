"""Unit tests for query construction and snapshot assembly."""

from __future__ import annotations

import re
import time
import unittest
from datetime import datetime, timezone

from nrp_usage.usage import (
    MIN_QUEUE_SAMPLES,
    WindowError,
    build_queries,
    describe_queries,
    escape_prom_regex,
    fetch_snapshot,
    list_teams,
    make_snapshot,
    prom_matcher,
    regex_matcher,
    team_traffic_query,
    validate_window,
    window_seconds,
)
from tests.fakes import error, sample, vector

ENGINE_KEYS = {
    "running", "waiting", "replicas", "busy_replicas",
    "kv_cache", "waiting_reason", "queue_avg", "queue_rate",
    "waiting_peak", "waiting_share",
}


class TestWindow(unittest.TestCase):
    def test_valid(self):
        for value in ("30s", "5m", "1h", "10ms", "7d", "1w", "1y"):
            self.assertEqual(validate_window(value), value)

    def test_invalid(self):
        for value in ("", "5", "m", "5x", "5 minutes", "-5m"):
            with self.subTest(value=value), self.assertRaises(WindowError):
                validate_window(value)

    def test_window_seconds(self):
        self.assertEqual(window_seconds("5m"), 300)
        self.assertEqual(window_seconds("90s"), 90)
        self.assertEqual(window_seconds("1h30m"), 5400)
        self.assertEqual(window_seconds("250ms"), 0.25)


class TestMatchers(unittest.TestCase):
    def test_empty_selection_matches_everything(self):
        self.assertEqual(regex_matcher([]), ".*")
        self.assertEqual(regex_matcher(["$__all", ""]), ".*")

    def test_values_are_anchored_and_escaped(self):
        self.assertEqual(regex_matcher(["glm-5", "qwen.plus"]), "^(glm-5|qwen\\.plus)$")
        # RE2 has no escape sequence for '-', so it must stay literal
        self.assertNotIn("\\-", regex_matcher(["glm-5"]))

    def test_matchers_survive_go_string_quoting(self):
        self.assertEqual(prom_matcher("team_id", ["team-a"]), 'team_id=~"^(team-a)$"')
        self.assertEqual(prom_matcher("team_id", ["a.b"]), 'team_id=~"^(a\\\\.b)$"')
        self.assertEqual(prom_matcher("team_id", []), 'team_id=~".*"')

    def test_escape_prom_regex_covers_metacharacters(self):
        for raw in ("a.b", "a+b", "a(b", "a|b", "a[b", "a{2}", "^a$", "a\\b", "a*b", "a?b"):
            escaped = escape_prom_regex(raw)
            self.assertTrue(re.fullmatch(escaped, raw), (raw, escaped))


class TestBuildQueries(unittest.TestCase):
    def test_engine_only_by_default(self):
        queries = build_queries()
        self.assertEqual(set(queries), ENGINE_KEYS)
        self.assertEqual(queries["running"], 'sum by (model_name) (vllm:num_requests_running)')
        self.assertEqual(queries["busy_replicas"], 'count by (model_name) (vllm:num_requests_running > 0)')

    def test_window_is_substituted_everywhere(self):
        queries = build_queries(window="30s", gateway=True)
        self.assertIn("[30s]", queries["queue_avg"])
        self.assertIn("[30s]", queries["queue_rate"])
        self.assertIn("[30s]", queries["alias_rpm"])
        for expr in queries.values():
            self.assertNotIn("{window}", expr)

    def test_gateway_queries_use_alias_label(self):
        queries = build_queries(gateway=True)
        self.assertEqual(
            set(queries) - ENGINE_KEYS,
            {"alias_concurrency", "alias_rpm", "alias_out_tps"},
        )
        for key in ("alias_concurrency", "alias_rpm", "alias_out_tps"):
            self.assertIn("gen_ai_original_model", queries[key])
        self.assertIn('gen_ai_token_type="output"', queries["alias_out_tps"])

    def test_dashboard_vars_become_matchers(self):
        queries = build_queries(gateway=True, team_ids=["t1"], token_aliases=["a", "b"])
        self.assertIn('team_id=~"^(t1)$"', queries["alias_concurrency"])
        self.assertIn('token_alias=~"^(a|b)$"', queries["alias_concurrency"])

    def test_no_matchers_when_unfiltered(self):
        queries = build_queries(gateway=True)
        self.assertIn("duration_seconds_sum{}", queries["alias_concurrency"])
        self.assertNotIn("team_id", queries["alias_concurrency"])

    def test_bad_window_raises(self):
        with self.assertRaises(WindowError):
            build_queries(window="soon")

    def test_describe_queries_lists_expressions(self):
        text = describe_queries(gateway=True)
        self.assertIn("running", text)
        self.assertIn("vllm:num_requests_running", text)
        self.assertIn("alias_rpm", text)


    def test_contention_queries_use_subqueries_over_the_sum(self):
        # must be max/share of the summed series: sum(max_over_time(...)) overshoots
        queries = build_queries(window="15m")
        self.assertEqual(
            queries["waiting_peak"],
            'max_over_time((sum by (model_name) (vllm:num_requests_waiting))[15m:15s])',
        )
        self.assertEqual(
            queries["waiting_share"],
            'sum_over_time((sum by (model_name) (vllm:num_requests_waiting) > bool 0)[15m:15s]) / '
            'count_over_time((sum by (model_name) (vllm:num_requests_waiting))[15m:15s])',
        )

    def test_no_running_peak_query(self):
        # pairing a running max with a waiting max implies a state that never occurred
        self.assertNotIn("running_peak", build_queries(window="15m"))

    def test_contention_off_skips_the_subqueries(self):
        queries = build_queries(window="5m", contention=False)
        self.assertNotIn("waiting_peak", queries)
        self.assertNotIn("waiting_share", queries)
        self.assertIn("running", queries)


class TestTeamQueries(unittest.TestCase):
    def test_two_queries_per_team_grouped_by_key_and_model(self):
        queries = build_queries(window="15m", overview_teams=["acme-lab"])
        rpm = queries["team_rpm[acme-lab]"]
        mean = queries["team_mean[acme-lab]"]
        self.assertIn("sum by (token_alias, gen_ai_original_model)", rpm)
        self.assertIn('team_id=~"^(acme-lab)$"', rpm)
        self.assertTrue(rpm.endswith("* 60"))
        self.assertIn("/", mean)
        self.assertIn("_sum{", mean)
        self.assertIn("_count{", mean)

    def test_each_team_gets_its_own_scoped_queries(self):
        # 6 per team: rate, mean duration, in/out/reasoning counters, and the raw
        # cumulative counter that frame-to-frame arrows are computed from
        queries = build_queries(window="5m", overview_teams=["a", "b"])
        self.assertEqual(len([k for k in queries if k.startswith("team_")]), 12)
        self.assertIn('team_id=~"^(a)$"', queries["team_rpm[a]"])
        self.assertNotIn("(b)", queries["team_rpm[a]"])

    def test_regex_metacharacters_in_a_team_id_are_escaped(self):
        query = build_queries(window="5m", overview_teams=["my.team-x"])["team_rpm[my.team-x]"]
        # RE2 needs the dot escaped, and PromQL string literals escape the backslash again
        self.assertIn('team_id=~"^(my\\\\.team-x)$"', query)
        # and '-' must stay literal: RE2 rejects \- outright
        self.assertNotIn("\\-", query)

    def test_no_team_queries_by_default(self):
        self.assertFalse([k for k in build_queries() if k.startswith("team_")])

    def test_team_section_reaches_the_snapshot(self):
        results = {
            "running": vector([("Qwen/a", 5)]),
            "waiting": vector([("Qwen/a", 0)]),
            "replicas": vector([("Qwen/a", 1)]),
            "busy_replicas": vector([("Qwen/a", 1)]),
            "kv_cache": vector([("Qwen/a", 0.1)]),
            "queue_avg": vector([]),
            "queue_rate": vector([]),
            "waiting_reason": vector([]),
            "waiting_peak": vector([]),
            "waiting_share": vector([]),
            "team_rpm[acme-lab]": vector([
                (("token_alias", "alpha-key"), ("gen_ai_original_model", "qwen3"), 12.5)
            ]),
            "team_mean[acme-lab]": vector([
                (("token_alias", "alpha-key"), ("gen_ai_original_model", "qwen3"), 12.58)
            ]),
        }
        snap = make_snapshot(results, source="unit", window="15m")
        self.assertEqual([t.team_id for t in snap.teams], ["acme-lab"])
        key = snap.teams[0].keys[0]
        self.assertEqual((key.token_alias, key.model), ("alpha-key", "qwen3"))
        self.assertAlmostEqual(key.req_per_min, 12.5)
        self.assertAlmostEqual(key.mean_seconds, 12.58)

    def test_one_failing_team_query_does_not_lose_the_other(self):
        results = {
            "running": vector([("Qwen/a", 5)]),
            "waiting": vector([]),
            "replicas": vector([("Qwen/a", 1)]),
            "busy_replicas": vector([]),
            "kv_cache": vector([]),
            "queue_avg": vector([]),
            "queue_rate": vector([]),
            "waiting_reason": vector([]),
            "waiting_peak": vector([]),
            "waiting_share": vector([]),
            "team_rpm[acme-lab]": error("team_rpm[acme-lab]", "422"),
            "team_mean[acme-lab]": vector([
                (("token_alias", "alpha-key"), ("gen_ai_original_model", "qwen3"), 12.58)
            ]),
        }
        snap = make_snapshot(results, source="unit", window="15m")
        self.assertIn("team_rpm", snap.errors[0])
        self.assertEqual(snap.teams[0].keys, [])  # no rpm means nothing to rank by

    def test_team_traffic_query_is_plain_rate_per_team(self):
        self.assertEqual(
            team_traffic_query("15m"),
            "sum by (team_id) (rate(gen_ai_server_request_duration_seconds_count[15m])) * 60",
        )

    def test_token_queries_cover_all_three_types(self):
        queries = build_queries(window="15m", overview_teams=["acme-lab"])
        for suffix, token_type in (("tin", "input"), ("tout", "output"), ("treason", "reasoning")):
            expr = queries[f"team_{suffix}[acme-lab]"]
            self.assertIn(f'gen_ai_token_type="{token_type}"', expr)
            self.assertIn("increase(", expr)
            self.assertIn("gen_ai_client_token_usage_sum", expr)
            self.assertIn('team_id=~"^(acme-lab)$"', expr)
            self.assertIn("[15m]", expr)

    def test_token_maps_reach_the_overview(self):
        results = {
            "running": vector([("Qwen/a", 5)]),
            "waiting": vector([]),
            "replicas": vector([("Qwen/a", 1)]),
            "busy_replicas": vector([]),
            "kv_cache": vector([]),
            "queue_avg": vector([]),
            "queue_rate": vector([]),
            "waiting_reason": vector([]),
            "waiting_peak": vector([]),
            "waiting_share": vector([]),
            "team_rpm[t1]": vector([(("token_alias", "k1"), ("gen_ai_original_model", "m1"), 3.0)]),
            "team_mean[t1]": vector([(("token_alias", "k1"), ("gen_ai_original_model", "m1"), 1.5)]),
            "team_tin[t1]": vector([(("token_alias", "k1"), ("gen_ai_original_model", "m1"), 7_104_961.4)]),
            "team_tout[t1]": vector([(("token_alias", "k1"), ("gen_ai_original_model", "m1"), 33_962.0)]),
            "team_treason[t1]": vector([(("token_alias", "k1"), ("gen_ai_original_model", "m1"), 21_671.0)]),
        }
        team = make_snapshot(results, source="unit", window="15m").teams[0]
        key = team.keys[0]
        self.assertEqual((round(key.in_tokens), round(key.out_tokens)), (7_104_961, 33_962))
        self.assertEqual(round(key.reasoning_tokens), 21_671)
        self.assertEqual(round(team.in_tokens), 7_104_961)

    def test_reasoning_never_exceeds_output(self):
        # increase() extrapolates, so a rounding wobble must not imply extra tokens
        results = {
            "running": vector([]), "waiting": vector([]), "replicas": vector([]),
            "busy_replicas": vector([]), "kv_cache": vector([]), "queue_avg": vector([]),
            "queue_rate": vector([]), "waiting_reason": vector([]), "waiting_peak": vector([]),
            "waiting_share": vector([]),
            "team_rpm[t1]": vector([(("token_alias", "k"), ("gen_ai_original_model", "m"), 3.0)]),
            "team_mean[t1]": vector([]),
            "team_tin[t1]": vector([]),
            "team_tout[t1]": vector([(("token_alias", "k"), ("gen_ai_original_model", "m"), 100.0)]),
            "team_treason[t1]": vector([(("token_alias", "k"), ("gen_ai_original_model", "m"), 104.0)]),
        }
        key = make_snapshot(results, source="unit").teams[0].keys[0]
        self.assertEqual(key.reasoning_tokens, 100.0)

    def test_missing_token_data_is_zero_not_none(self):
        results = {
            "running": vector([]), "waiting": vector([]), "replicas": vector([]),
            "busy_replicas": vector([]), "kv_cache": vector([]), "queue_avg": vector([]),
            "queue_rate": vector([]), "waiting_reason": vector([]), "waiting_peak": vector([]),
            "waiting_share": vector([]),
            "team_rpm[t1]": vector([(("token_alias", "k"), ("gen_ai_original_model", "m"), 3.0)]),
            "team_mean[t1]": vector([]),
        }
        key = make_snapshot(results, source="unit").teams[0].keys[0]
        self.assertEqual((key.in_tokens, key.out_tokens, key.reasoning_tokens), (0.0, 0.0, 0.0))

    def test_json_carries_token_totals(self):
        results = {
            "running": vector([]), "waiting": vector([]), "replicas": vector([]),
            "busy_replicas": vector([]), "kv_cache": vector([]), "queue_avg": vector([]),
            "queue_rate": vector([]), "waiting_reason": vector([]), "waiting_peak": vector([]),
            "waiting_share": vector([]),
            "team_rpm[t1]": vector([(("token_alias", "k"), ("gen_ai_original_model", "m"), 3.0)]),
            "team_mean[t1]": vector([]),
            "team_tin[t1]": vector([(("token_alias", "k"), ("gen_ai_original_model", "m"), 1500.0)]),
            "team_tout[t1]": vector([(("token_alias", "k"), ("gen_ai_original_model", "m"), 250.0)]),
            "team_treason[t1]": vector([(("token_alias", "k"), ("gen_ai_original_model", "m"), 90.0)]),
        }
        entry = make_snapshot(results, source="unit").to_dict()["teams"][0]
        self.assertEqual(entry["in_tokens"], 1500)
        self.assertEqual(entry["out_tokens"], 250)
        self.assertEqual(entry["reasoning_tokens"], 90)
        self.assertEqual(entry["keys"][0]["in_tokens"], 1500.0)

    def test_json_includes_the_team_section(self):
        results = {
            "running": vector([("Qwen/a", 5)]),
            "waiting": vector([]),
            "replicas": vector([("Qwen/a", 1)]),
            "busy_replicas": vector([]),
            "kv_cache": vector([]),
            "queue_avg": vector([]),
            "queue_rate": vector([]),
            "waiting_reason": vector([]),
            "waiting_peak": vector([]),
            "waiting_share": vector([]),
            "team_rpm[t1]": vector([(("token_alias", "k1"), ("gen_ai_original_model", "m1"), 7.5)]),
            "team_mean[t1]": vector([(("token_alias", "k1"), ("gen_ai_original_model", "m1"), 2.5)]),
        }
        payload = make_snapshot(results, source="unit", window="15m").to_dict()
        self.assertEqual(payload["teams"][0]["team_id"], "t1")
        self.assertEqual(payload["teams"][0]["keys"][0]["token_alias"], "k1")


class TestContentionAssembly(unittest.TestCase):
    def base(self, **overrides):
        results = {
            "running": vector([("Qwen/a", 49)]),
            "waiting": vector([("Qwen/a", 0)]),
            "replicas": vector([("Qwen/a", 3)]),
            "busy_replicas": vector([("Qwen/a", 3)]),
            "kv_cache": vector([("Qwen/a", 0.4)]),
            "queue_avg": vector([("Qwen/a", 0.05)]),
            "queue_rate": vector([("Qwen/a", 1.0)]),
            "waiting_reason": vector([]),
            "waiting_peak": vector([("Qwen/a", 12)]),
            "waiting_share": vector([("Qwen/a", 0.333)]),
        }
        results.update(overrides)
        return make_snapshot(results, source="unit", window="15m")

    def test_contention_reaches_the_row(self):
        row = self.base().models[0]
        self.assertEqual(row.waiting_peak, 12)
        self.assertAlmostEqual(row.queued_share, 0.333)

    def test_missing_share_stays_none_rather_than_zero(self):
        # 0% would claim "never queued"; missing data must not look like that
        snap = self.base(waiting_share=error("waiting_share", "502"))
        self.assertIsNone(snap.models[0].queued_share)
        self.assertIn("waiting_share", snap.errors[0])

    def test_headline_keeps_live_queue_detail_on_its_own_line(self):
        snap = self.base(waiting=vector([("Qwen/a", 3)]))
        self.assertIn("queued on a +3", snap.headline())

    def test_headline_prefers_the_live_queue_over_the_peak(self):
        snap = self.base(waiting=vector([("Qwen/a", 3)]))
        self.assertIn("queued on a +3", snap.headline())
        self.assertNotIn("recently", snap.headline())

    def test_headline_stays_clean_when_nothing_ever_queued(self):
        snap = self.base(waiting_peak=vector([("Qwen/a", 0)]), waiting_share=vector([("Qwen/a", 0.0)]))
        self.assertIn("no queues", snap.headline())
        self.assertNotIn("recently", snap.headline())

    def test_json_carries_the_contention_fields(self):
        payload = self.base().to_dict()
        self.assertEqual(payload["models"][0]["waiting_peak"], 12)
        self.assertAlmostEqual(payload["models"][0]["queued_share"], 0.333)


class TestMakeSnapshot(unittest.TestCase):
    def snapshot(self, **overrides):
        results = {
            "running": vector([("Qwen/x", 40), ("GLM/y", 10)]),
            "waiting": vector([("GLM/y", 3)]),
            "replicas": vector([("Qwen/x", 3), ("GLM/y", 1)]),
            "busy_replicas": vector([("Qwen/x", 2), ("GLM/y", 1)]),
            "kv_cache": vector([("Qwen/x", 0.44), ("GLM/y", 0.9)]),
            "queue_avg": vector([("Qwen/x", 0.05), ("GLM/y", 4.2)]),
            "queue_rate": vector([("Qwen/x", 1.0), ("GLM/y", 0.5)]),
            "waiting_reason": vector([(("model_name", "GLM/y"), ("reason", "capacity"), 3.0)]),
        }
        results.update(overrides)
        return make_snapshot(results, source="unit", window="5m")

    def test_rows_are_built_from_all_metrics(self):
        snap = self.snapshot()
        self.assertEqual([m.name for m in snap.models], ["GLM/y", "Qwen/x"])
        glm = snap.models[0]
        self.assertEqual((glm.running, glm.waiting, glm.replicas, glm.busy_replicas), (10, 3, 1, 1))
        self.assertAlmostEqual(glm.kv_cache, 0.9)
        self.assertEqual(glm.waiting_capacity, 3)
        self.assertFalse(snap.errors)

    def test_failed_query_is_reported_not_fatal(self):
        snap = self.snapshot(kv_cache=error("kv_cache", "502 bad gateway"))
        self.assertEqual(len(snap.models), 2)
        self.assertIsNone(snap.models[0].kv_cache)
        self.assertIn("kv_cache", snap.errors[0])
        self.assertIn("502", snap.errors[0])

    def test_queue_average_needs_enough_samples(self):
        # 0.001 req/s * 300s = 0.3 requests in the window: noise, not a measurement
        snap = self.snapshot(queue_rate=vector([("Qwen/x", 0.001), ("GLM/y", 0.5)]))
        self.assertIsNone(next(m for m in snap.models if m.name == "Qwen/x").queue_avg)
        self.assertIsNotNone(next(m for m in snap.models if m.name == "GLM/y").queue_avg)

    def test_boundary_of_sample_gate(self):
        needed = MIN_QUEUE_SAMPLES / window_seconds("5m")  # req/s that yields exactly MIN_QUEUE_SAMPLES
        snap = self.snapshot(queue_rate=vector([("Qwen/x", needed), ("GLM/y", needed * 0.99)]))
        by_name = {m.name: m for m in snap.models}
        self.assertIsNotNone(by_name["Qwen/x"].queue_avg)
        self.assertIsNone(by_name["GLM/y"].queue_avg)

    def test_gateway_section_only_when_queried(self):
        self.assertIsNone(self.snapshot().aliases)
        snap = self.snapshot()
        results = {
            "alias_concurrency": vector([(("gen_ai_original_model", "glm-5"), 2.0)]),
            "alias_rpm": vector([(("gen_ai_original_model", "glm-5"), 120.0)]),
            "alias_out_tps": vector([(("gen_ai_original_model", "glm-5"), 900.0)]),
        }
        # engine keys must still be present for the row builder
        merged = {**{k: vector([]) for k in ENGINE_KEYS}, **results}
        with_gateway = make_snapshot(merged, source="unit", window="5m")
        self.assertEqual([a.name for a in with_gateway.aliases], ["glm-5"])
        self.assertAlmostEqual(with_gateway.aliases[0].requests_per_min, 120.0)
        self.assertTrue(snap.aliases is None)

    def test_nan_samples_are_ignored(self):
        snap = self.snapshot(kv_cache=vector([("Qwen/x", float("nan"))]))
        self.assertIsNone(next(m for m in snap.models if m.name == "Qwen/x").kv_cache)

    def test_fetched_at_defaults_to_now(self):
        snap = make_snapshot({k: vector([]) for k in ENGINE_KEYS}, source="s")
        self.assertLess(abs((snap.fetched_at - datetime.now(timezone.utc)).total_seconds()), 60)


class _StubClient:
    """Records the queries it was asked for and replays canned results."""

    def __init__(self, results):
        self.results = results
        self.seen = None

    def query_many(self, queries):
        self.seen = dict(queries)
        return {name: self.results.get(name, vector([])) for name in queries}


class TestFetchSnapshotTiming(unittest.TestCase):
    def test_snapshot_is_stamped_when_queries_run_not_when_assembly_ends(self):
        # the footer time is the instant the samples were read, so re-querying at
        # snapshot.fetched_at reproduces what is displayed
        class TimingClient:
            def query_many(self, queries):
                time.sleep(0.05)
                return {name: vector([]) for name in queries}

        snap = fetch_snapshot(TimingClient(), source="unit", window="5m")
        age = (datetime.now(timezone.utc) - snap.fetched_at).total_seconds()
        self.assertGreaterEqual(age, 0.05)


class TestFetchSnapshot(unittest.TestCase):
    def test_passes_filters_through_to_queries(self):
        client = _StubClient({})
        fetch_snapshot(
            client,
            source="unit",
            window="10m",
            gateway=True,
            team_ids=["team-a"],
            token_aliases=["tk"],
        )
        self.assertIn("alias_out_tps", client.seen)
        self.assertIn("[10m]", client.seen["queue_avg"])
        self.assertIn('team_id=~"^(team-a)$"', client.seen["alias_concurrency"])
        self.assertIn('token_alias=~"^(tk)$"', client.seen["alias_concurrency"])

    def test_snapshot_carries_source_and_window(self):
        snap = fetch_snapshot(_StubClient({}), source="unit", window="1m")
        self.assertEqual((snap.source, snap.window), ("unit", "1m"))
        self.assertEqual(snap.models, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TestListTeams(unittest.TestCase):
    """Discovery of team ids that are actually running traffic."""

    class Client:
        def __init__(self, samples):
            self._samples = samples
            self.seen = None

        def query(self, expr, *, time=None):  # noqa: ARG002 - mirrors PromClient.query
            self.seen = expr
            return self._samples

    def samples(self, *pairs):
        return [sample(rate, team_id=team) for team, rate in pairs]

    def test_sorted_busiest_first(self):
        client = self.Client(self.samples(("slow", 1.0), ("fast", 40.0), ("mid", 9.0)))
        self.assertEqual([t for t, _ in list_teams(client)], ["fast", "mid", "slow"])

    def test_zero_and_missing_labels_are_dropped(self):
        client = self.Client([*self.samples(("idle", 0.0), ("live", 3.0)), sample(5.0)])
        self.assertEqual([t for t, _ in list_teams(client)], ["live"])

    def test_patterns_filter(self):
        client = self.Client(self.samples(("acme-lab", 10.0), ("other", 5.0)))
        self.assertEqual([t for t, _ in list_teams(client, patterns=["acme"])], ["acme-lab"])

    def test_invalid_pattern_is_treated_as_literal(self):
        client = self.Client(self.samples(("a(b", 1.0), ("other", 2.0)))
        self.assertEqual([t for t, _ in list_teams(client, patterns=["a(b"])], ["a(b"])

    def test_window_is_applied_to_the_query(self):
        client = self.Client([])
        list_teams(client, window="30s")
        self.assertIn("[30s]", client.seen)

    def test_bad_window_is_rejected(self):
        with self.assertRaises(WindowError):
            list_teams(self.Client([]), window="nope")
