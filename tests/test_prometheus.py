"""Unit tests for the HTTP transport: parsing, error mapping, TLS and uid recovery.

Nothing here touches the network; ``urllib.request.urlopen`` is patched.
"""

from __future__ import annotations

import io
import json
import ssl
import unittest
import urllib.error
import urllib.request
from unittest import mock

from nrp_usage.prometheus import (
    AuthError,
    PromClient,
    PromError,
    QueryError,
    Sample,
    UnknownDatasource,
    _ca_sources,
    _is_verify_error,
)

API = "https://grafana.test/api/datasources/proxy/uid/DS1/api/v1"


def response(payload, status=200):
    """A context-manager stand-in for urlopen's response."""
    body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
    raw = body.encode() if isinstance(body, str) else body
    resp = mock.MagicMock(spec=["read", "__enter__", "__exit__", "status"])
    resp.status = status
    resp.read.return_value = raw
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def http_error(code, body):
    raw = body.encode() if isinstance(body, str) else body
    return urllib.error.HTTPError("https://grafana.test/x", code, "error", None, io.BytesIO(raw))  # type: ignore[arg-type]


def success(items):
    return {"status": "success", "data": {"resultType": "vector", "result": items}}


class TestQueryParsing(unittest.TestCase):
    def client(self, payload):
        client = PromClient(API, timeout=1)
        with mock.patch("urllib.request.urlopen", return_value=response(payload)):
            return client, client.query("up")

    def test_vector_is_parsed(self):
        client, samples = self.client(success([
            {"metric": {"model_name": "a", "team_id": "t"}, "value": [1700000000.5, "3"]},
            {"metric": {"model_name": "b"}, "value": [1700000000.5, "0.25"]},
        ]))
        self.assertEqual([s.label("model_name") for s in samples], ["a", "b"])
        self.assertEqual(samples[0].value, 3.0)
        self.assertEqual(samples[1].value, 0.25)
        self.assertEqual(samples[0].timestamp, 1700000000.5)
        self.assertTrue(client.api_base.endswith("/api/v1"))

    def test_non_numeric_and_missing_values_are_skipped(self):
        _, samples = self.client(success([
            {"metric": {"model_name": "ok"}, "value": [1, "1"]},
            {"metric": {"model_name": "bad"}, "value": [1, "not-a-number"]},
            {"metric": {"model_name": "stream"}, "values": [[1, "2"]]},
            {"metric": {"model_name": "no-value"}},
        ]))
        self.assertEqual([s.label("model_name") for s in samples], ["ok"])

    def test_promql_error_becomes_query_error(self):
        client = PromClient(API, timeout=1)
        payload = {"status": "error", "errorType": "bad_data", "error": "1:2: parse error"}
        with mock.patch("urllib.request.urlopen", return_value=response(payload)):
            with self.assertRaises(QueryError) as ctx:
                client.query("sum by (")
        self.assertIn("parse error", ctx.exception.message)
        self.assertEqual(ctx.exception.expr, "sum by (")

    def test_missing_result_is_an_error(self):
        client = PromClient(API, timeout=1)
        with mock.patch("urllib.request.urlopen", return_value=response({"status": "success", "data": {}})):
            with self.assertRaises(QueryError):
                client.query("up")

    def test_html_body_is_reported(self):
        client = PromClient(API, timeout=1)
        with mock.patch("urllib.request.urlopen", return_value=response("<html>login</html>")):
            with self.assertRaises(PromError) as ctx:
                client.query("up")
        self.assertIn("not JSON", str(ctx.exception))

    def test_time_parameter_is_sent(self):
        client = PromClient(API, timeout=1)
        with mock.patch("urllib.request.urlopen", return_value=response(success([]))) as open_:
            client.query("up", time=1700000000.0)
        url = open_.call_args.args[0].full_url
        self.assertIn("time=1700000000.000", url)

    def test_trailing_slash_is_normalised(self):
        self.assertEqual(PromClient(API + "/").api_base, API)


class TestErrorMapping(unittest.TestCase):
    def raise_for(self, exc, expr="up"):
        client = PromClient(API, timeout=1)
        with mock.patch("urllib.request.urlopen", side_effect=exc), self.assertRaises(PromError) as ctx:
            client.query(expr)
        return ctx.exception

    def test_unauthorised(self):
        self.assertIsInstance(self.raise_for(http_error(401, '{"message":"Unauthorized"}')), AuthError)
        self.assertIsInstance(self.raise_for(http_error(403, "nope")), AuthError)

    def test_unknown_datasource(self):
        err = self.raise_for(http_error(404, '{"message":"Unable to find datasource","traceID":""}'))
        self.assertIsInstance(err, UnknownDatasource)

    def test_plain_404_is_not_a_datasource_problem(self):
        self.assertIsInstance(self.raise_for(http_error(404, "not found")), PromError)

    def test_server_error(self):
        err = self.raise_for(http_error(500, "upstream timeout"))
        self.assertIn("500", str(err))
        self.assertNotIsInstance(err, UnknownDatasource)

    def test_connection_failure(self):
        err = self.raise_for(urllib.error.URLError("connection refused"))
        self.assertIn("connection refused", str(err))

    def test_raw_socket_error_mid_response_is_mapped(self):
        # urlopen raises these unwrapped; they must not escape as a traceback
        for exc in (
            ConnectionResetError(54, "Connection reset by peer"),
            ConnectionAbortedError(53, "Software caused connection abort"),
            TimeoutError("timed out"),
            OSError(9, "bad file descriptor"),
        ):
            with self.subTest(exc=type(exc).__name__):
                err = self.raise_for(exc)
                self.assertIsInstance(err, PromError)
                self.assertNotIsInstance(err, QueryError)

    def test_unexpected_error_type_still_yields_a_query_error(self):
        client = PromClient(API, timeout=1)
        with mock.patch.object(PromClient, "query", side_effect=RuntimeError("bug in a dependency")):
            results = client.query_many({"a": "up"})
        self.assertIsInstance(results["a"], QueryError)
        self.assertIn("bug in a dependency", results["a"].message)

    def test_dns_failure_says_so(self):
        err = self.raise_for(urllib.error.URLError(OSError(8, "nodename nor servname provided, or not known")))
        self.assertIn("could not resolve host", str(err))

    def test_sample_helpers(self):
        self.assertTrue(Sample({"a": "b"}, 1.0, 0.0).is_finite)
        self.assertFalse(Sample({}, float("nan")).is_finite)
        self.assertFalse(Sample({}, float("inf")).is_finite)
        self.assertEqual(Sample({"x": "1"}).label("x"), "1")
        self.assertEqual(Sample({}).label("missing", "dflt"), "dflt")


class TestTlsFallback(unittest.TestCase):
    def verify_error(self):
        return urllib.error.URLError(
            ssl.SSLCertVerificationError(18, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        )

    def test_next_ca_source_is_tried_then_succeeds(self):
        client = PromClient(API, timeout=1)
        payloads = [self.verify_error(), response(success([{"metric": {"model_name": "a"}, "value": [1, "2"]}]))]
        with mock.patch("urllib.request.urlopen", side_effect=payloads):
            samples = client.query("up")
        self.assertEqual(samples[0].label("model_name"), "a")
        self.assertGreater(client._tls_index, 0)

    def test_all_sources_exhausted_gives_actionable_message(self):
        client = PromClient(API, timeout=1)
        sources = len(_ca_sources())
        failures = [self.verify_error() for _ in range(sources + 2)]
        with mock.patch("urllib.request.urlopen", side_effect=failures):
            with self.assertRaises(PromError) as ctx:
                client.query("up")
        message = str(ctx.exception)
        self.assertIn("TLS certificate verification failed", message)
        self.assertIn("NRP_CA_BUNDLE", message)
        self.assertIn("NRP_INSECURE", message)

    def test_insecure_skips_verification_and_retries(self):
        client = PromClient(API, timeout=1, insecure=True)
        with mock.patch("urllib.request.urlopen", side_effect=[self.verify_error(), response(success([]))]):
            with self.assertRaises(PromError):
                client.query("up")  # one attempt only when insecure

    def test_is_verify_error_detects_both_shapes(self):
        self.assertTrue(_is_verify_error(self.verify_error().reason))
        self.assertTrue(_is_verify_error(ssl.SSLError("unable to get local issuer certificate")))
        self.assertFalse(_is_verify_error(TimeoutError("timed out")))


class TestDatasourceRecovery(unittest.TestCase):
    def test_stale_uid_is_replaced_and_retried(self):
        calls: list[str] = []

        def urlopen(request, *args, **kwargs):
            calls.append(request.full_url)
            if "BAD/api/v1" in request.full_url:
                raise http_error(404, '{"message":"Unable to find datasource","traceID":""}')
            return response(success([{"metric": {"model_name": "healed"}, "value": [1, "9"]}]))

        client = PromClient(API.replace("DS1", "BAD"), timeout=1)
        client.recover = lambda failed: API
        with mock.patch("urllib.request.urlopen", side_effect=urlopen):
            samples = client.query("up")
        self.assertEqual(len(calls), 2)
        self.assertEqual(samples[0].label("model_name"), "healed")
        self.assertEqual(client.api_base, API)

    def test_without_a_recovery_hook_the_error_surfaces(self):
        client = PromClient(API, timeout=1)
        missing = http_error(404, '{"message":"Unable to find datasource"}')
        with mock.patch("urllib.request.urlopen", side_effect=missing):
            with self.assertRaises(UnknownDatasource):
                client.query("up")

    def test_recovery_returning_the_same_base_does_not_loop(self):
        client = PromClient(API, timeout=1)
        client.recover = lambda failed: API
        failure = http_error(404, '{"message":"Unable to find datasource"}')
        with mock.patch("urllib.request.urlopen", side_effect=failure) as open_:
            with self.assertRaises(UnknownDatasource):
                client.query("up")
        self.assertEqual(open_.call_count, 1)

    def test_concurrent_queries_all_survive_a_stale_uid(self):
        """query_many runs in parallel: one winner heals, the rest must still succeed."""
        discovered = {"n": 0}

        def urlopen(request, *args, **kwargs):
            if "BAD/api/v1" in request.full_url:
                raise http_error(404, '{"message":"Unable to find datasource"}')
            return response(success([{"metric": {"model_name": "healed"}, "value": [1, "9"]}]))

        def recover(failed: PromClient) -> str:
            discovered["n"] += 1
            return API

        client = PromClient(API.replace("DS1", "BAD"), timeout=1)
        client.recover = recover
        exprs = {f"q{i}": f"up{i}" for i in range(8)}
        with mock.patch("urllib.request.urlopen", side_effect=urlopen):
            results = client.query_many(exprs)
        failures = {k: v for k, v in results.items() if isinstance(v, QueryError)}
        self.assertEqual(failures, {}, f"{len(failures)}/8 queries did not heal")
        self.assertTrue(all(r[0].label("model_name") == "healed" for r in results.values()))
        # discovery is expensive, so it must happen once, not once per query
        self.assertEqual(discovered["n"], 1)


class TestQueryMany(unittest.TestCase):
    def test_runs_all_and_isolates_failures(self):
        client = PromClient(API, timeout=1)

        def fake_query(expr, *, time=None):
            if "boom" in expr:
                raise QueryError(expr, "boom")
            return [Sample({"model_name": expr}, float(len(expr)))]

        with mock.patch.object(PromClient, "query", side_effect=lambda expr, time=None: fake_query(expr)):
            results = client.query_many({"a": "one", "b": "boom", "c": "three"})
        self.assertIsInstance(results["a"], list)
        self.assertIsInstance(results["b"], QueryError)
        self.assertEqual(results["c"][0].value, 5.0)
        self.assertEqual(results["a"][0].value, 3.0)

    def test_empty_input(self):
        self.assertEqual(PromClient(API).query_many({}), {})

    def test_single_query_path(self):
        with mock.patch.object(PromClient, "query", return_value=[Sample({"model_name": "x"}, 1.0)]):
            results = PromClient(API).query_many({"only": "up"})
        self.assertEqual(results["only"][0].label("model_name"), "x")

    def test_label_values(self):
        client = PromClient(API, timeout=1)
        with mock.patch("urllib.request.urlopen", return_value=response({"status": "success", "data": ["b", "a"]})):
            self.assertEqual(client.label_values("model_name"), ["b", "a"])
        payload = {"status": "success", "data": {"values": ["z"]}}
        with mock.patch("urllib.request.urlopen", return_value=response(payload)):
            self.assertEqual(client.label_values("model_name", 'vllm:num_requests_running'), ["z"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
