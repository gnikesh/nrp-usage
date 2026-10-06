"""Unit tests for endpoint resolution and datasource discovery."""

from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from unittest import mock

from nrp_usage.prometheus import PromClient
from nrp_usage.source import (
    DASHBOARD_UID,
    DATASOURCE_UID,
    GRAFANA_URL,
    SourceError,
    discover_datasource_uid,
    proxy_api_base,
    resolve_source,
    token_from_file,
)


def _unlink(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


PROXY = f"{GRAFANA_URL}/api/datasources/proxy/uid/{DATASOURCE_UID}/api/v1"


class TestResolveDefaults(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_grafana_proxy_is_the_default(self):
        source = resolve_source()
        self.assertEqual(source.api_base, PROXY)
        self.assertIn("datasource", source.description)
        self.assertEqual(source.datasource_uid, DATASOURCE_UID)

    def test_explicit_overrides(self):
        source = resolve_source(grafana_url="https://g.example", datasource_uid="XYZ", dashboard_uid="abc")
        self.assertEqual(source.api_base, "https://g.example/api/datasources/proxy/uid/XYZ/api/v1")
        self.assertIn("dashboard abc", source.description)

    def test_environment_overrides(self):
        with mock.patch.dict(os.environ, {"NRP_GRAFANA_URL": "https://env.example", "NRP_DATASOURCE_UID": "E1"}):
            self.assertIn("env.example/api/datasources/proxy/uid/E1", resolve_source().api_base)

    def test_cli_beats_environment(self):
        with mock.patch.dict(os.environ, {"NRP_DATASOURCE_UID": "ENVUID"}):
            self.assertIn("CLIUID", resolve_source(datasource_uid="CLIUID").api_base)

    def test_direct_prometheus_url_bypasses_grafana(self):
        source = resolve_source(prom_url="http://thanos.local:9090")
        self.assertEqual(source.api_base, "http://thanos.local:9090/api/v1")
        self.assertIsNone(source.dashboard_uid)  # nothing to rediscover from

    def test_direct_url_is_normalised(self):
        self.assertEqual(resolve_source(prom_url="prom.internal:9090/").api_base, "https://prom.internal:9090/api/v1")
        self.assertEqual(resolve_source(prom_url="http://x/api/v1").api_base, "http://x/api/v1")

    def test_env_direct_url(self):
        with mock.patch.dict(os.environ, {"NRP_PROM_URL": "http://p:9090"}):
            self.assertEqual(resolve_source().api_base, "http://p:9090/api/v1")

    def test_blank_prom_url_is_ignored(self):
        self.assertEqual(resolve_source(prom_url="   ").api_base, PROXY)

    def test_proxy_api_base_helper(self):
        self.assertEqual(proxy_api_base("https://g/", "U"), "https://g/api/datasources/proxy/uid/U/api/v1")


class TestAutoDiscovery(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_auto_discovers_the_uid(self):
        with mock.patch("nrp_usage.source.discover_datasource_uid", return_value="DISCOVERED") as found:
            source = resolve_source(datasource_uid="auto")
        self.assertIn("DISCOVERED", source.api_base)
        found.assert_called_once()

    def test_discover_alias_works_too(self):
        with mock.patch("nrp_usage.source.discover_datasource_uid", return_value="D2") as found:
            self.assertIn("D2", resolve_source(datasource_uid="discover").api_base)
            # discovery reads the dashboard it was pointed at
            self.assertEqual(found.call_args.args[:2], (GRAFANA_URL, DASHBOARD_UID))

    def test_empty_uid_means_the_default_not_discovery(self):
        with mock.patch("nrp_usage.source.discover_datasource_uid", side_effect=AssertionError("must not fetch")):
            self.assertIn(DATASOURCE_UID, resolve_source(datasource_uid="").api_base)

    def test_failure_to_discover_is_actionable(self):
        with mock.patch("nrp_usage.source.discover_datasource_uid", return_value=None):
            with self.assertRaises(SourceError) as ctx:
                resolve_source(datasource_uid="auto")
        self.assertIn("--ds-uid", str(ctx.exception))


class TestDiscoverDatasourceUid(unittest.TestCase):
    """Patched at ``nrp_usage.source.PromClient`` (what the module imported), not in prometheus."""

    def payload(self):
        return {
            "dashboard": {
                "panels": [
                    {"datasource": {"type": "grafana", "uid": "-- Grafana --"}},
                    {"targets": [{"datasource": {"type": "prometheus", "uid": "PROM1"}}]},
                    {"panels": [{"datasource": {"type": "loki", "uid": "LOKI1"}}]},
                ],
                "templating": {"list": [{"datasource": {"type": "prometheus", "uid": "PROM2"}}]},
            }
        }

    def patch_client(self, get_json):
        client = mock.MagicMock()
        client.get_json = get_json
        patcher = mock.patch("nrp_usage.source.PromClient", return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)
        return client

    def test_returns_first_prometheus_uid_and_skips_builtins(self):
        client = self.patch_client(mock.MagicMock(return_value=self.payload()))
        self.assertEqual(discover_datasource_uid(), "PROM1")
        self.assertIn("/dashboards/uid/" + DASHBOARD_UID, client.get_json.call_args.args[0])

    def test_only_builtin_datasources_returns_none(self):
        self.patch_client(mock.MagicMock(return_value={"dashboard": {"panels": [
            {"datasource": {"type": "prometheus", "uid": "-- Grafana --"}}]}}))
        self.assertIsNone(discover_datasource_uid())

    def test_network_failure_returns_none(self):
        self.patch_client(mock.MagicMock(side_effect=OSError("dns down")))
        self.assertIsNone(discover_datasource_uid())

    def test_dashboard_without_panels(self):
        self.patch_client(mock.MagicMock(return_value={"dashboard": {}}))
        self.assertIsNone(discover_datasource_uid())

    def test_malformed_document(self):
        self.patch_client(mock.MagicMock(return_value=None))
        self.assertIsNone(discover_datasource_uid())


class TestToken(unittest.TestCase):
    def write(self, text: str) -> str:
        path = tempfile.mktemp(suffix=".token")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        self.addCleanup(_unlink, path)
        return path

    def test_first_real_line_wins(self):
        path = self.write("# comment\n\nglsa_abc123\nsecond\n")
        self.assertEqual(token_from_file(path), "glsa_abc123")

    def test_missing_file(self):
        with self.assertRaises(SourceError):
            token_from_file("/nonexistent/dir/token")

    def test_empty_file(self):
        with self.assertRaises(SourceError):
            token_from_file(self.write("# only a comment\n"))

    def test_token_becomes_an_auth_header(self):
        source = resolve_source(prom_url="http://p:9090", token="sekrit")
        client = source.make_client(timeout=5, insecure=False)
        self.assertEqual(client._headers.get("Authorization"), "Bearer sekrit")

    def test_anonymous_by_default(self):
        client = resolve_source().make_client(timeout=5, insecure=False)
        self.assertIsNone(client._headers.get("Authorization"))


class TestClientWiring(unittest.TestCase):
    def test_grafana_source_can_recover_a_rotated_uid(self):
        source = resolve_source()
        client = source.make_client(timeout=7, insecure=False)
        self.assertIsNotNone(client.recover)
        self.assertEqual(client.timeout, 7)

    def test_recovery_uses_the_discovered_uid(self):
        source = resolve_source()
        client = source.make_client(timeout=5, insecure=False)
        with mock.patch("nrp_usage.source.discover_datasource_uid", return_value="NEW"):
            self.assertEqual(client.recover(client), proxy_api_base(GRAFANA_URL, "NEW"))

    def test_recovery_is_a_noop_when_discovery_fails(self):
        source = resolve_source()
        client = source.make_client(timeout=5, insecure=False)
        with mock.patch("nrp_usage.source.discover_datasource_uid", return_value=None):
            self.assertIsNone(client.recover(client))

    def test_recovery_stops_when_the_uid_has_not_changed(self):
        source = resolve_source()
        client = source.make_client(timeout=5, insecure=False)
        with mock.patch("nrp_usage.source.discover_datasource_uid", return_value=DATASOURCE_UID):
            self.assertIsNone(client.recover(client))

    def test_direct_source_has_no_recovery(self):
        client = resolve_source(prom_url="http://p:9090").make_client(timeout=5, insecure=False)
        self.assertIsNone(client.recover)

    def test_insecure_flag_is_forwarded(self):
        self.assertTrue(resolve_source().make_client(timeout=5, insecure=True).insecure)

    def test_client_type(self):
        self.assertIsInstance(resolve_source().make_client(timeout=5, insecure=False), PromClient)

    def test_defaults_match_the_live_dashboard(self):
        self.assertEqual(DASHBOARD_UID, "ad8bzhl")
        self.assertEqual(GRAFANA_URL, "https://grafana.nrp-nautilus.io")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
