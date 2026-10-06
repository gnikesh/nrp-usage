"""Endpoint resolution for the NRP (Nautilus Regional Production) LLM platform.

The public Grafana at ``grafana.nrp-nautilus.io`` proxies its Thanos querier and
allows anonymous reads, so no token is needed. If the hard-coded datasource uid
ever rotates, it is rediscovered from the dashboard JSON and the request is
retried once (see :meth:`Source.make_client`).
"""

from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass

from .prometheus import PromClient, PromError

GRAFANA_URL = "https://grafana.nrp-nautilus.io"
#: The "Envoy LLMs" dashboard: /d/ad8bzhl/envoy-llms
DASHBOARD_UID = "ad8bzhl"
#: "thanos", the default datasource of that dashboard
DATASOURCE_UID = "PC96415006F908B67"
#: Panel 4: "vLLM queue depth - running vs waiting (all models)"
PANEL_QUEUE_DEPTH = 4


class SourceError(PromError):
    """No usable metrics endpoint could be determined."""


def _normalize_api_base(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        return ""
    if "://" not in url:
        url = "https://" + url
    return url if url.endswith("/api/v1") else url + "/api/v1"


def proxy_api_base(grafana_url: str, datasource_uid: str) -> str:
    """API base for a Grafana datasource proxy path."""
    return f"{grafana_url.rstrip('/')}/api/datasources/proxy/uid/{datasource_uid}/api/v1"


def discover_datasource_uid(
    grafana_url: str = GRAFANA_URL,
    dashboard_uid: str = DASHBOARD_UID,
    *,
    timeout: float = 20.0,
    insecure: bool = False,
) -> str | None:
    """Read a dashboard's JSON and return the uid of the Prometheus datasource it uses."""
    client = PromClient(f"{grafana_url.rstrip('/')}/api", timeout=timeout, insecure=insecure)
    try:
        doc = client.get_json(f"/dashboards/uid/{urllib.parse.quote(dashboard_uid, safe='')}")
    except (PromError, OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None

    found: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("type") == "prometheus" and isinstance(node.get("uid"), str):
                found.append(node["uid"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc.get("dashboard") or {})
    return next((uid for uid in found if not uid.startswith("--")), None)


def token_from_file(path: str) -> str:
    """Read the first non-comment line of a token file."""
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    except OSError as exc:
        raise SourceError(f"could not read token file {path}: {exc}") from exc
    raise SourceError(f"no token found in {path}")


@dataclass(frozen=True)
class Source:
    """Where and how to read the metrics."""

    api_base: str
    description: str
    label: str = ""
    grafana_url: str | None = None
    dashboard_uid: str | None = None
    datasource_uid: str | None = None
    token: str | None = None

    def make_client(self, *, timeout: float, insecure: bool) -> PromClient:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        return PromClient(
            self.api_base,
            timeout=timeout,
            insecure=insecure,
            headers=headers,
            recover=self._recover if (self.grafana_url and self.dashboard_uid) else None,
        )

    def _recover(self, client: PromClient) -> str | None:
        """Swap in the datasource uid the dashboard actually uses."""
        assert self.grafana_url is not None and self.dashboard_uid is not None
        uid = discover_datasource_uid(
            self.grafana_url,
            self.dashboard_uid,
            timeout=client.timeout,
            insecure=client.insecure,
        )
        if not uid:
            return None
        replacement = proxy_api_base(self.grafana_url, uid)
        return None if replacement == client.api_base else replacement


def resolve_source(
    *,
    prom_url: str | None = None,
    grafana_url: str | None = None,
    datasource_uid: str | None = None,
    dashboard_uid: str | None = None,
    token: str | None = None,
    timeout: float = 20.0,
    insecure: bool = False,
) -> Source:
    """Build the metrics source from CLI options, then environment, then defaults."""
    direct = (prom_url or os.environ.get("NRP_PROM_URL") or "").strip()
    if direct:
        return Source(api_base=_normalize_api_base(direct), description=direct, label=direct, token=token)

    grafana = (grafana_url or os.environ.get("NRP_GRAFANA_URL") or GRAFANA_URL).strip()
    dash = (dashboard_uid or os.environ.get("NRP_DASHBOARD_UID") or DASHBOARD_UID).strip()
    uid = (datasource_uid or os.environ.get("NRP_DATASOURCE_UID") or DATASOURCE_UID).strip()
    if uid.lower() in ("auto", "discover"):
        found = discover_datasource_uid(grafana, dash, timeout=timeout, insecure=insecure)
        if not found:
            raise SourceError(
                f"could not discover a Prometheus datasource from {grafana}/d/{dash}; "
                "pass --ds-uid or --prom-url."
            )
        uid = found
    return Source(
        api_base=proxy_api_base(grafana, uid),
        description=f"{grafana} · datasource {uid} · dashboard {dash} · panel {PANEL_QUEUE_DEPTH}",
        label=grafana,
        grafana_url=grafana,
        dashboard_uid=dash,
        datasource_uid=uid,
        token=token,
    )
