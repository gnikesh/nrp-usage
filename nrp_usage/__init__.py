"""nrp-usage: live model queue depth for the NRP LLM platform.

Reads vLLM engine metrics through the public NRP Grafana datasource proxy and
reports how many requests each model is running and how many are waiting.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
