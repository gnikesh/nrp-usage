# nrp-usage

**Is the model I want busy right now, and would I be stuck behind a queue?**

A zero-dependency terminal command for the [NRP LLM platform](https://docs.nrp-nautilus.io/llms/llms/),
built on the same metrics as the platform's Grafana dashboards. Run it before you launch a job, a
batch run, or an agent loop and pick a model that will actually answer.

[![tests](https://github.com/gnikesh/nrp-usage/actions/workflows/ci.yml/badge.svg)](https://github.com/gnikesh/nrp-usage/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![no dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)

![nrp-usage output in a dark terminal: a table of vLLM models with RUN/WAIT counts, the window's worst queue depth (WAIT MAX), the share of time each model was queueing (QUEUED), replica counts, KV-cache gauges, mean queue wait and a status verdict, with models grouped by family](docs/assets/nrp-usage.png)

<details>
<summary>Same output as text</summary>

```
NRP model usage
50 running · 0 waiting · no queues now, up to 5 queued in the last 15m · 7/11 models active
WAIT MAX, QUEUED and AVG WAIT cover the last 15m

MODEL                           RUN/WAIT    WAIT MAX    QUEUED    NODES         KV CACHE    AVG WAIT   STATUS
-------------------------------------------------------------------------------------------------------------
DeepSeek-V4-Flash-Vision-Exp         3/0           0        0%      2/2    ········   4%        <1ms   serving

gemma (3 models)                     7/0           -         -      1/9    █·······  15%           -   serving
  gemma-4-31B-it-qat-w4a16-ct        7/0           0        0%      1/3    █·······  15%        <1ms   serving
  gemma-4-12B-it-qat-w4a16-ct        0/0           0        0%      0/5    ········   0%        <1ms   idle
  gemma-4-E4B-it                     0/0           0        0%      0/1    ········   0%           -   idle

GLM-5.3-NVFP4                       11/0           5       18%      2/2    ███·····  39%       237ms   serving

gpt-oss-120b                         4/0           5        8%      1/1    ········   5%       211ms   serving

Kimi-K2.7-Code                       3/0           0        0%      1/1    ········   5%        <1ms   serving

MiniMax-M2.7                         0/0           0        0%      0/1    ········   0%         4ms   idle

Qwen (3 models)                     22/0           -         -      5/7    █·······  17%           -   serving
  Qwen3.8-27B                       20/0           3        5%      3/3    █·······  17%        64ms   serving
  Qwen3.8-Flash-Next-FP8             2/0           1        2%      2/2    ········   1%        51ms   serving
  Qwen3-VL-Embedding-8B              0/0           0        0%      0/2    ········   0%        <1ms   idle
-------------------------------------------------------------------------------------------------------------
TOTAL                               50/0           -         -    12/23                -           -   7/11 active

https://grafana.nrp-nautilus.io  ·  17:05:06
```

</details>

Note the headline: nothing is queued *right now*, but several models had a queue at points during the last 15 minutes. That is the difference between a lucky sample and a usable signal.

## Why this exists

The Grafana dashboard is the source of truth, but it is a dashboard: you open a browser, find the
panel, hover a line. This is the same numbers as `pip install`-and-go, in the place you already are,
and it fixes two ways the raw dashboard misleads you:

- **A queue gauge read at one instant lies.** `vllm:num_requests_waiting` drains in seconds, so the
  live number is usually `0` even on a model that spends most of its time queueing. `nrp-usage` shows
  the live value *and* how often the model was actually queueing in the window, so DeepSeek's scary
  "25 queued" that happened 10% of the time is not confused with GLM's modest "11 queued" that
  happens 28% of the time.
- **Alphabetical soup.** Engine names arrive as `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.8-Flash-Next-FP8`,
  `google/gemma-4-31B-it-qat-w4a16-ct`. They are grouped by model family with a subtotal, and the
  org prefix is dropped.

## Install

Requires Python 3.10+ and network access to `grafana.nrp-nautilus.io` (the datasource is publicly
readable; no account, token, or VPN needed).

```bash
pipx install git+https://github.com/gnikesh/nrp-usage.git
```

or with pip, in any environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install git+https://github.com/gnikesh/nrp-usage.git
```

Then just run it. No configuration is required:

```bash
nrp-usage
```

Optionally, point it at your team to also see which of your API keys are driving traffic
([how to find your team id](#pointing-it-at-your-team)):

```bash
nrp-usage --team your-team-id
echo 'export NRP_TEAMS=your-team-id' >> ~/.zshrc    # make it permanent
```

<details>
<summary>Running from a clone</summary>

```bash
git clone https://github.com/gnikesh/nrp-usage.git && cd nrp-usage
python -m venv venv && source venv/bin/activate
pip install -e .
ln -sf "$PWD/venv/bin/nrp-usage" ~/.local/bin/nrp-usage   # available without activating
```

</details>

## Pointing it at your team

The fleet table needs no account or configuration. Separately, you can add a short section under
`TOTAL` covering **your** traffic: which API keys are in use, which models they call, how many
requests/minute, and the mean duration of those calls.

It is opt-in and defaults to nothing, because a fresh install has no business showing anyone's API key
aliases. Enable it with `--team` (repeatable) or `NRP_TEAMS` (comma separated).

Find your team id from live traffic (names below are placeholders):

```
$ nrp-usage --list-teams --window 5m
Teams with traffic in the last 5m
  alpha-research-lab                   158.4 req/min
  beta-u-computing                      88.7 req/min
  gamma-bioinformatics                  20.5 req/min
  delta-climate-model                   12.8 req/min
  ...

use one of them:  nrp-usage --team alpha-research-lab
```

Team ids are also in the dashboard's `team_id` dropdown and match your platform account name.

```
$ nrp-usage --team acme-lab        # the section added under TOTAL
--------------------------------------------------------------------------------
team acme-lab
49.2 req/min across 3 keys and 3 models
API KEY             MODEL    REQ/MIN    MEAN SEC
------------------------------------------------
main                qwen3       41.2       9.60s
nightly-batch       glm-5        6.8      52.40s
dev-scratch         gemma        0.9       1.40s
dev-scratch         qwen3        0.3       600ms
4 key/model pairs               49.2           -
```

That example covers the behaviours worth knowing:

- **Zero-traffic pairs are hidden.** A team may have five key × model pairs in the counters while only
  four did anything in the window; the fifth would be a row of zeroes.
- `MEAN SEC` is real wall time per request (`rate(_sum) / rate(_count)`), so long streaming calls
  legitimately show tens of seconds. Above 30s it turns red, as `nightly-batch` would.
- One key can appear on several rows because each model is tracked separately.
- The totals row leaves `MEAN SEC` blank on purpose: averaging the per-pair means without weighting by
  request count would misrepresent the team.
- The list caps at `--keys N` (default 8); the `... N more pairs` row carries their combined rate, and
  `--json` always returns everything.

## Reading the output

| Column | Source | Meaning |
|---|---|---|
| `RUN/WAIT` | `vllm:num_requests_running` / `_waiting` | **right now**: requests in the execution batch, and requests queued behind you |
| `WAIT MAX` | `max_over_time` of the summed queue | the worst queue depth seen in the window; this is the dashboard's visible peak |
| `QUEUED` | share of window samples with a queue | **how often** you would have had to wait, which the peak alone hides |
| `AVG WAIT` | `vllm:request_queue_time_seconds` | mean seconds a finished request spent queued |
| `NODES` | count of replicas | busy / reporting |
| `KV CACHE` | `vllm:kv_cache_usage_perc` | worst replica; near 100% means new requests get deferred |
| `STATUS` | derived | `idle`, `serving`, `queued`, `saturated`, `congested` |

- `RUN/WAIT`, `NODES`, `KV CACHE` are instantaneous. `WAIT MAX`, `QUEUED`, `AVG WAIT` cover `--window`
  (default `15m`, matching the dashboard's `now-15m`), and the header says so above the table.
- Rows go **yellow** at `--warn-waiting` (default 1) and **red** at `--crit-waiting` (default 8), or on
  KV pressure. In `RUN/WAIT` only the waiting half carries the colour; it is the actionable number.
- Models are grouped by family (the leading word of the name: `Qwen`, `gemma`, `GLM`). A family with
  one model gets no header, since a heading identical to its only row is noise.
- **Family blocks never reorder**, so `--watch` does not jump around; `--sort` only reorders models
  inside a block. Use `-s name` for a completely static layout.
- `WAIT MAX`, `QUEUED` and `AVG WAIT` are `-` on subtotal and total rows on purpose. A family's peak
  queue is the peak of its *summed* series, which cannot be reconstructed from member peaks (they need
  not happen at the same moment), and averaging per-model means without their request rates would
  invent a number.

## Everything it can do

```bash
nrp-usage                       # every model, grouped by family
nrp-usage glm qwen              # filter (case-insensitive regex, falls back to literal)
nrp-usage -w                    # live view, every 5s;  --watch 2 for every 2s
nrp-usage -b                    # hide idle models
nrp-usage -s waiting            # within each family: load|running|waiting|name|kv|wait
nrp-usage --gateway             # traffic per model alias, fleet-wide
nrp-usage --gateway --team X    # ...scoped to one team
nrp-usage --reason              # split queues by vLLM wait reason (capacity vs deferred)
nrp-usage --list-teams          # which team ids are running traffic right now
nrp-usage --team acme-lab       # add that team's API-key overview under TOTAL (repeatable)
nrp-usage --no-teams            # drop the team section even if $NRP_TEAMS is set
nrp-usage --org                 # keep the Qwen/ google/ prefixes
nrp-usage --no-contention       # drop the WAIT MAX / QUEUED columns
nrp-usage --keys 20             # list more team rows before truncating
nrp-usage --width 200           # or pipe it; colour turns itself off
nrp-usage --explain             # print the PromQL it would run
nrp-usage -v                    # show the resolved endpoint
```

### Scripting

```bash
nrp-usage -q                       # "93 0"  (total running, total waiting)
nrp-usage -q glm                   # same, filtered to matching models
nrp-usage -x glm                   # exit 1 if glm has a queue right now
nrp-usage --json | jq -r '.models[] | select(.queued_share != null and .queued_share < 0.05) | .short_name'
nrp-usage --json | jq '.teams'                                                   # [] unless --team is set
nrp-usage --team acme-lab --json | jq '.teams[0].keys'                           # that team's API keys
nrp-usage --json | jq '.families[] | select(.waiting == 0) | .name'
```

A one-liner for "start when the model I want has a clear queue":

```bash
while ! nrp-usage -x -q qwen3 >/dev/null; do sleep 30; done
```

`--json` output is stable per-model data (with `short_name`, `family`, `queued_share`, `waiting_peak`)
plus a `families` rollup and the `teams` section. Display caps such as `--keys` do not apply to it.

**Exit codes:** `0` ok · `1` `--exit-if-busy` matched a queue · `2` the question could not be answered
· `130` interrupted.

Partial failure (some metrics unavailable) still renders and exits `0` with a `! partial data` note.
But if the backend answers successfully with **zero series**, or every query fails, you get exit `2`
and a message instead of a table of reassuring zeroes - a degraded metrics endpoint must not look like
an idle cluster.

## Options and environment

| Variable | Flag | Default |
|---|---|---|
| `NRP_TEAMS` | `--team` (repeatable) | unset, so no team section; `--list-teams` shows ids |
| `NRP_WINDOW` | `--window` | `15m` |
| `NRP_GRAFANA_URL` | `--grafana-url` | `https://grafana.nrp-nautilus.io` |
| `NRP_DATASOURCE_UID` | `--ds-uid` | `PC96415006F908B67` (`auto` discovers it from the dashboard) |
| `NRP_DASHBOARD_UID` | `--dashboard-uid` | `ad8bzhl` |
| `NRP_PROM_URL` | `--prom-url` | unset; point straight at any Prometheus/Thanos `/api/v1` |
| `NRP_TOKEN_FILE` | `--token-file` | unset (only needed if the datasource ever stops being public) |
| `NRP_TIMEOUT` | `--timeout` | `20` |
| `NRP_CA_BUNDLE` | | unset; explicit CA bundle |
| `NRP_INSECURE=1` | `--insecure` | off; skips TLS verification, use only as a last resort |
| `NO_COLOR=1` | `--no-color` | honoured automatically when stdout is not a tty |

## How it works

One Prometheus-compatible endpoint, eight-ish queries fired concurrently, rendered as a table.

```
GET https://grafana.nrp-nautilus.io/api/datasources/proxy/uid/<datasource>/api/v1/query
```

Grafana proxies its Thanos querier and permits anonymous reads; the querier's own address is
cluster-internal, so the proxy is the way in. The engine metrics are exactly panel 4 of the
[Envoy LLMs dashboard](https://grafana.nrp-nautilus.io/d/ad8bzhl/envoy-llms), plus neighbouring vLLM
gauges; the team section uses the gateway's `gen_ai_server_request_duration_seconds_*` counters.
`--explain` prints every expression.

If the datasource uid ever rotates, the resulting 404 is caught and the uid is re-read from the
publicly readable dashboard JSON and retried - under a lock, so the eight concurrent queries heal from
a single discovery rather than each racing to do it.

TLS trust is resolved by walking `NRP_CA_BUNDLE` / `SSL_CERT_FILE`, the interpreter default store,
`/etc/ssl/cert.pem` and friends, `certifi` if installed, and finally the macOS system keychain. That
chain exists because framework builds of Python on macOS ship without a CA bundle; verification is
never silently disabled.

## Design notes

Things that look like they should be one way and are not:

- **Mean queue time, not p95.** vLLM's lowest `request_queue_time_seconds` bucket is `le=0.3` and it
  holds ~98% of observations, so `histogram_quantile(0.95, ...)` interpolates inside that single bucket
  and reported ~285ms for idle models (real mean: 54ms). The mean is exact. It is suppressed to `-`
  when fewer than 3 requests left the queue in the window, because then even a mean is noise.
- **Peaks are computed over the summed series**:
  `max_over_time((sum by (model_name) (vllm:num_requests_waiting))[15m:15s])`. The tempting
  `sum by (...)(max_over_time(metric[15m]))` adds each replica's own peak and overshoots (measured 81
  against the true 60).
- **No peak-running column.** Pairing a running maximum with a waiting maximum implies a bad moment
  that never happened: measured over 61 samples, the two maxima co-occurred in 0 of 61 for seven of
  eight models.
- **Zero-traffic API keys are filtered out**, and the filter has to be applied to the right thing.
  The gateway keeps a counter series for every key × model it has ever seen, so one team's 16 pairs
  contained only 2 that did anything in 15 minutes. Note `count by (token_alias, model)(rate(...))`
  reports all 16 as non-zero because it counts series, not traffic - filter on the summed rate.
- **Engine names are not dashboard aliases.** `glm-5` (gateway) and `Inferact/GLM-5.3-NVFP4` (engine)
  share no mapping metric. Filters apply to engine names (`nrp-usage glm` matches), and `--gateway`
  shows alias-side load. Nothing is fuzzy-guessed.
- **PromQL regexes are not Python regexes.** `re.escape` escapes `-`, which RE2 rejects outright; the
  escape helpers cover that plus Go string quoting.

## Limitations

- Unofficial, read-only, and coupled to the NRP platform's Grafana datasource and metric names. If the
  dashboards change, this needs updating (`--explain` shows what it expects).
- Values are sample-granular: this tool and Grafana see the same ~30s scrapes, so a burst entirely
  between two scrapes is invisible to both.
- Engine metrics carry no team, so per-team queue depth cannot be attributed; `RUN/WAIT` is always the
  whole platform and the team section is gateway traffic.
- Colour assumes a modern terminal; everything degrades to plain text when piped.

## Development

```bash
source venv/bin/activate
pip install -e '.[dev]'
python -m unittest discover -s tests -t .   # 268 tests, fully offline
ruff check .
nrp-usage --explain --gateway               # inspect the queries
```

Tests never touch the network: `urlopen` and the fetch step are mocked, and `tests/fakes.py` builds
canned metric responses. Pure logic (`model.py`, `render.py`) is separated from I/O
(`prometheus.py`, `source.py`, `usage.py`) so layout and thresholds are unit-testable directly.

Contributions welcome - please add a test for anything you change and keep `ruff` clean.

## License

MIT, see [LICENSE](LICENSE).

This is a community tool, not an official project of the NRP / Nautilus Regional Production
platform or its operators.
