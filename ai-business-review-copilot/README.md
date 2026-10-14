# 🤖 AI Business Review Copilot

**Automates the Monday-morning Weekly Business Review: a KPI engine finds what moved and why, an LLM writes the narrative, and a number guard makes sure it can't make numbers up.**

![Python](https://img.shields.io/badge/Python-3.10+-1F3A5F?style=flat-square&logo=python&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-optional-1F3A5F?style=flat-square&logo=langchain&logoColor=white) ![Slack](https://img.shields.io/badge/Slack-Block%20Kit-1F3A5F?style=flat-square&logo=slack&logoColor=white) ![tests](https://img.shields.io/badge/tests-pytest-1F3A5F?style=flat-square)

> ⚠️ Ships with a **synthetic** 12-week, 6-city KPI feed (`wbr/data.py`). It is not real data from any company. Point `--csv` at your own export to use it for real.

---

## Why

Writing a WBR by hand means pulling numbers, eyeballing 48 city×metric cells, working out *why* GMV moved, and writing it all up. That's hours of analyst time every week. LLMs can write the narrative, but **an LLM that makes up a number in an exec review is worse than no LLM at all.**

This project splits the work:

| Layer | Job | Deterministic? |
|---|---|---|
| **KPI engine** (`wbr/metrics.py`) | Weekly rollups, WoW and vs-4-week deltas, **LMDI GMV bridge** (orders vs AOV), **robust z-score anomaly scan** (median/MAD) | ✅ |
| **Facts builder** (`wbr/facts.py`) | Picks issues, **ranks them by ₹ impact** (not by z-score), attaches likely driver and owner action, pre-formats every number | ✅ |
| **Narrator** (`wbr/narrator.py`) | LangChain chat model (OpenAI or Anthropic) writes the WBR from the facts JSON only | 🤖 |
| **Number guard** | Pulls every figure out of the LLM draft and **rejects the draft if any number isn't in the facts**, then falls back to the rule-based narrative | ✅ |
| **Delivery** (`wbr/report.py`) | Markdown report, charts, **Slack Block Kit** payload, optional webhook post | ✅ |

## Sample output

The synthetic feed has three planted incidents in the latest week. The copilot finds all three and **ranks them by ₹ impact**:

> **Headline:** GMV ₹65.32 Cr (-1.0% WoW), orders 1,617,493 (+0.9%), AOV ₹404 (-1.9%).
>
> 1. **Mumbai – AOV** fell to ₹416 vs ₹457 baseline (-8.9%, impact -₹1.45 Cr / week). Orders moved +1.8% WoW, so this is a basket-size issue, not demand. → *Category: audit menu pricing, combo availability and top-merchant basket mix*
> 2. **Bengaluru – Promo % of GMV** rose to 13.8% vs 8.8% baseline (+55.6%, impact -₹77.0 L / week). Orders only +3.7% WoW against much higher spend, so the incremental orders are expensive. → *Growth: set a promo-as-%-of-GMV guardrail*
> 3. **Kolkata – Cancellation rate** rose to 9.2% vs 3.9% baseline (+135.0%, impact -₹26.1 L / week). Moving together with delivery time (+19.8%), which points to a supply-side ops incident. → *Ops: check rider supply and restaurant acceptance by zone*

Full report: [`examples/WBR.md`](examples/WBR.md) · facts payload: [`examples/facts.json`](examples/facts.json) · Slack message: [`examples/slack_payload.json`](examples/slack_payload.json)

**Anomaly scan: 6 cities × 8 metrics, one glance:**

![Anomaly heatmap](examples/figures/anomaly_heatmap.png)

**GMV bridge: is it demand or basket size?** Mumbai's orders went *up*, so the AOV drop is the whole story:

![GMV bridge](examples/figures/gmv_bridge.png)

---

## Run it

```bash
git clone https://github.com/<your-username>/ai-business-review-copilot.git
cd ai-business-review-copilot
pip install -r requirements.txt

python -m wbr --narrator rules          # no API key needed
pytest -q                               # 12 tests, LLM is faked
```

### With an LLM

```bash
pip install langchain-core langchain-openai          # or langchain-anthropic
cp .env.example .env                                  # add your key locally, never commit it
export OPENAI_API_KEY=...                             # or LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY
python -m wbr --narrator llm
```

`--narrator auto` (the default) uses the LLM when a key is present, otherwise the rule-based narrator. Set `LLM_MODEL` to pick the model.

### Your own data

```bash
python -m wbr --csv my_daily_kpis.csv
# columns: date, city, orders, cancelled_orders, gmv, promo_spend, new_users, avg_delivery_min
```

### Post to Slack

```bash
export SLACK_WEBHOOK_URL=...      # from your Slack app; read from env, never written to disk
python -m wbr --post-slack
```

## Design notes

- **Robust z, not plain z.** Median/MAD baselines don't get dragged around by last month's incident, so the next one still trips the alarm.
- **Direction-aware alerts.** Only *bad* moves alert (cancellations up, AOV down). A good week doesn't page anyone.
- **One issue per city×theme.** Cancellation rate and delivery time spiking together is one ops incident, not two alerts. The second becomes supporting evidence.
- **Rank by ₹, not by z.** Bengaluru's promo z-score (38.8) is bigger than Mumbai's AOV z-score (14.9), but Mumbai costs about 2× more per week, so it goes first.
- **LMDI decomposition.** Orders effect + AOV effect add up *exactly* to the GMV change, with no unexplained residual (unit-tested).

## Limitations

- The number guard checks *that* a number exists in the facts, not that it's attached to the right metric. The next step is claim-level checking (city + metric + value).
- Weekly granularity only. Add zone-level drill-down to localise ops incidents.

---

**Author:** Siddhant Srivastava · [LinkedIn](https://linkedin.com/in/SiddhantSrivastava)
