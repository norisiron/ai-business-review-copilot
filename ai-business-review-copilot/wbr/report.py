"""Render the WBR: markdown report, charts and a Slack Block Kit payload."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import metrics as m
from .facts import LABEL

NAVY, CORAL, GREY, GREEN = "#1F3A5F", "#E4572E", "#9AA5B1", "#2E8B57"
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12})


def charts(weekly: pd.DataFrame, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. GMV trend by city
    fig, ax = plt.subplots(figsize=(9, 4))
    piv = weekly.pivot(index="week", columns="city", values="gmv") / 1e7
    for city in piv.columns:
        ax.plot(piv.index, piv[city], marker="o", ms=3, label=city)
    ax.axvspan(piv.index[-2], piv.index[-1], color=CORAL, alpha=0.08)
    ax.set_ylabel("Weekly GMV (₹ Cr)")
    ax.set_title("Weekly GMV by city (latest week shaded)")
    ax.legend(frameon=False, ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    p = out / "gmv_trend.png"; fig.savefig(p); plt.close(fig); paths.append(p)

    # 2. GMV bridge: orders vs AOV effect
    b = m.gmv_bridge(weekly)
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(b))
    ax.barh(y, b.orders_effect / 1e5, color=NAVY, label="Orders effect")
    ax.barh(y, b.aov_effect / 1e5, left=np.where(np.sign(b.aov_effect) == np.sign(b.orders_effect),
                                                  b.orders_effect / 1e5, 0), color=CORAL, label="AOV effect")
    ax.scatter(b.gmv_change / 1e5, y, color="black", zorder=3, s=18, label="Net GMV change")
    ax.set_yticks(y, b.city)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("WoW GMV change (₹ lakh)")
    ax.set_title("GMV bridge: is it demand or basket size?")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    p = out / "gmv_bridge.png"; fig.savefig(p); plt.close(fig); paths.append(p)

    # 3. Anomaly heatmap (robust z, signed so that red = bad)
    rows = []
    for city, g in weekly.sort_values("week").groupby("city"):
        hist, cur = g.iloc[-9:-1], g.iloc[-1]
        for metric, bad in m.BAD_DIRECTION.items():
            med = hist[metric].median()
            mad = max((hist[metric] - med).abs().median() * 1.4826, abs(med) * 0.005)
            rows.append({"city": city, "metric": LABEL[metric], "z": bad * (cur[metric] - med) / mad})
    z = pd.DataFrame(rows).pivot(index="city", columns="metric", values="z")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(np.clip(z.values, -10, 10), cmap="RdYlGn_r", vmin=-10, vmax=10, aspect="auto")
    for (i, j), v in np.ndenumerate(z.values):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                fontweight="bold" if v >= 3.5 else "normal")
    ax.set_xticks(range(z.shape[1]), z.columns, rotation=25, ha="right")
    ax.set_yticks(range(z.shape[0]), z.index)
    ax.set_title("Anomaly scan: robust z-score vs 8-week baseline (red = bad, alert ≥ 3.5)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    p = out / "anomaly_heatmap.png"; fig.savefig(p); plt.close(fig); paths.append(p)
    return paths


def markdown(facts: dict, narrative: str, narrator: str, figures: list[Path], root: Path) -> str:
    h = facts["headline"]
    kpis = "\n".join(
        f"| {LABEL.get(k, k)} | {v['display']} | {v['wow_display']} | {v['vs_4w_display']} |" for k, v in h.items())
    figs = "\n\n".join(f"![{p.stem}]({p.relative_to(root).as_posix()})" for p in figures)
    return f"""# Weekly Business Review: week of {facts['week_start']}

_Narrative by: `{narrator}` narrator · every number is traced back to the KPI engine._

| KPI | This week | WoW | vs 4-wk avg |
|---|---|---|---|
{kpis}

## Narrative

{narrative}

## Charts

{figs}
"""


def slack_payload(facts: dict, narrative: str) -> dict:
    h = facts["headline"]
    head = (f"*GMV* {h['gmv']['display']} ({h['gmv']['wow_display']})  ·  "
            f"*Orders* {h['orders']['display']} ({h['orders']['wow_display']})  ·  "
            f"*AOV* {h['aov']['display']} ({h['aov']['wow_display']})")
    issues = "\n".join(
        f"{'🔴' if i['severity'] == 'high' else '🟠'} *{i['city']}*: {i['metric_label']} "
        f"{i['direction_word']} to {i['current_display']} ({i['change_display']}) → {i['action'].split(':')[0]}"
        for i in facts["issues"][:3]) or "✅ No anomalies this week."
    return {"text": f"WBR {facts['week_start']}", "blocks": [
        {"type": "header", "text": {"type": "plain_text", "text": f"📊 WBR: week of {facts['week_start']}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": head}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": "*Top issues*\n" + issues}},
    ]}


def post_to_slack(payload: dict) -> None:
    """Posts only if SLACK_WEBHOOK_URL is set. The URL is never logged or written to disk."""
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        raise RuntimeError("SLACK_WEBHOOK_URL is not set")
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 (https webhook supplied by user)
        r.read()
