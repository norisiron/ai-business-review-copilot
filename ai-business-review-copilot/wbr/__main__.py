"""
CLI:  python -m wbr [--narrator auto|rules|llm] [--csv path] [--out examples] [--post-slack]

Without --csv, a synthetic 12-week KPI feed is generated.
CSV schema: date, city, orders, cancelled_orders, gmv, promo_spend, new_users, avg_delivery_min
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import metrics as m
from .data import generate
from .facts import build_facts
from .narrator import pick_narrator
from .report import charts, markdown, post_to_slack, slack_payload


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Business Review Copilot")
    ap.add_argument("--csv", help="daily KPI CSV (defaults to synthetic data)")
    ap.add_argument("--narrator", choices=["auto", "rules", "llm"], default="auto")
    ap.add_argument("--out", default="examples")
    ap.add_argument("--post-slack", action="store_true", help="post summary to SLACK_WEBHOOK_URL")
    a = ap.parse_args()

    daily = pd.read_csv(a.csv, parse_dates=["date"]) if a.csv else generate()
    weekly = m.to_weekly(daily)
    facts = build_facts(weekly)

    narrator = pick_narrator(a.narrator)
    narrative = narrator.write(facts)

    out = Path(a.out)
    figs = charts(weekly, out / "figures")
    (out / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "WBR.md").write_text(markdown(facts, narrative, narrator.name, figs, out), encoding="utf-8")
    payload = slack_payload(facts, narrative)
    (out / "slack_payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(narrative)
    print(f"\n[{narrator.name}] wrote {out}/WBR.md, facts.json, slack_payload.json, figures/")
    if a.post_slack:
        post_to_slack(payload)
        print("posted to Slack")


if __name__ == "__main__":
    main()
