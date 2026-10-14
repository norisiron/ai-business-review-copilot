"""Build the grounded facts payload that both narrators consume."""
from __future__ import annotations

import pandas as pd

from . import metrics as m

LABEL = {"gmv": "GMV", "orders": "Orders", "aov": "AOV", "new_users": "New users",
         "cancel_rate": "Cancellation rate", "avg_delivery_min": "Delivery time",
         "promo_per_order": "Promo cost per order", "promo_pct_gmv": "Promo % of GMV"}

ACTION = {
    "aov": "Category: audit menu pricing, combo availability and top-merchant basket mix",
    "cancel_rate": "Ops: check rider supply and restaurant acceptance by zone, daily war-room until back to baseline",
    "avg_delivery_min": "Ops: rebalance rider fleet to peak-slot zones, review surge config",
    "promo_per_order": "Growth: cap coupon depth and re-target to dormant users, pause blanket offers",
    "promo_pct_gmv": "Growth: set a promo-as-%-of-GMV guardrail and review the campaign calendar",
    "orders": "Growth + Ops: split the decline by new vs repeat users and by zone",
    "gmv": "City head: run the GMV bridge by zone to isolate the driver",
    "new_users": "Growth: check paid-channel delivery and referral funnel",
}


def inr(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e7:
        return f"{sign}₹{x/1e7:.2f} Cr"
    if x >= 1e5:
        return f"{sign}₹{x/1e5:.1f} L"
    return f"{sign}₹{x:,.0f}"


def signed_inr(x: float) -> str:
    return ("+" if x >= 0 else "") + inr(x)


def pct(x: float, signed: bool = True) -> str:
    return f"{x*100:+.1f}%" if signed else f"{x*100:.1f}%"


def fmt_metric(metric: str, v: float) -> str:
    if metric in {"gmv"}:
        return inr(v)
    if metric in {"aov", "promo_per_order"}:
        return f"₹{v:,.0f}"
    if metric in {"cancel_rate", "promo_pct_gmv"}:
        return pct(v, signed=False)
    if metric == "avg_delivery_min":
        return f"{v:.1f} min"
    return f"{v:,.0f}"


def _driver(a: m.Anomaly, bridge: pd.DataFrame, related: list[m.Anomaly], weekly: pd.DataFrame) -> str:
    b = bridge.set_index("city").loc[a.city]
    cw = m.wow(weekly[weekly.city == a.city], ["orders", "aov"]).set_index("metric")
    if a.metric == "aov":
        return (f"Orders moved {pct(cw.loc['orders','wow_pct'])} WoW, so this is a basket-size issue, not demand. "
                f"AOV effect on GMV: {signed_inr(b.aov_effect)}.")
    if a.metric in {"cancel_rate", "avg_delivery_min"}:
        other = [r for r in related if r.metric in {"cancel_rate", "avg_delivery_min"} and r.metric != a.metric]
        tail = (f" Moving together with {LABEL[other[0].metric].lower()} ({pct(other[0].change_pct)}), "
                "which points to a supply-side ops incident.") if other else ""
        return f"Delivered orders {pct(cw.loc['orders','wow_pct'])} WoW, GMV impact {signed_inr(b.gmv_change)}.{tail}"
    if a.metric in {"promo_per_order", "promo_pct_gmv"}:
        return (f"Orders only {pct(cw.loc['orders','wow_pct'])} WoW against much higher spend, "
                "so the incremental orders are expensive.")
    return f"GMV change {signed_inr(b.gmv_change)} (orders {signed_inr(b.orders_effect)}, AOV {signed_inr(b.aov_effect)})."


def _impact(a: m.Anomaly, bridge: pd.DataFrame, weekly: pd.DataFrame) -> float:
    """Rough weekly ₹ impact, used only to rank issues."""
    b = bridge.set_index("city").loc[a.city]
    if a.metric == "aov":
        return float(b.aov_effect)
    if a.metric in {"promo_per_order", "promo_pct_gmv"}:
        g = weekly[weekly.city == a.city].sort_values("week")
        return -float(g.promo_spend.iloc[-1] - g.promo_spend.iloc[-9:-1].median())
    return float(b.gmv_change)


def build_facts(weekly: pd.DataFrame, max_issues: int = 5) -> dict:
    nat = m.national(weekly)
    hw = m.wow(nat, ["gmv", "orders", "aov", "cancel_rate", "promo_pct_gmv", "new_users"]).set_index("metric")
    bridge = m.gmv_bridge(weekly)
    anomalies = m.detect_anomalies(weekly)

    # one issue per city+theme: keep the strongest signal, use the rest as context
    issues, seen = [], set()
    for a in anomalies:
        theme = {"promo_pct_gmv": "promo", "promo_per_order": "promo",
                 "cancel_rate": "ops", "avg_delivery_min": "ops"}.get(a.metric, a.metric)
        if (a.city, theme) in seen:
            continue
        seen.add((a.city, theme))
        related = [r for r in anomalies if r.city == a.city and r is not a]
        issues.append({
            "city": a.city, "metric": a.metric, "metric_label": LABEL[a.metric],
            "severity": a.severity, "robust_z": round(a.robust_z, 1),
            "direction_word": "rose" if a.direction == "up" else "fell",
            "current_display": fmt_metric(a.metric, a.current),
            "baseline_display": fmt_metric(a.metric, a.baseline),
            "change_display": pct(a.change_pct),
            "impact_inr": _impact(a, bridge, weekly),
            "driver": _driver(a, bridge, related, weekly),
            "action": ACTION[a.metric],
        })
    issues.sort(key=lambda i: i["impact_inr"])          # most negative ₹ impact first
    issues = issues[:max_issues]
    for i in issues:
        i["impact_display"] = signed_inr(i.pop("impact_inr")) + " / week"

    week = pd.Timestamp(nat.week.max())
    return {
        "week_start": week.strftime("%d %b %Y"),
        "headline": {
            k: {"display": fmt_metric(k, hw.loc[k, "current"]), "wow_display": pct(hw.loc[k, "wow_pct"]),
                "vs_4w_display": pct(hw.loc[k, "vs_4w_pct"])}
            for k in hw.index
        },
        "issues": issues,
        "gmv_bridge": [
            {"city": r.city, "gmv_change_display": signed_inr(r.gmv_change),
             "orders_effect_display": signed_inr(r.orders_effect), "aov_effect_display": signed_inr(r.aov_effect)}
            for r in bridge.itertuples()
        ],
    }
