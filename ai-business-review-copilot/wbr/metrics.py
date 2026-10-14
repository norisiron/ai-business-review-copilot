"""Weekly KPI engine: WoW deltas, GMV driver decomposition, robust anomaly detection."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["week"] = d["date"].dt.to_period("W-SUN").dt.start_time
    w = d.groupby(["week", "city"], as_index=False).agg(
        orders=("orders", "sum"), cancelled_orders=("cancelled_orders", "sum"),
        gmv=("gmv", "sum"), promo_spend=("promo_spend", "sum"),
        new_users=("new_users", "sum"), avg_delivery_min=("avg_delivery_min", "mean"),
    )
    return add_ratios(w)


def add_ratios(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    w["aov"] = w.gmv / w.orders
    w["cancel_rate"] = w.cancelled_orders / (w.orders + w.cancelled_orders)
    w["promo_pct_gmv"] = w.promo_spend / w.gmv
    w["promo_per_order"] = w.promo_spend / w.orders
    return w


def national(weekly: pd.DataFrame) -> pd.DataFrame:
    n = weekly.groupby("week", as_index=False).agg(
        orders=("orders", "sum"), cancelled_orders=("cancelled_orders", "sum"),
        gmv=("gmv", "sum"), promo_spend=("promo_spend", "sum"),
        new_users=("new_users", "sum"), avg_delivery_min=("avg_delivery_min", "mean"),
    )
    n["city"] = "All India"
    return add_ratios(n)


def wow(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Latest week vs previous week and vs trailing 4-week average, per city."""
    out = []
    for city, g in df.sort_values("week").groupby("city"):
        cur, prev, base4 = g.iloc[-1], g.iloc[-2], g.iloc[-5:-1]
        for m in metrics:
            out.append({
                "city": city, "metric": m, "current": cur[m], "previous": prev[m],
                "wow_pct": cur[m] / prev[m] - 1 if prev[m] else np.nan,
                "vs_4w_pct": cur[m] / base4[m].mean() - 1 if base4[m].mean() else np.nan,
            })
    return pd.DataFrame(out)


def gmv_bridge(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose each city's WoW GMV change into an orders effect and an AOV effect
    (LMDI log-mean split, so the two parts add up exactly to the total change).
    """
    rows = []
    for city, g in weekly.sort_values("week").groupby("city"):
        p, c = g.iloc[-2], g.iloc[-1]
        d_gmv = c.gmv - p.gmv
        # LMDI: logarithmic mean weight makes the two effects sum exactly to d_gmv
        lm = d_gmv / np.log(c.gmv / p.gmv) if c.gmv != p.gmv else c.gmv
        rows.append({"city": city, "gmv_prev": p.gmv, "gmv_curr": c.gmv, "gmv_change": d_gmv,
                     "orders_effect": lm * np.log(c.orders / p.orders),
                     "aov_effect": lm * np.log(c.aov / p.aov)})
    return pd.DataFrame(rows).sort_values("gmv_change")


@dataclass
class Anomaly:
    city: str
    metric: str
    current: float
    baseline: float
    change_pct: float
    robust_z: float
    direction: str
    severity: str

    def as_dict(self) -> dict:
        return asdict(self)


# A metric moving in the "bad" direction is what we alert on.
BAD_DIRECTION = {"gmv": -1, "orders": -1, "aov": -1, "new_users": -1,
                 "cancel_rate": +1, "avg_delivery_min": +1, "promo_per_order": +1, "promo_pct_gmv": +1}


def detect_anomalies(weekly: pd.DataFrame, lookback: int = 8, z_threshold: float = 3.5) -> list[Anomaly]:
    """
    Robust z-score of the latest week against the previous `lookback` weeks
    (median / MAD, so one bad week in the baseline doesn't hide the next one).
    """
    found = []
    for city, g in weekly.sort_values("week").groupby("city"):
        hist, cur = g.iloc[-lookback - 1:-1], g.iloc[-1]
        for m, bad in BAD_DIRECTION.items():
            med = hist[m].median()
            mad = (hist[m] - med).abs().median() * 1.4826
            mad = max(mad, abs(med) * 0.005)          # floor to avoid divide-by-~0
            z = (cur[m] - med) / mad
            if abs(z) >= z_threshold and np.sign(z) == bad:
                found.append(Anomaly(city, m, float(cur[m]), float(med), float(cur[m] / med - 1),
                                     float(z), "up" if z > 0 else "down",
                                     "high" if abs(z) >= 2 * z_threshold else "medium"))
    return sorted(found, key=lambda a: -abs(a.robust_z))
