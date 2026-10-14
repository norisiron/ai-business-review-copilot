"""
Synthetic daily KPI feed for a multi-city delivery business (12 weeks).

Every number is fake and seeded. Three incidents are planted in the final week so the
diagnostics have something real to find:
  1. Mumbai     - AOV drops ~9% (basket-size problem, orders are fine)
  2. Kolkata    - cancellation rate spikes from ~4% to ~9% (ops incident)
  3. Bengaluru  - promo spend jumps ~60% with only a small order lift (inefficient burn)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CITIES = {  # base daily orders, base AOV (INR)
    "Delhi NCR": (52_000, 410), "Mumbai": (46_000, 455), "Bengaluru": (44_000, 430),
    "Hyderabad": (30_000, 380), "Pune": (22_000, 395), "Kolkata": (18_000, 350),
}
N_WEEKS = 12


def generate(seed: int = 7, start: str = "2026-07-06") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=N_WEEKS * 7, freq="D")
    last_week = dates[-7:]
    rows = []
    for city, (base_orders, base_aov) in CITIES.items():
        trend = np.linspace(1.0, 1.08, len(dates))                      # steady growth
        weekly = 1 + 0.12 * np.isin(dates.dayofweek, [4, 5, 6])          # weekend peak
        orders = base_orders * trend * weekly * rng.normal(1, 0.025, len(dates))
        aov = base_aov * rng.normal(1, 0.015, len(dates))
        cancel_rate = np.clip(rng.normal(0.04, 0.004, len(dates)), 0.02, None)
        promo_per_order = rng.normal(38, 2, len(dates))
        new_users = orders * rng.normal(0.06, 0.004, len(dates))
        delivery_min = rng.normal(31, 1.0, len(dates))

        lw = np.isin(dates, last_week)
        if city == "Mumbai":
            aov = np.where(lw, aov * 0.91, aov)
        if city == "Kolkata":
            cancel_rate = np.where(lw, cancel_rate + 0.05, cancel_rate)
            delivery_min = np.where(lw, delivery_min + 6, delivery_min)
        if city == "Bengaluru":
            promo_per_order = np.where(lw, promo_per_order * 1.55, promo_per_order)
            orders = np.where(lw, orders * 1.03, orders)

        delivered = orders * (1 - cancel_rate)
        rows.append(pd.DataFrame({
            "date": dates, "city": city,
            "orders": delivered.round().astype(int),
            "cancelled_orders": (orders * cancel_rate).round().astype(int),
            "gmv": (delivered * aov).round(0),
            "promo_spend": (delivered * promo_per_order).round(0),
            "new_users": new_users.round().astype(int),
            "avg_delivery_min": delivery_min.round(1),
        }))
    return pd.concat(rows, ignore_index=True)
