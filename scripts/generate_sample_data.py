"""Generate reproducible sample datasets with deliberately injected quality issues.

Usage::

    python scripts/generate_sample_data.py

Writes ``sample_data/customer_orders.csv`` and ``sample_data/sensor_readings.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "sample_data"


def build_orders(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Customer orders with missing values, invalid types/dates, outliers and duplicates."""
    rng = np.random.default_rng(seed)
    order_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    ship_delay = rng.integers(1, 10, n)
    df = pd.DataFrame(
        {
            "order_id": np.arange(100001, 100001 + n),
            "customer_id": [f"CUST-{i:04d}" for i in rng.integers(1, 180, n)],
            "customer_email": [f"user{i}@example.com" for i in rng.integers(1, 180, n)],
            "order_date": order_dates.strftime("%Y-%m-%d"),
            "ship_date": (order_dates + pd.to_timedelta(ship_delay, unit="D")).strftime("%Y-%m-%d"),
            "product_category": rng.choice(["Electronics", "Home", "Apparel", "Sports", "Books"], n,
                                           p=[0.3, 0.25, 0.2, 0.15, 0.1]),
            "quantity": rng.integers(1, 6, n).astype(object),
            "unit_price": np.round(rng.lognormal(3.5, 0.5, n), 2).astype(object),
            "discount_pct": np.round(rng.choice([0, 5, 10, 15, 20], n), 1),
            "is_returned": rng.choice(["Yes", "No"], n, p=[0.08, 0.92]),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "payment_method": rng.choice(["card", "paypal", "bank_transfer", "cash"], n),
        }
    )
    # Missing values.
    df.loc[rng.choice(n, 35, replace=False), "customer_email"] = np.nan
    df.loc[rng.choice(n, 12, replace=False), "unit_price"] = np.nan
    df.loc[rng.choice(n, 4, replace=False), "region"] = "   "
    # Invalid types.
    df.loc[[17, 230, 411], "quantity"] = ["two", "N/A?", "3 units"]
    df.loc[[58, 309], "unit_price"] = ["free", "12,99"]
    # Invalid dates: impossible, unparseable, too old, too far in the future.
    df.loc[[5, 77, 140, 260, 333], "order_date"] = ["2024-02-30", "not a date", "1850-06-01", "2099-01-01", "2024-13-01"]
    df.loc[[90, 91], "ship_date"] = ["TBD", "pending"]
    # Outliers.
    df.loc[[25, 199, 444], "quantity"] = [250, 400, 1000]
    df.loc[[60, 380], "unit_price"] = [9999.0, 15000.0]
    # Exact duplicates.
    duplicates = df.iloc[[10, 11, 12, 100, 101, 250]]
    return pd.concat([df, duplicates], ignore_index=True)


def build_sensor_readings(n: int = 120, seed: int = 7) -> list:
    """Nested JSON sensor readings."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-06-01T00:00:00Z")
    records = []
    for i in range(n):
        temperature = float(np.round(rng.normal(21.5, 1.2), 2))
        if i in (33, 87):
            temperature = 85.0  # sensor spike
        records.append(
            {
                "reading_id": f"R{i:05d}",
                "timestamp": (start + pd.Timedelta(minutes=15 * i)).isoformat(),
                "sensor": {"id": f"S-{(i % 4) + 1}", "location": ["lab", "warehouse", "office", "dock"][i % 4]},
                "temperature_c": temperature if i not in (12, 13) else None,
                "humidity_pct": float(np.round(rng.uniform(30, 60), 1)),
                "status": "ok" if i % 25 else "degraded",
                "calibrated": bool(i % 10),
            }
        )
    return records


def main() -> None:
    """Write the sample datasets to ``sample_data/``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    orders_path = OUTPUT_DIR / "customer_orders.csv"
    build_orders().to_csv(orders_path, index=False)
    sensors_path = OUTPUT_DIR / "sensor_readings.json"
    sensors_path.write_text(json.dumps({"data": build_sensor_readings()}, indent=2))
    print(f"Wrote {orders_path}\nWrote {sensors_path}")


if __name__ == "__main__":
    main()
