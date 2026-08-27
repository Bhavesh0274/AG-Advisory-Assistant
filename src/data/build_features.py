"""Build analysis-ready features from data/processed/prices.parquet.

Per commodity x market: resample to daily frequency, impute short gaps,
flag long ones, treat outliers, and engineer calendar, lag/rolling, and
cross-mandi spread features. Arrivals-based features are skipped: no data
source wired up so far provides arrivals (see README.md "Data sources").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.config import DATA_PROCESSED, REPORTS  # noqa: E402

SHORT_GAP_DAYS = 3
LAG_DAYS = (1, 7, 14, 30)
ROLLING_WINDOWS = (7, 14, 30)
OUTLIER_Z_THRESH = 4.0

# Major pan-India festival windows likely to move demand (+-2 days each) for
# the years covered by the current price series (2023-2025). Hand-maintained
# since there's no festival-calendar data source wired up yet.
FESTIVAL_DATES = pd.to_datetime(
    [
        "2023-10-24",  # Dussehra 2023
        "2023-11-12",  # Diwali 2023
        "2024-03-25",  # Holi 2024
        "2024-10-12",  # Dussehra 2024
        "2024-11-01",  # Diwali 2024
        "2025-03-14",  # Holi 2025
        "2025-10-02",  # Dussehra 2025
        "2025-10-21",  # Diwali 2025
    ]
)

# Rough Indian crop-season bucketing by month, used as a calendar feature
# rather than a per-commodity sowing/harvest calendar (which would need a
# real agronomic reference we don't have yet).
SEASON_BY_MONTH = {
    1: "rabi", 2: "rabi", 3: "rabi_harvest",
    4: "zaid", 5: "zaid", 6: "kharif_sowing",
    7: "kharif", 8: "kharif", 9: "kharif",
    10: "kharif_harvest", 11: "rabi_sowing", 12: "rabi",
}


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (commodity, market, date); short gaps ffilled, long gaps flagged."""
    out = []
    for (commodity, market), g in df.groupby(["commodity", "market"]):
        g = g.groupby("date", as_index=False)["modal_price"].mean()
        full_idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(full_idx)
        g.index.name = "date"

        g["is_observed"] = g["modal_price"].notna()
        gap_id = g["is_observed"].cumsum()
        gap_len = g.groupby(gap_id)["is_observed"].transform("size")
        fillable = (~g["is_observed"]) & (gap_len <= SHORT_GAP_DAYS)

        g["modal_price"] = g["modal_price"].ffill(limit=SHORT_GAP_DAYS).where(
            g["is_observed"] | fillable
        )
        g["is_imputed"] = fillable
        g["commodity"] = commodity
        g["market"] = market
        out.append(g.reset_index())

    return pd.concat(out, ignore_index=True)


def treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag (not drop) prices whose z-score vs a trailing window exceeds OUTLIER_Z_THRESH.

    The baseline window excludes the point being tested (via shift(1)) - including
    it would pull the window's own mean/std toward the spike and mask it.
    """
    df = df.sort_values(["commodity", "market", "date"]).copy()
    shifted = df.groupby(["commodity", "market"])["modal_price"].shift(1)
    grouped_shifted = shifted.groupby([df["commodity"], df["market"]])

    roll_mean = grouped_shifted.transform(lambda s: s.rolling(14, min_periods=5).mean())
    roll_std = grouped_shifted.transform(lambda s: s.rolling(14, min_periods=5).std())
    z = (df["modal_price"] - roll_mean) / roll_std.replace(0, np.nan)
    zero_std_break = (roll_std == 0) & df["modal_price"].ne(roll_mean) & roll_mean.notna()
    df["is_outlier"] = ((z.abs() > OUTLIER_Z_THRESH) | zero_std_break).fillna(False)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["season"] = df["month"].map(SEASON_BY_MONTH)
    near_festival = pd.Series(False, index=df.index)
    for fd in FESTIVAL_DATES:
        near_festival |= (df["date"] - fd).abs() <= pd.Timedelta(days=2)
    df["is_festival_window"] = near_festival
    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["commodity", "market", "date"]).copy()
    grouped = df.groupby(["commodity", "market"])["modal_price"]

    for lag in LAG_DAYS:
        df[f"price_lag_{lag}"] = grouped.shift(lag)

    for window in ROLLING_WINDOWS:
        shifted = grouped.shift(1)
        df[f"price_roll_mean_{window}"] = shifted.groupby(
            [df["commodity"], df["market"]]
        ).transform(lambda s: s.rolling(window, min_periods=max(3, window // 3)).mean())
        df[f"price_roll_std_{window}"] = shifted.groupby(
            [df["commodity"], df["market"]]
        ).transform(lambda s: s.rolling(window, min_periods=max(3, window // 3)).std())

    return df


def add_cross_mandi_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Each market's price minus the same-day mean across its peer markets."""
    df = df.copy()
    peer_mean = df.groupby(["commodity", "date"])["modal_price"].transform("mean")
    peer_count = df.groupby(["commodity", "date"])["modal_price"].transform("count")
    df["cross_mandi_spread"] = np.where(
        peer_count > 1,
        df["modal_price"] - (peer_mean * peer_count - df["modal_price"]) / (peer_count - 1),
        np.nan,
    )
    return df


def build_seasonality_report(df: pd.DataFrame) -> str:
    lines = ["# EDA: Seasonality and Price Coverage", ""]
    lines.append(
        "Arrivals-vs-price analysis skipped: no data source wired up so far "
        "provides arrivals volume (see README.md Data sources)."
    )
    lines.append("")
    lines.append("## Average modal price by month, per commodity x market")
    lines.append("")
    monthly = (
        df.groupby(["commodity", "market", "month"])["modal_price"]
        .mean()
        .round(1)
        .unstack("month")
        .sort_index()
    )
    lines.append(monthly.to_markdown())
    lines.append("")
    lines.append("## Data quality per commodity x market")
    lines.append("")
    quality = df.groupby(["commodity", "market"]).agg(
        n_days=("date", "count"),
        pct_imputed=("is_imputed", lambda s: round(100 * s.mean(), 1)),
        pct_outlier=("is_outlier", lambda s: round(100 * s.mean(), 1)),
    )
    lines.append(quality.to_markdown())
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    prices = pd.read_parquet(DATA_PROCESSED / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    daily = resample_daily(prices)
    daily = treat_outliers(daily)
    daily = add_calendar_features(daily)
    daily = add_lag_rolling_features(daily)
    daily = add_cross_mandi_spread(daily)

    out_path = DATA_PROCESSED / "features.parquet"
    daily.to_parquet(out_path, index=False)
    print(f"Saved {len(daily)} rows x {len(daily.columns)} cols to {out_path}")

    report = build_seasonality_report(daily)
    report_path = REPORTS / "eda_seasonality.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
