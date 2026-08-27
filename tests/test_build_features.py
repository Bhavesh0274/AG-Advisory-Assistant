import numpy as np
import pandas as pd

from src.data.build_features import (
    add_calendar_features,
    add_cross_mandi_spread,
    add_lag_rolling_features,
    resample_daily,
    treat_outliers,
)


def _prices(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_resample_daily_fills_short_gap_and_flags_long_gap():
    rows = [
        {"commodity": "Onion", "market": "Pune", "date": d, "modal_price": p}
        for d, p in [
            ("2024-01-01", 100),
            ("2024-01-02", 110),
            # gap of 2 days (short, fillable)
            ("2024-01-05", 120),
            # gap of 5 days (long, not fillable)
            ("2024-01-11", 130),
        ]
    ]
    out = resample_daily(_prices(rows))

    assert len(out) == 11  # full daily range 01-01..01-11
    jan3 = out[out["date"] == "2024-01-03"].iloc[0]
    assert jan3["is_imputed"]
    assert jan3["modal_price"] == 110  # ffilled from Jan 2

    jan7 = out[out["date"] == "2024-01-07"].iloc[0]
    assert not jan7["is_imputed"]
    assert pd.isna(jan7["modal_price"])


def test_treat_outliers_flags_extreme_spike():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    prices = [100.0] * 15 + [1000.0] + [100.0] * 4  # one extreme spike
    rows = [
        {"commodity": "Onion", "market": "Pune", "date": d, "modal_price": p}
        for d, p in zip(dates, prices)
    ]
    df = _prices(rows)

    out = treat_outliers(df)

    assert out.loc[out["modal_price"] == 1000.0, "is_outlier"].all()
    assert not out.loc[out["modal_price"] == 100.0, "is_outlier"].any()


def test_add_calendar_features_flags_festival_window():
    rows = [
        {"date": "2023-11-12"},  # Diwali 2023 itself
        {"date": "2023-11-20"},  # far from any festival
    ]
    df = _prices(rows)

    out = add_calendar_features(df)

    assert out.iloc[0]["is_festival_window"]
    assert not out.iloc[1]["is_festival_window"]
    assert out.iloc[0]["month"] == 11


def test_add_lag_rolling_features_shifts_correctly():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    rows = [
        {"commodity": "Onion", "market": "Pune", "date": d, "modal_price": p}
        for d, p in zip(dates, [10, 20, 30, 40, 50])
    ]
    df = _prices(rows)

    out = add_lag_rolling_features(df)

    np.testing.assert_allclose(
        out["price_lag_1"].tolist(), [np.nan, 10, 20, 30, 40], equal_nan=True
    )


def test_add_cross_mandi_spread_uses_peer_markets_only():
    rows = [
        {"commodity": "Onion", "market": "Pune", "date": "2024-01-01", "modal_price": 100},
        {"commodity": "Onion", "market": "Agra", "date": "2024-01-01", "modal_price": 200},
        {"commodity": "Onion", "market": "Indore", "date": "2024-01-01", "modal_price": 300},
    ]
    df = _prices(rows)

    out = add_cross_mandi_spread(df)

    pune_row = out[out["market"] == "Pune"].iloc[0]
    # Pune's spread vs the mean of its peers (Agra=200, Indore=300) -> 100 - 250 = -150
    assert pune_row["cross_mandi_spread"] == -150
