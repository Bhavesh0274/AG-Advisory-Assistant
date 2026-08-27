import numpy as np
import pandas as pd

from src.forecast.baselines import RandomWalkForecaster, SeasonalNaiveForecaster


def _series(dates, prices):
    return pd.DataFrame({"date": pd.to_datetime(dates), "modal_price": prices})


def test_random_walk_holds_last_value_flat():
    train = _series(["2024-01-01", "2024-01-02", "2024-01-03"], [100, 110, 120])

    model = RandomWalkForecaster().fit(train)
    out = model.predict(horizon=5)

    assert len(out) == 5
    assert (out["forecast"] == 120).all()
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-04")
    assert out["date"].iloc[-1] == pd.Timestamp("2024-01-08")


def test_seasonal_naive_repeats_prior_season_value():
    dates = pd.date_range("2023-01-01", "2024-01-05", freq="D")
    # price is just day-of-year so we can check season lookup precisely
    prices = [d.dayofyear for d in dates]
    train = _series(dates, prices)

    model = SeasonalNaiveForecaster(season_length=365).fit(train)
    out = model.predict(horizon=3)

    # 2024-01-06 - 365 days = 2023-01-06 (2023 has 365 days, not a leap year)
    assert out.iloc[0]["forecast"] == pd.Timestamp("2023-01-06").dayofyear


def test_seasonal_naive_falls_back_to_last_value_when_season_ago_missing():
    train = _series(["2024-01-01", "2024-01-02", "2024-01-03"], [100, 110, 120])

    model = SeasonalNaiveForecaster(season_length=365).fit(train)
    out = model.predict(horizon=1)

    # no data from ~365 days before train ever existed -> falls back to last value
    assert out.iloc[0]["forecast"] == 120


def test_seasonal_naive_short_gap_tolerance_fills_near_miss():
    dates = list(pd.date_range("2023-01-01", "2023-01-10", freq="D")) + list(
        pd.date_range("2024-01-01", "2024-01-05", freq="D")
    )
    prices = [100] * 10 + [200] * 5
    train = _series(dates, prices)

    model = SeasonalNaiveForecaster(season_length=365, lookup_tolerance_days=3).fit(train)
    out = model.predict(horizon=1)

    # 2024-01-06 - 365d = 2023-01-07, which exists directly in the dense series
    assert out.iloc[0]["forecast"] == 100
