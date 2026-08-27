import numpy as np
import pandas as pd
import pytest

from src.data.build_features import (
    add_calendar_features,
    add_lag_rolling_features,
    resample_daily,
)
from src.forecast.lightgbm_model import MIN_TRAIN_ROWS, LightGBMForecaster


def _built_features(n_days=200, start="2023-01-01", trend=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    base = 1000 + trend * np.arange(n_days) + noise * rng.standard_normal(n_days)
    raw = pd.DataFrame(
        {"commodity": "Onion", "market": "Pune", "date": dates, "modal_price": base}
    )
    daily = resample_daily(raw)
    daily = add_calendar_features(daily)
    daily = add_lag_rolling_features(daily)
    return daily


def test_fit_raises_on_too_little_data():
    features = _built_features(n_days=10)

    with pytest.raises(ValueError):
        LightGBMForecaster().fit(features)


def test_fit_predict_shape_and_dates():
    features = _built_features(n_days=200, trend=1.0, noise=0.0)

    model = LightGBMForecaster().fit(features)
    out = model.predict(horizon=7)

    assert list(out.columns) == ["date", "forecast"]
    assert len(out) == 7
    assert out["date"].iloc[0] == features["date"].iloc[-1] + pd.Timedelta(days=1)
    assert out["forecast"].notna().all()


def test_learns_flat_series_reasonably():
    # constant price -> a well-fit model's 1-step-ahead forecast should be close to it
    features = _built_features(n_days=200, trend=0.0, noise=0.0)

    model = LightGBMForecaster().fit(features)
    out = model.predict(horizon=1)

    assert out["forecast"].iloc[0] == pytest.approx(1000, rel=0.05)


def test_min_train_rows_constant_is_reasonable():
    assert 10 <= MIN_TRAIN_ROWS <= 60
