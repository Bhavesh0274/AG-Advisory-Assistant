import numpy as np
import pandas as pd

from src.data.build_features import add_calendar_features, add_lag_rolling_features, resample_daily
from src.forecast.backtest import (
    BASELINE_MODEL,
    build_leaderboard,
    mase,
    walk_forward_evaluate,
    write_interpretation,
)


def test_mase_returns_nan_without_full_season_of_history():
    train_series = pd.Series(
        [100, 101, 102], index=pd.date_range("2024-01-01", periods=3, freq="D")
    )

    result = mase(np.array([105]), np.array([100]), train_series)

    assert np.isnan(result)


def test_mase_scales_by_seasonal_naive_error():
    import pytest

    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    # price alternates 100/110; since 365 is odd, every 365-day-apart pair has
    # opposite parity, so every seasonal diff is exactly +-10 -> scale = 10
    prices = [100 + (i % 2) * 10 for i in range(400)]
    train_series = pd.Series(prices, index=dates)

    result = mase(np.array([100.0]), np.array([105.0]), train_series)

    assert result == pytest.approx(5 / 10)


def test_build_leaderboard_aggregates_by_commodity_model_horizon():
    results = pd.DataFrame(
        [
            {
                "commodity": "Onion", "market": "Pune", "model": "RandomWalk",
                "split_date": pd.Timestamp("2024-01-01"), "horizon": 7,
                "mae": 10, "rmse": 12, "mape": 5, "mase": 1.0, "directional_accuracy": 0.5,
            },
            {
                "commodity": "Onion", "market": "Agra", "model": "RandomWalk",
                "split_date": pd.Timestamp("2024-01-01"), "horizon": 7,
                "mae": 20, "rmse": 22, "mape": 8, "mase": 1.4, "directional_accuracy": 0.6,
            },
        ]
    )

    board = build_leaderboard(results)

    assert len(board) == 1
    row = board.iloc[0]
    assert row["commodity"] == "Onion"
    assert row["model"] == "RandomWalk"
    assert row["horizon"] == 7
    assert row["mae"] == 15.0  # mean of 10 and 20
    assert row["n_splits"] == 2


def test_write_interpretation_reports_when_nothing_beats_baseline():
    board = pd.DataFrame(
        [
            {"commodity": "Onion", "model": BASELINE_MODEL, "horizon": 7, "mase": 1.0,
             "mae": 1, "rmse": 1, "mape": 1, "directional_accuracy": 0.5, "n_splits": 3},
            {"commodity": "Onion", "model": "RandomWalk", "horizon": 7, "mase": 1.5,
             "mae": 1, "rmse": 1, "mape": 1, "directional_accuracy": 0.5, "n_splits": 3},
        ]
    )

    text = write_interpretation(board)

    assert "nothing beats" in text


def test_write_interpretation_reports_a_winner():
    board = pd.DataFrame(
        [
            {"commodity": "Onion", "model": BASELINE_MODEL, "horizon": 7, "mase": 1.0,
             "mae": 1, "rmse": 1, "mape": 1, "directional_accuracy": 0.5, "n_splits": 3},
            {"commodity": "Onion", "model": "LightGBM", "horizon": 7, "mase": 0.7,
             "mae": 1, "rmse": 1, "mape": 1, "directional_accuracy": 0.5, "n_splits": 3},
        ]
    )

    text = write_interpretation(board)

    assert "LightGBM beats" in text


def test_walk_forward_evaluate_runs_end_to_end_on_synthetic_series():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=500, freq="D")
    prices = 1000 + 5 * np.sin(np.arange(500) / 30) + rng.standard_normal(500) * 2
    raw = pd.DataFrame(
        {"commodity": "Onion", "market": "Pune", "date": dates, "modal_price": prices}
    )
    daily = resample_daily(raw)
    daily = add_calendar_features(daily)
    daily = add_lag_rolling_features(daily)

    results = walk_forward_evaluate(daily, "Onion", "Pune")

    assert len(results) > 0
    df = pd.DataFrame(results)
    assert set(df["model"].unique()) <= {"SeasonalNaive", "RandomWalk", "LightGBM"}
    assert set(df["horizon"].unique()) <= {7, 14, 30}
    assert (df["mae"] >= 0).all()
