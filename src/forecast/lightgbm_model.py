"""LightGBM forecaster on lag/calendar features, stepped recursively for multi-day horizons.

Recursive (not direct-per-horizon) strategy: one model predicts 1 day ahead;
multi-day forecasts feed each prediction back in as the next step's lag
features. Simpler than training a separate model per horizon, at the cost of
accumulating error over the horizon - that tradeoff should show up honestly
in the Phase 5 backtest (directional accuracy/MASE degrading with horizon).

cross_mandi_spread/is_outlier/is_imputed/is_observed from build_features.py
are deliberately excluded: they either need peer-market data unavailable at
future dates, or describe data quality rather than predictive signal.
"""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

from src.data.build_features import FESTIVAL_DATES
from src.forecast.base import Forecaster, future_dates

FEATURE_COLS = [
    "month",
    "day_of_week",
    "quarter",
    "is_festival_window",
    "price_lag_1",
    "price_lag_7",
    "price_lag_14",
    "price_lag_30",
    "price_roll_mean_7",
    "price_roll_std_7",
    "price_roll_mean_14",
    "price_roll_std_14",
    "price_roll_mean_30",
    "price_roll_std_30",
]

MIN_TRAIN_ROWS = 30


def _is_festival_window(date: pd.Timestamp) -> bool:
    return bool((pd.Series(FESTIVAL_DATES - date).abs() <= pd.Timedelta(days=2)).any())


def _features_at(history: pd.Series, target_date: pd.Timestamp) -> dict:
    return {
        "month": target_date.month,
        "day_of_week": target_date.dayofweek,
        "quarter": target_date.quarter,
        "is_festival_window": _is_festival_window(target_date),
        "price_lag_1": history.iloc[-1],
        "price_lag_7": history.iloc[-7] if len(history) >= 7 else history.iloc[0],
        "price_lag_14": history.iloc[-14] if len(history) >= 14 else history.iloc[0],
        "price_lag_30": history.iloc[-30] if len(history) >= 30 else history.iloc[0],
        "price_roll_mean_7": history.iloc[-7:].mean(),
        "price_roll_std_7": history.iloc[-7:].std(),
        "price_roll_mean_14": history.iloc[-14:].mean(),
        "price_roll_std_14": history.iloc[-14:].std(),
        "price_roll_mean_30": history.iloc[-30:].mean(),
        "price_roll_std_30": history.iloc[-30:].std(),
    }


class LightGBMForecaster(Forecaster):
    def __init__(self, **lgbm_kwargs):
        params = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, verbosity=-1)
        params.update(lgbm_kwargs)
        self.model = LGBMRegressor(**params)

    def fit(self, train: pd.DataFrame) -> "LightGBMForecaster":
        clean = train.sort_values("date")
        X = clean[FEATURE_COLS]
        y = clean["modal_price"]
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]
        if len(X) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"LightGBMForecaster.fit: only {len(X)} usable rows, need >= {MIN_TRAIN_ROWS}"
            )

        self.model.fit(X, y)
        self.last_date = clean["date"].iloc[-1]
        self.history = (
            clean.set_index("date")["modal_price"].sort_index().ffill().dropna()
        )
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        history = self.history.copy()
        dates = future_dates(self.last_date, horizon)
        preds = []
        for target_date in dates:
            feats = _features_at(history, target_date)
            x_row = pd.DataFrame([feats])[FEATURE_COLS]
            pred = float(self.model.predict(x_row)[0])
            preds.append(pred)
            history.loc[target_date] = pred

        return pd.DataFrame({"date": dates, "forecast": preds})
