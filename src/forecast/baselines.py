"""Seasonal-naive and random-walk baselines. Every other model must beat these."""

from __future__ import annotations

import pandas as pd

from src.forecast.base import Forecaster, future_dates


class RandomWalkForecaster(Forecaster):
    """Flat forecast: tomorrow's price = today's price, held for the whole horizon."""

    def fit(self, train: pd.DataFrame) -> "RandomWalkForecaster":
        clean = train.dropna(subset=["modal_price"]).sort_values("date")
        if clean.empty:
            raise ValueError("RandomWalkForecaster.fit: no non-null modal_price in train")
        self.last_date = clean["date"].iloc[-1]
        self.last_value = clean["modal_price"].iloc[-1]
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        dates = future_dates(self.last_date, horizon)
        return pd.DataFrame({"date": dates, "forecast": self.last_value})


class SeasonalNaiveForecaster(Forecaster):
    """Forecast = the price observed `season_length` days before the target date.

    Falls back to the last observed price (random-walk) when the season-ago
    date isn't available even after a small forward-fill tolerance - keeps
    the baseline usable on ~2-year series where a full prior season is thin
    near the edges, while staying honest that it's degrading to naive there.
    """

    def __init__(self, season_length: int = 365, lookup_tolerance_days: int = 3):
        self.season_length = season_length
        self.lookup_tolerance_days = lookup_tolerance_days

    def fit(self, train: pd.DataFrame) -> "SeasonalNaiveForecaster":
        clean = train.dropna(subset=["modal_price"]).sort_values("date")
        if clean.empty:
            raise ValueError("SeasonalNaiveForecaster.fit: no non-null modal_price in train")
        self.last_date = clean["date"].iloc[-1]
        self.last_value = clean["modal_price"].iloc[-1]

        dense = clean.set_index("date")["modal_price"]
        full_idx = pd.date_range(dense.index.min(), dense.index.max(), freq="D")
        self.dense_series = dense.reindex(full_idx).ffill(limit=self.lookup_tolerance_days)
        return self

    def _lookup(self, target: pd.Timestamp) -> float:
        if target in self.dense_series.index and pd.notna(self.dense_series[target]):
            return self.dense_series[target]
        return self.last_value

    def predict(self, horizon: int) -> pd.DataFrame:
        dates = future_dates(self.last_date, horizon)
        values = [self._lookup(d - pd.Timedelta(days=self.season_length)) for d in dates]
        return pd.DataFrame({"date": dates, "forecast": values})
