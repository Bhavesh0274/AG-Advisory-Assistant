"""Common fit/predict interface shared by every forecasting model.

Every model works on a single (commodity, market) series at a time - the
walk-forward backtest harness (Phase 5) loops over series and retrain windows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Forecaster(ABC):
    """fit() on a single series' history; predict(horizon) for the next N days."""

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "Forecaster":
        """train has columns 'date' (daily, ascending) and 'modal_price', plus
        whatever feature columns the model needs (see build_features.py)."""

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """Returns a DataFrame with columns 'date' and 'forecast', one row per
        day from the day after the last training date, for `horizon` days."""


def future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
