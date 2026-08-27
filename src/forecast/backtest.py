"""Walk-forward / expanding-window backtest harness. The point of this
project isn't any one model - it's this harness honestly measuring whether
a model earns its complexity over seasonal-naive. Never a random split:
that leaks the future on a time series.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.config import DATA_PROCESSED, REPORTS  # noqa: E402
from src.forecast.baselines import RandomWalkForecaster, SeasonalNaiveForecaster  # noqa: E402
from src.forecast.lightgbm_model import LightGBMForecaster  # noqa: E402

HORIZONS = (7, 14, 30)
INITIAL_TRAIN_DAYS = 365
STEP_DAYS = 30
SEASON_LENGTH = 365
BASELINE_MODEL = "SeasonalNaive"

MODEL_FACTORIES = {
    "SeasonalNaive": lambda: SeasonalNaiveForecaster(season_length=SEASON_LENGTH),
    "RandomWalk": lambda: RandomWalkForecaster(),
    "LightGBM": lambda: LightGBMForecaster(),
}


def mase(actual: np.ndarray, forecast: np.ndarray, train_series: pd.Series) -> float:
    """MAE scaled by the in-sample seasonal-naive error - the honest baseline."""
    naive_errors = train_series.diff(SEASON_LENGTH).abs().dropna()
    if naive_errors.empty:
        return np.nan
    scale = naive_errors.mean()
    if not scale or np.isnan(scale):
        return np.nan
    return float(np.mean(np.abs(actual - forecast)) / scale)


def _split_metrics(actual: np.ndarray, forecast: np.ndarray, train_series: pd.Series,
                    last_train_value: float) -> dict:
    mae = float(np.mean(np.abs(actual - forecast)))
    rmse = float(np.sqrt(np.mean((actual - forecast) ** 2)))
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs((actual[nonzero] - forecast[nonzero]) / actual[nonzero])) * 100)
        if nonzero.any()
        else np.nan
    )
    actual_dir = np.sign(actual - last_train_value)
    forecast_dir = np.sign(forecast - last_train_value)
    directional_accuracy = float(np.mean(actual_dir == forecast_dir))
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mase": mase(actual, forecast, train_series),
        "directional_accuracy": directional_accuracy,
    }


def walk_forward_evaluate(series: pd.DataFrame, commodity: str, market: str) -> list[dict]:
    """series: one commodity x market slice of features.parquet, sorted by date."""
    series = series.sort_values("date").reset_index(drop=True)
    n = len(series)
    max_h = max(HORIZONS)
    results = []

    train_end = INITIAL_TRAIN_DAYS
    while train_end + max_h <= n:
        train = series.iloc[:train_end]
        test = series.iloc[train_end:train_end + max_h][["date", "modal_price"]]
        train_price_series = train.set_index("date")["modal_price"]
        last_train_value = train["modal_price"].dropna().iloc[-1]
        split_date = train["date"].iloc[-1]

        for model_name, factory in MODEL_FACTORIES.items():
            model = factory()
            try:
                model.fit(train)
            except ValueError:
                continue

            forecast = model.predict(max_h)
            merged = forecast.merge(test, on="date", how="inner").dropna()
            if merged.empty:
                continue

            for h in HORIZONS:
                sub = merged.iloc[:h]
                if len(sub) < h:
                    continue
                metrics = _split_metrics(
                    sub["modal_price"].to_numpy(),
                    sub["forecast"].to_numpy(),
                    train_price_series,
                    last_train_value,
                )
                results.append(
                    {
                        "commodity": commodity,
                        "market": market,
                        "model": model_name,
                        "split_date": split_date,
                        "horizon": h,
                        **metrics,
                    }
                )

        train_end += STEP_DAYS

    return results


def build_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["commodity", "model", "horizon"])
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mape=("mape", "mean"),
            mase=("mase", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
            n_splits=("mae", "count"),
        )
        .round(3)
        .reset_index()
        .sort_values(["commodity", "horizon", "mase"])
    )


def write_interpretation(leaderboard: pd.DataFrame) -> str:
    lines = ["## Interpretation", "", "### Point-error (MASE vs " + BASELINE_MODEL + ")", ""]
    for (commodity, horizon), g in leaderboard.groupby(["commodity", "horizon"]):
        g = g.set_index("model")
        if BASELINE_MODEL not in g.index:
            continue
        baseline_mase = g.loc[BASELINE_MODEL, "mase"]
        beaters = g[(g["mase"] < baseline_mase) & (g.index != BASELINE_MODEL)]
        beaters = beaters.dropna(subset=["mase"])
        if pd.isna(baseline_mase):
            lines.append(
                f"- **{commodity}, {horizon}d**: MASE undefined for {BASELINE_MODEL} "
                "(not enough training history yet for a full prior season) - skipped."
            )
        elif beaters.empty:
            lines.append(
                f"- **{commodity}, {horizon}d**: nothing beats {BASELINE_MODEL} "
                f"(MASE={baseline_mase:.2f}). The honest result on this noisy a series."
            )
        else:
            best = beaters["mase"].idxmin()
            lines.append(
                f"- **{commodity}, {horizon}d**: {best} beats {BASELINE_MODEL} "
                f"(MASE {beaters.loc[best, 'mase']:.2f} vs {baseline_mase:.2f})."
            )

    lines += ["", "### Directional accuracy caveat", ""]
    rw = leaderboard[leaderboard["model"] == "RandomWalk"]
    other_best = (
        leaderboard[leaderboard["model"] != "RandomWalk"]
        .groupby(["commodity", "horizon"])["directional_accuracy"]
        .max()
    )
    if not rw.empty:
        rw_avg_dir = rw["directional_accuracy"].mean()
        other_avg_dir = other_best.mean() if not other_best.empty else float("nan")
        lines.append(
            f"RandomWalk's forecast is flat (no explicit up/down call), so its directional "
            f"accuracy is structurally near-zero by construction (avg "
            f"{rw_avg_dir:.2f} here) - low MAE does not mean it is useful for a hold/sell "
            f"call. The best non-RandomWalk model per commodity/horizon averages "
            f"{other_avg_dir:.2f} directional accuracy instead. For this project's actual "
            "decision (sell now or hold), directional accuracy is the more relevant metric "
            "than raw MAE, per the guide's own framing - low MAE from a model that never "
            "calls a direction is not a usable signal on its own."
        )
    return "\n".join(lines)


def main() -> None:
    features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    features["date"] = pd.to_datetime(features["date"])

    all_results = []
    for (commodity, market), g in features.groupby(["commodity", "market"]):
        print(f"Backtesting {commodity} x {market} ({len(g)} days) ...")
        all_results.extend(walk_forward_evaluate(g, commodity, market))

    results = pd.DataFrame(all_results)
    results_path = DATA_PROCESSED / "backtest_results.parquet"
    results.to_parquet(results_path, index=False)
    print(f"\nSaved {len(results)} split-level results to {results_path}")

    leaderboard = build_leaderboard(results)

    report_lines = [
        "# Forecast Evaluation",
        "",
        "Walk-forward / expanding-window backtest. Never a random split - each split "
        "trains only on the past and evaluates on unseen future days. MASE is scaled "
        f"against {BASELINE_MODEL} (the honest baseline); values below 1.0 beat it.",
        "",
        f"Initial train window: {INITIAL_TRAIN_DAYS}d, step: {STEP_DAYS}d, "
        f"horizons: {list(HORIZONS)}d, season length: {SEASON_LENGTH}d.",
        "",
        "## Leaderboard (mean across walk-forward splits, all markets per commodity)",
        "",
        leaderboard.to_markdown(index=False),
        "",
        write_interpretation(leaderboard),
        "",
    ]
    report_path = REPORTS / "forecast_eval.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
