"""Fetch Agmarknet daily mandi price data and report coverage.

Primary source: data.gov.in Catalog API (requires DATA_GOV_IN_API_KEY).
Fallback: CEDA's cleaned Agmarknet export, manually downloaded into
data/raw/ceda/ (https://agmarknet.ceda.ashoka.edu.in/ has no public REST API,
so this path expects CSV/Parquet files the user has exported from there).

Note: the data.gov.in price dataset does not include arrivals volume; the
`arrivals` column is left NaN here and is a Phase-2 follow-up (Agmarknet
portal's separate price+arrival report, or a CEDA export that includes it).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.config import (  # noqa: E402
    AGMARKNET_API_BASE,
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_COMMODITIES,
    DEFAULT_MARKETS,
    canonical_commodity,
    canonical_market,
)

RAW_COLUMNS = {
    "state": "state",
    "district": "district",
    "market": "market",
    "commodity": "commodity",
    "variety": "variety",
    "grade": "grade",
    "arrival_date": "date",
    "min_price": "min_price",
    "max_price": "max_price",
    "modal_price": "modal_price",
}

PAGE_SIZE = 1000
MAX_RETRIES = 3
# data.gov.in's WAF silently hangs (no response, no error) on the default
# python-requests User-Agent; a browser-like UA is required.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def load_dotenv(path: Path = ROOT_DIR / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def fetch_commodity_api(commodity: str, api_key: str, page_size: int = PAGE_SIZE) -> pd.DataFrame:
    """Pull all pages for one commodity from the data.gov.in Catalog API."""
    records: list[dict] = []
    offset = 0
    while True:
        params = {
            "api-key": api_key,
            "format": "json",
            "limit": page_size,
            "offset": offset,
            "filters[commodity]": commodity,
        }
        resp = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(
                    AGMARKNET_API_BASE, params=params, headers=REQUEST_HEADERS, timeout=30
                )
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                print(f"  retry {attempt + 1} for {commodity} offset={offset}: {exc}")
                time.sleep(2 ** attempt)

        payload = resp.json()
        batch = payload.get("records", [])
        if not batch:
            break
        records.extend(batch)
        offset += page_size
        total = int(payload.get("total", 0) or 0)
        if total and offset >= total:
            break

    return pd.DataFrame.from_records(records)


def normalize(df: pd.DataFrame, date_format: str = "%d/%m/%Y") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(RAW_COLUMNS.values()) + ["arrivals"])

    df = df.rename(columns={k: v for k, v in RAW_COLUMNS.items() if k in df.columns})
    keep = [c for c in RAW_COLUMNS.values() if c in df.columns]
    df = df[keep].copy()

    df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")
    for col in ("min_price", "max_price", "modal_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["commodity"] = df["commodity"].map(canonical_commodity)
    df["market"] = df["market"].map(canonical_market)
    df["arrivals"] = pd.NA

    df = df.dropna(subset=["date", "modal_price"])
    df = df.drop_duplicates(subset=["date", "market", "commodity", "variety"])
    return df.sort_values(["commodity", "market", "date"]).reset_index(drop=True)


def load_ceda_fallback(ceda_dir: Path) -> pd.DataFrame:
    files = list(ceda_dir.glob("*.csv")) + list(ceda_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        raw = pd.read_csv(f) if f.suffix == ".csv" else pd.read_parquet(f)
        raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
        frames.append(raw)
    combined = pd.concat(frames, ignore_index=True)
    return normalize(combined)


# Kaggle "Indian Agricultural Mandi Prices" export uses different column
# names/casing than Agmarknet, and a US-style M/D/Y date, so it gets its own
# rename map instead of reusing RAW_COLUMNS.
KAGGLE_COLUMN_RENAMES = {
    "district_name": "district",
    "market_name": "market",
    "price_date": "arrival_date",
}


def load_kaggle_fallback(kaggle_dir: Path) -> pd.DataFrame:
    files = list(kaggle_dir.glob("*.csv")) + list(kaggle_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        raw = pd.read_csv(f) if f.suffix == ".csv" else pd.read_parquet(f)
        raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
        raw = raw.rename(columns=KAGGLE_COLUMN_RENAMES)
        frames.append(raw)
    combined = pd.concat(frames, ignore_index=True)
    return normalize(combined, date_format="%m/%d/%Y")


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (commodity, market), g in df.groupby(["commodity", "market"]):
        start, end = g["date"].min(), g["date"].max()
        full_range = pd.date_range(start, end, freq="D")
        pct_missing = 100 * (1 - g["date"].nunique() / max(len(full_range), 1))
        rows.append(
            {
                "commodity": commodity,
                "market": market,
                "start": start.date(),
                "end": end.date(),
                "n_days_observed": g["date"].nunique(),
                "pct_missing": round(pct_missing, 1),
            }
        )
    return pd.DataFrame(rows).sort_values(["commodity", "market"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commodities", default=",".join(DEFAULT_COMMODITIES))
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    parser.add_argument("--out", default=str(DATA_PROCESSED / "prices.parquet"))
    args = parser.parse_args()

    commodities = [c.strip() for c in args.commodities.split(",") if c.strip()]
    markets = {canonical_market(m.strip()) for m in args.markets.split(",") if m.strip()}

    load_dotenv()
    api_key = os.environ.get("DATA_GOV_IN_API_KEY")
    frames = []

    if api_key:
        DATA_RAW.joinpath("agmarknet").mkdir(parents=True, exist_ok=True)
        for commodity in commodities:
            print(f"Fetching {commodity} from data.gov.in ...")
            raw = fetch_commodity_api(commodity, api_key)
            raw.to_csv(DATA_RAW / "agmarknet" / f"{commodity.lower()}_raw.csv", index=False)
            frames.append(normalize(raw))
    else:
        print(
            "DATA_GOV_IN_API_KEY not set - skipping the live API pull.\n"
            "Get a free key at https://data.gov.in (Sign In -> My Account -> API keys),\n"
            "then: export DATA_GOV_IN_API_KEY=your_key_here\n\n"
            "Falling back to any CEDA exports found in data/raw/ceda/ "
            "(download manually from https://agmarknet.ceda.ashoka.edu.in/)."
        )

    ceda_dir = DATA_RAW / "ceda"
    ceda_df = load_ceda_fallback(ceda_dir)
    if not ceda_df.empty:
        print(f"Loaded {len(ceda_df)} rows from CEDA fallback in {ceda_dir}")
        frames.append(ceda_df)

    kaggle_dir = DATA_RAW / "kaggle"
    kaggle_df = load_kaggle_fallback(kaggle_dir)
    if not kaggle_df.empty:
        print(f"Loaded {len(kaggle_df)} rows from Kaggle export in {kaggle_dir}")
        frames.append(kaggle_df)

    if not frames:
        print(
            "No data fetched from any source. Set DATA_GOV_IN_API_KEY, place CEDA "
            f"exports in {ceda_dir}, or place a Kaggle export in {kaggle_dir}, then re-run."
        )
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["market"].isin(markets)] if markets else combined
    target_commodities = {canonical_commodity(c) for c in commodities}
    combined = combined[combined["commodity"].isin(target_commodities)]
    combined = combined.drop_duplicates(subset=["date", "market", "commodity", "variety"])
    combined = combined.sort_values(["commodity", "market", "date"]).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved {len(combined)} rows to {out_path}")

    if not combined.empty:
        report = coverage_report(combined)
        report_path = DATA_PROCESSED / "coverage_report.csv"
        report.to_csv(report_path, index=False)
        print(f"\nCoverage report ({report_path}):")
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
