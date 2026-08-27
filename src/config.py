"""Project-wide configuration: scope, paths, and name-normalization maps."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_CORPUS = ROOT / "data" / "corpus"
REPORTS = ROOT / "reports"

# Default scope for the first build slice. Narrowed to Onion + Potato after
# Phase 2's coverage report: in the Kaggle historical export backing this
# slice, only these two commodities have ~2 full years of history regardless
# of market. Tomato (5mo), Wheat (8mo), and Rice (2mo) are capped short in
# that source and too thin for a seasonal-naive baseline (needs a full prior
# season) - keep them out of the working set until a better source is added,
# rather than carry commodities the backtest harness can't honestly evaluate.
DEFAULT_COMMODITIES = ["Onion", "Potato"]
DEFAULT_MARKETS = ["Lasalgaon", "Pune", "Agra", "Indore", "Latur"]

# data.gov.in Catalog API resource id for
# "Current Daily Price of Various Commodities for Various Markets (Mandi)".
AGMARKNET_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
AGMARKNET_API_BASE = f"https://api.data.gov.in/resource/{AGMARKNET_RESOURCE_ID}"

# Canonicalize common spelling/casing variants seen in raw Agmarknet dumps.
COMMODITY_ALIASES = {
    "onion": "Onion",
    "tomato": "Tomato",
    "potato": "Potato",
    "wheat": "Wheat",
    "tur": "Tur",
    "arhar": "Tur",
    "arhar (tur/red gram)(whole)": "Tur",
    "arhar dal(tur dal)": "Tur",
}

MARKET_ALIASES = {
    "lasalgaon": "Lasalgaon",
    "nasik": "Nashik",
    "nashik": "Nashik",
    "pune": "Pune",
    "agra": "Agra",
    "indore": "Indore",
    "latur": "Latur",
}


def canonical_commodity(name: str) -> str:
    if not isinstance(name, str):
        return name
    return COMMODITY_ALIASES.get(name.strip().lower(), name.strip().title())


def canonical_market(name: str) -> str:
    if not isinstance(name, str):
        return name
    return MARKET_ALIASES.get(name.strip().lower(), name.strip().title())
