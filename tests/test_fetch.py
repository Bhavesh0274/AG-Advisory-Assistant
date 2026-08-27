import pandas as pd

from src.data.fetch import KAGGLE_COLUMN_RENAMES, coverage_report, normalize


def _raw_row(**overrides):
    row = {
        "state": "Maharashtra",
        "district": "Nashik",
        "market": "lasalgaon",
        "commodity": "onion",
        "variety": "Local",
        "grade": "FAQ",
        "arrival_date": "01/01/2024",
        "min_price": "1000",
        "max_price": "1500",
        "modal_price": "1250",
    }
    row.update(overrides)
    return row


def test_normalize_canonicalizes_names_and_types():
    df = pd.DataFrame([_raw_row(), _raw_row(arrival_date="02/01/2024", modal_price="1300")])

    out = normalize(df)

    assert list(out["commodity"].unique()) == ["Onion"]
    assert list(out["market"].unique()) == ["Lasalgaon"]
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out["modal_price"].tolist() == [1250.0, 1300.0]
    assert out["arrivals"].isna().all()


def test_normalize_drops_rows_missing_price_or_date():
    df = pd.DataFrame(
        [_raw_row(), _raw_row(modal_price=""), _raw_row(arrival_date="")]
    )

    out = normalize(df)

    assert len(out) == 1


def test_normalize_deduplicates():
    df = pd.DataFrame([_raw_row(), _raw_row()])

    out = normalize(df)

    assert len(out) == 1


def test_normalize_parses_kaggle_style_us_dates():
    raw = pd.DataFrame(
        [
            {
                "state": "Maharashtra",
                "district": "nashik",
                "market": "Lasalgaon",
                "commodity": "Onion",
                "variety": "Local",
                "grade": "FAQ",
                "arrival_date": "6/13/2023",
                "min_price": "1000",
                "max_price": "1500",
                "modal_price": "1250",
            }
        ]
    )

    out = normalize(raw, date_format="%m/%d/%Y")

    assert out["date"].iloc[0] == pd.Timestamp("2023-06-13")


def test_kaggle_column_renames_map_to_agmarknet_schema():
    assert KAGGLE_COLUMN_RENAMES["district_name"] == "district"
    assert KAGGLE_COLUMN_RENAMES["market_name"] == "market"
    assert KAGGLE_COLUMN_RENAMES["price_date"] == "arrival_date"


def test_coverage_report_flags_missing_days():
    df = pd.DataFrame(
        {
            "commodity": ["Onion", "Onion"],
            "market": ["Lasalgaon", "Lasalgaon"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-04"]),
        }
    )

    report = coverage_report(df)

    assert len(report) == 1
    row = report.iloc[0]
    assert row["n_days_observed"] == 2
    assert row["pct_missing"] == 50.0
