"""
Convert a Redfin CSV export into the format expected by the Lead Scorer app.

Usage:
    python convert_redfin.py redfin_download.csv
    python convert_redfin.py redfin_download.csv --rent-pct 0.009 --default-score 7
    python convert_redfin.py redfin_download.csv --output my_leads.csv

How to get the Redfin CSV:
    1. Go to redfin.com → search Tulsa, OK
    2. Apply your filters (price, beds, etc.)
    3. Scroll to bottom of results → click "Download All"
"""

import argparse
import sys
import os
import pandas as pd


# Redfin column → our column mapping
COLUMN_MAP = {
    "BEDS": "bedrooms",
    "BATHS": "bathrooms",
    "SQUARE FEET": "sqft",
    "YEAR BUILT": "year_built",
}


def clean_numeric(series: pd.Series) -> pd.Series:
    """Strip $, commas, spaces and convert to float."""
    return (
        series.astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .replace("nan", None)
        .pipe(pd.to_numeric, errors="coerce")
    )


def build_address(row: pd.Series) -> str:
    parts = [
        str(row.get("ADDRESS", "")).strip(),
        str(row.get("CITY", "")).strip(),
        str(row.get("STATE OR PROVINCE", "")).strip(),
        str(row.get("ZIP OR POSTAL CODE", "")).strip(),
    ]
    return " ".join(p for p in parts if p and p.lower() != "nan")


def convert(input_path: str, rent_pct: float, default_score: int, output_path: str) -> None:
    print(f"\nReading: {input_path}")
    df = pd.read_csv(input_path, skiprows=1)  # Redfin adds a header blurb on row 0
    df.columns = df.columns.str.strip().str.upper()

    # Try without skiprows if ADDRESS column not found
    if "ADDRESS" not in df.columns:
        df = pd.read_csv(input_path)
        df.columns = df.columns.str.strip().str.upper()

    print(f"Found {len(df)} rows. Columns: {list(df.columns)}\n")

    missing = [c for c in ["ADDRESS", "PRICE", "BEDS", "BATHS", "SQUARE FEET"] if c not in df.columns]
    if missing:
        print(f"ERROR: Required columns not found: {missing}")
        print("Make sure you're using a Redfin CSV download.")
        sys.exit(1)

    out = pd.DataFrame()

    out["address"] = df.apply(build_address, axis=1)
    out["price"] = clean_numeric(df["PRICE"])
    out["bedrooms"] = clean_numeric(df["BEDS"]).fillna(3).astype(int)
    out["bathrooms"] = clean_numeric(df["BATHS"]).fillna(1).astype(int)
    out["sqft"] = clean_numeric(df["SQUARE FEET"]).fillna(1200).astype(int)

    if "YEAR BUILT" in df.columns:
        out["year_built"] = clean_numeric(df["YEAR BUILT"]).fillna(1980).astype(int)
    else:
        out["year_built"] = 1980

    # asking_rent: use 1% rule by default (monthly rent = price × rent_pct)
    out["asking_rent"] = (out["price"] * rent_pct).round(0).astype(int)

    # neighborhood_score: default value (user can edit the CSV to refine)
    out["neighborhood_score"] = default_score

    # Drop rows with missing price or address
    out = out.dropna(subset=["price", "address"])
    out = out[out["address"].str.strip() != ""]
    out = out[out["price"] > 0]
    out = out.reset_index(drop=True)

    out.to_csv(output_path, index=False)

    print(f"Converted {len(out)} properties → {output_path}")
    print(f"  Rent estimate: {rent_pct*100:.1f}% of price per month")
    print(f"  Default neighborhood score: {default_score}/10")
    print("\nTip: Open the CSV and adjust 'asking_rent' and 'neighborhood_score'")
    print("     columns for individual properties before uploading to the app.\n")


def main():
    parser = argparse.ArgumentParser(description="Convert Redfin CSV to Lead Scorer format")
    parser.add_argument("input", help="Path to Redfin CSV file")
    parser.add_argument(
        "--rent-pct",
        type=float,
        default=0.009,
        help="Monthly rent as a fraction of price (default: 0.009 = 0.9%% rule)",
    )
    parser.add_argument(
        "--default-score",
        type=int,
        default=7,
        choices=range(1, 11),
        metavar="1-10",
        help="Default neighborhood score for all properties (default: 7)",
    )
    parser.add_argument(
        "--output",
        default="leads_from_redfin.csv",
        help="Output CSV filename (default: leads_from_redfin.csv)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    convert(args.input, args.rent_pct, args.default_score, args.output)


if __name__ == "__main__":
    main()
