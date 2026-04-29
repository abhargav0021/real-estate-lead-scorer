import io
import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = "You are a real estate investment analyst. Always respond with valid JSON only — no markdown fences, no extra text."

EXPECTED_COLS = {"address", "price", "bedrooms", "bathrooms", "sqft", "year_built", "asking_rent", "neighborhood_score"}
REDFIN_COLS   = {"ADDRESS", "PRICE", "BEDS", "BATHS", "SQUARE FEET"}

# ── helpers ──────────────────────────────────────────────────────────────────


def get_client() -> Groq | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _clean_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .replace("nan", None)
        .pipe(pd.to_numeric, errors="coerce")
    )


def convert_redfin(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw Redfin export DataFrame to the app's expected schema."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()

    out = pd.DataFrame()
    out["address"] = df.apply(
        lambda r: " ".join(
            str(r.get(c, "")).strip()
            for c in ["ADDRESS", "CITY", "STATE OR PROVINCE", "ZIP OR POSTAL CODE"]
            if str(r.get(c, "")).strip().lower() not in ("", "nan")
        ),
        axis=1,
    )
    out["price"] = _clean_numeric(df["PRICE"])

    # Drop rows with no price before any int conversions
    out = out.dropna(subset=["price"])
    out = out[out["price"] > 0].reset_index(drop=True)
    df  = df.loc[out.index].reset_index(drop=True)

    out["bedrooms"]     = _clean_numeric(df["BEDS"]).fillna(3).astype(int)
    out["bathrooms"]    = _clean_numeric(df["BATHS"]).fillna(1).astype(int)
    out["sqft"]         = _clean_numeric(df["SQUARE FEET"]).fillna(1200).astype(int)
    out["year_built"]   = _clean_numeric(df.get("YEAR BUILT", pd.Series(dtype=str))).fillna(1980).astype(int)
    out["asking_rent"]  = (out["price"] * 0.009).round(0).astype(int)
    out["neighborhood_score"] = 7

    return out.reset_index(drop=True)


def load_csv(file) -> tuple[pd.DataFrame, str]:
    """Read uploaded file, auto-detect Redfin format, return (df, message)."""
    raw = pd.read_csv(file, skiprows=1)
    raw.columns = raw.columns.str.strip().str.upper()

    if not REDFIN_COLS.issubset(set(raw.columns)):
        # Try without skiprows
        file.seek(0)
        raw = pd.read_csv(file)
        raw.columns = raw.columns.str.strip().str.upper()

    if REDFIN_COLS.issubset(set(raw.columns)):
        df = convert_redfin(raw)
        # filter to only residential (skip vacant land, etc.)
        if "PROPERTY TYPE" in raw.columns:
            residential_mask = raw["PROPERTY TYPE"].str.contains(
                "Family|Townhouse|Condo|Multi", case=False, na=False
            )
            df = df[residential_mask.values[:len(df)]].reset_index(drop=True)
        return df, f"Redfin CSV detected — auto-converted {len(df)} residential properties."

    # Already in expected format
    file.seek(0)
    df = pd.read_csv(file)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        return df, f"Warning: missing columns {missing}. Scoring may fail."
    return df, f"Loaded {len(df)} leads."


def score_lead(client: Groq, row: pd.Series) -> dict:
    annual_rent = row["asking_rent"] * 12
    grm = round(row["price"] / annual_rent, 2)
    price_per_sqft = round(row["price"] / row["sqft"], 2)

    prompt = f"""Analyze this rental property investment and return ONLY a JSON object — no markdown, no prose.

Property:
- Address: {row["address"]}
- Price: ${row["price"]:,}
- Bedrooms: {row["bedrooms"]} | Bathrooms: {row["bathrooms"]}
- Square Footage: {row["sqft"]} sqft | Year Built: {row["year_built"]}
- Monthly Asking Rent: ${row["asking_rent"]:,} | Annual Rent: ${annual_rent:,}
- Gross Rent Multiplier (GRM): {grm}  (price / annual rent; lower = better)
- Price per Sqft: ${price_per_sqft}
- Neighborhood Score: {row["neighborhood_score"]}/10

Scoring rules — apply these strictly:
- score 8-10 → decision MUST be BUY   (GRM ≤ 12 AND neighborhood ≥ 6, OR GRM ≤ 10 any neighborhood)
- score 5-7  → decision MUST be INVESTIGATE  (GRM 12-15 OR neighborhood 4-5)
- score 1-4  → decision MUST be PASS  (GRM > 15 OR neighborhood ≤ 3 OR price_per_sqft > $250)

Return exactly this JSON shape:
{{
  "score": <integer 1-10>,
  "decision": "<BUY|PASS|INVESTIGATE>",
  "grm": {grm},
  "price_per_sqft": {price_per_sqft},
  "reasoning": "<2 sentences max explaining the investment score>"
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

        return json.loads(text)

    except Exception:
        return {
            "score": 5,
            "decision": "INVESTIGATE",
            "grm": grm,
            "price_per_sqft": price_per_sqft,
            "reasoning": "Scoring response could not be parsed. Manual review recommended.",
        }


def highlight_decision(row: pd.Series):
    colors = {
        "BUY": "background-color: #d4edda; color: #155724",
        "INVESTIGATE": "background-color: #fff3cd; color: #856404",
        "PASS": "background-color: #f8d7da; color: #721c24",
    }
    color = colors.get(row["Decision"], "")
    return [color] * len(row)


# ── page layout ──────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Real Estate Lead Scorer",
        page_icon=None,
        layout="wide",
    )

    st.title("Real Estate Lead Scorer")
    st.markdown("*Score and rank investment leads automatically using Groq AI (free)*")
    st.divider()

    client = get_client()
    if client is None:
        st.error(
            "**GROQ_API_KEY not found.**  \n"
            "Add your free Groq API key to `.env`, then restart Streamlit.  \n"
            "Get one free at https://console.groq.com"
        )
        st.stop()

    # ── file upload ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a CSV — Redfin export or pre-formatted leads file",
        type=["csv"],
        help="Accepts raw Redfin downloads or CSVs with columns: address, price, bedrooms, bathrooms, sqft, year_built, asking_rent, neighborhood_score",
    )

    sample_path = os.path.join(os.path.dirname(__file__), "sample_leads.csv")

    if uploaded is not None:
        df, msg = load_csv(uploaded)
        if "Warning" in msg:
            st.warning(msg)
        elif "Redfin" in msg:
            st.success(msg)
        else:
            st.success(msg)
    elif os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        st.info(f"No file uploaded — using **sample_leads.csv** ({len(df)} leads).")
    else:
        st.warning("No file uploaded and sample_leads.csv not found.")
        st.stop()

    st.subheader("Leads Preview")
    st.dataframe(df, width="stretch", hide_index=True)

    # ── scoring button ───────────────────────────────────────────────────────
    if st.button("Score Leads", type="primary", width="stretch"):
        results = []
        progress = st.progress(0.0, text="Starting…")

        with st.spinner("Scoring leads with Groq AI…"):
            for i, (_, row) in enumerate(df.iterrows()):
                pct = i / len(df)
                progress.progress(pct, text=f"Scoring {i + 1}/{len(df)}: {str(row['address'])[:40]}…")
                scored = score_lead(client, row)
                results.append(
                    {
                        "address": row["address"],
                        "price": row["price"],
                        "asking_rent": row["asking_rent"],
                        **scored,
                    }
                )

        progress.progress(1.0, text="Done!")

        results_df = (
            pd.DataFrame(results)
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        # ── summary metrics ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Results")

        buy_count = int((results_df["decision"] == "BUY").sum())
        not_buy_count = int((results_df["decision"] != "BUY").sum())
        c1, c2 = st.columns(2)
        c1.metric("BUY", buy_count)
        c2.metric("NOT BUY", not_buy_count)

        # ── color-coded table ────────────────────────────────────────────────
        display = results_df.rename(
            columns={
                "address": "Address",
                "score": "Score",
                "decision": "Decision",
                "price": "Price",
                "asking_rent": "Rent/mo",
                "grm": "GRM",
                "price_per_sqft": "$/sqft",
                "reasoning": "Reasoning",
            }
        )[["Address", "Score", "Decision", "Price", "Rent/mo", "GRM", "$/sqft", "Reasoning"]]

        display["Price"]  = display["Price"].apply(lambda x: f"${x:,.0f}")
        display["Rent/mo"] = display["Rent/mo"].apply(lambda x: f"${x:,.0f}")
        display["$/sqft"] = display["$/sqft"].apply(lambda x: f"${x:.2f}")

        st.dataframe(
            display.style.apply(highlight_decision, axis=1),
            width="stretch",
            hide_index=True,
        )

        # ── bar chart ────────────────────────────────────────────────────────
        st.subheader("Score by Property")
        chart_df = results_df.set_index("address")[["score"]]
        st.bar_chart(chart_df, y="score", width="stretch")

        # ── download button ──────────────────────────────────────────────────
        buf = io.StringIO()
        results_df.to_csv(buf, index=False)
        st.download_button(
            label="Download scored_leads.csv",
            data=buf.getvalue(),
            file_name="scored_leads.csv",
            mime="text/csv",
            width="stretch",
        )


if __name__ == "__main__":
    main()
