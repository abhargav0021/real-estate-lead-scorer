import os
import sys
import json

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = "You are a real estate investment analyst. Always respond with valid JSON only — no markdown fences, no extra text."


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found. Add your free Groq key to .env (get one at console.groq.com).")
        sys.exit(1)
    return Groq(api_key=api_key)


def get_decision(grm: float, neighborhood: int, price_per_sqft: float) -> tuple[str, int]:
    if (grm <= 12 and neighborhood >= 6) or grm <= 10:
        return "BUY", 9 if grm <= 10 else 8
    elif grm > 15 or neighborhood <= 3 or price_per_sqft > 250:
        return "PASS", 3
    else:
        return "INVESTIGATE", 6


def score_lead(client: Groq, row: pd.Series) -> dict:
    annual_rent = row["asking_rent"] * 12
    grm = round(row["price"] / annual_rent, 2)
    price_per_sqft = round(row["price"] / row["sqft"], 2)
    neighborhood = int(row["neighborhood_score"])

    decision, score = get_decision(grm, neighborhood, price_per_sqft)

    prompt = f"""You are a real estate analyst. Write a 2-sentence investment reasoning for this property.
The decision has already been determined as {decision} — your reasoning must support this conclusion.

Property:
- Address: {row["address"]}
- Price: ${row["price"]:,}
- Bedrooms: {row["bedrooms"]} | Bathrooms: {row["bathrooms"]}
- Square Footage: {row["sqft"]} sqft | Year Built: {row["year_built"]}
- Monthly Asking Rent: ${row["asking_rent"]:,} | Annual Rent: ${annual_rent:,}
- Gross Rent Multiplier (GRM): {grm}
- Price per Sqft: ${price_per_sqft}
- Neighborhood Score: {neighborhood}/10

Return ONLY this JSON — no markdown, no extra text:
{{
  "score": {score},
  "decision": "{decision}",
  "grm": {grm},
  "price_per_sqft": {price_per_sqft},
  "reasoning": "<2 sentences supporting the {decision} decision>"
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

        result = json.loads(text)
        result["decision"] = decision
        result["score"] = score
        result["grm"] = grm
        result["price_per_sqft"] = price_per_sqft
        return result

    except (json.JSONDecodeError, Exception) as exc:
        print(f" [parse error: {exc}]", end="")
        return {
            "score": score,
            "decision": decision,
            "grm": grm,
            "price_per_sqft": price_per_sqft,
            "reasoning": f"GRM of {grm} with neighborhood score {neighborhood}/10 — {decision}.",
        }


def print_ranked_table(df: pd.DataFrame) -> None:
    icons = {"BUY": "✅", "INVESTIGATE": "🔍", "PASS": "❌"}
    sep = "=" * 80

    print(f"\n{sep}")
    print("RANKED REAL ESTATE LEADS  (highest score first)")
    print(sep)

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        icon = icons.get(row["decision"], "❓")
        print(f"\n#{rank}  Score {row['score']}/10  {icon} {row['decision']}")
        print(f"    {row['address']}")
        print(
            f"    Price ${row['price']:,.0f}  |  Rent ${row['asking_rent']:,.0f}/mo  "
            f"|  GRM {row['grm']}  |  ${row['price_per_sqft']}/sqft"
        )
        print(f"    {row['reasoning']}")

    print(f"\n{sep}")


def main() -> None:
    client = get_client()

    csv_path = os.path.join(os.path.dirname(__file__), "sample_leads.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} leads from {csv_path}")
    print("Scoring each lead with Groq AI...\n")

    results = []
    for idx, row in df.iterrows():
        print(f"  [{idx + 1}/{len(df)}] {row['address'][:50]}...", end=" ", flush=True)
        scored = score_lead(client, row)
        results.append(
            {
                "address": row["address"],
                "price": row["price"],
                "asking_rent": row["asking_rent"],
                **scored,
            }
        )
        print(f"→ {scored['score']}/10 {scored['decision']}")

    results_df = (
        pd.DataFrame(results)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    print_ranked_table(results_df)

    output_path = os.path.join(os.path.dirname(__file__), "scored_leads.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}\n")


if __name__ == "__main__":
    main()
