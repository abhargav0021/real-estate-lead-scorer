# Real Estate Lead Scorer

## Problem

Manually reviewing real estate leads is slow and inconsistent. Investors lose time reading through spreadsheets of properties, running their own GRM calculations, and making judgment calls without a structured framework.

## What It Does

This tool reads a CSV of real estate leads, scores each property with the Claude AI API, and outputs a ranked report. For every lead it computes:

- **Gross Rent Multiplier (GRM)** — price ÷ annual rent (lower is better)
- **Price per square foot**
- **AI investment score** (1–10) with a BUY / INVESTIGATE / PASS decision and a 2-sentence reasoning

Results are sorted by score and saved to `scored_leads.csv`.

## Tech Stack

| Layer | Technology |
|---|---|
| AI scoring | [Claude API](https://docs.anthropic.com) — `claude-opus-4-5` |
| Data handling | pandas |
| CLI interface | Python stdlib |
| Web UI | Streamlit |
| Config | python-dotenv |

## Project Structure

```
real-estate-lead-scorer/
├── leads.py              ← CLI script
├── app.py                ← Streamlit UI
├── sample_leads.csv      ← 10 sample Tulsa OK properties
├── .env.example          ← API key template
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

**1. Clone / download the project**

```bash
cd real-estate-lead-scorer
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your API key**

```bash
cp .env.example .env
# Edit .env and replace "your_key_here" with your actual Anthropic API key
```

Get a key at [console.anthropic.com](https://console.anthropic.com).

## Running

### CLI

```bash
python leads.py
```

Scores all rows in `sample_leads.csv`, prints a ranked table to the terminal, and saves `scored_leads.csv`.

### Streamlit UI

```bash
streamlit run app.py
```

Opens a browser UI where you can:
- Upload your own CSV (or use the included sample)
- Click **Score Leads** to analyze all properties
- View a color-coded ranked table (green = BUY, yellow = INVESTIGATE, red = PASS)
- See a score bar chart
- Download `scored_leads.csv`

## CSV Format

Your CSV must have these columns:

| Column | Description | Example |
|---|---|---|
| `address` | Full property address | `1421 S Boston Ave Tulsa OK 74119` |
| `price` | Asking price in dollars | `125000` |
| `bedrooms` | Number of bedrooms | `3` |
| `bathrooms` | Number of bathrooms | `2` |
| `sqft` | Square footage | `1450` |
| `year_built` | Year the property was built | `1962` |
| `asking_rent` | Monthly asking rent in dollars | `1150` |
| `neighborhood_score` | Neighborhood quality score 1–10 | `7` |

## Sample Output

```
================================================================================
RANKED REAL ESTATE LEADS  (highest score first)
================================================================================

#1  Score 8/10  ✅ BUY
    1089 S Delaware Ave Tulsa OK 74104
    Price $350,000  |  Rent $2,200/mo  |  GRM 13.26  |  $127.27/sqft
    Strong neighborhood score of 9/10 with a competitive GRM of 13.26 indicates
    solid cash-flow potential. The newer construction (2005) reduces near-term
    maintenance risk.

#2  Score 7/10  🔍 INVESTIGATE
    2145 E 15th St Tulsa OK 74104
    Price $175,000  |  Rent $1,400/mo  |  GRM 10.42  |  $104.17/sqft
    ...
```

> **Screenshot placeholder** — add a screenshot of the Streamlit UI here after your first run.
