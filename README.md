# Walmart Sales AI Pipeline
### Industry-Simulated AI Product Workflow
**Using CrewAI · Python · Streamlit · GitHub**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [How to Run](#how-to-run)
7. [Crew 1 — Data Analyst Crew](#crew-1--data-analyst-crew)
8. [Crew 2 — Data Scientist Crew](#crew-2--data-scientist-crew)
9. [CrewAI Flow](#crewai-flow)
10. [Streamlit App](#streamlit-app)
11. [Output Artifacts](#output-artifacts)
12. [Tech Stack](#tech-stack)

---

## Project Overview

This project simulates how real AI product teams collaborate in industry to turn raw data into deployable insight and predictive intelligence.

A retail-tech company collects large volumes of sales data and wants to answer two questions:
- **"What has happened in the business?"** → handled by the Data Analyst Crew
- **"What is likely to happen next?"** → handled by the Data Scientist Crew

The solution is a **CrewAI Flow** that orchestrates two multi-agent teams end-to-end: ingesting raw data, producing EDA reports, training ML models, and serving predictions through a Streamlit dashboard.

---

## Architecture

```
Walmart_Sales.csv
       │
       ▼
┌─────────────────────────────┐
│   CrewAI Flow  (flow.py)    │
│                             │
│  ┌─────────────────────┐    │
│  │  Crew 1             │    │
│  │  Data Analyst Crew  │    │
│  │  • ingestion_agent  │    │
│  │  • eda_agent        │    │
│  │  • insights_agent   │    │
│  └────────┬────────────┘    │
│           │ validation      │
│  ┌────────▼────────────┐    │
│  │  Crew 2             │    │
│  │  Data Scientist Crew│    │
│  │  • feature_agent    │    │
│  │  • modeling_agent   │    │
│  │  • modelcard_agent  │    │
│  └────────┬────────────┘    │
│           │ validation      │
└───────────┼─────────────────┘
            │
            ▼
     artifacts/ folder
            │
            ▼
   ┌─────────────────┐
   │  Streamlit App  │
   │  (app.py)       │
   └─────────────────┘
```

---

## Dataset

**File:** `Walmart_Sales.csv`

| Column | Type | Description |
|--------|------|-------------|
| Store | int | Store ID (1–45) |
| Date | string | Week start date (DD-MM-YYYY) |
| Weekly_Sales | float | Total sales for that store/week |
| Holiday_Flag | int | 1 = holiday week, 0 = regular week |
| Temperature | float | Average temperature (°F) |
| Fuel_Price | float | Regional fuel price ($/gallon) |
| CPI | float | Consumer Price Index |
| Unemployment | float | Regional unemployment rate |

- **Rows:** 6,435
- **Stores:** 45
- **Date range:** February 2010 – October 2012
- **Target variable:** `Weekly_Sales`

---

## Project Structure

```
FinalProject/
│
├── Walmart_Sales.csv          # Raw dataset
│
├── crews/
│   ├── __init__.py            # Makes crews/ a Python package
│   ├── analyst_crew.py        # Crew 1: 3 data analyst agents
│   └── scientist_crew.py      # Crew 2: 3 data scientist agents
│
├── artifacts/                 # Auto-generated outputs (created at runtime)
│   ├── clean_data.csv
│   ├── eda_report.html
│   ├── insights.md
│   ├── dataset_contract.json
│   ├── features.csv
│   ├── model.pkl
│   ├── evaluation_report.md
│   └── model_card.md
│
├── flow.py                    # CrewAI Flow — orchestrates both crews
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not committed to GitHub)
└── .gitignore
```

---

## Setup & Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/) or Miniconda
- An [OpenAI API key](https://platform.openai.com/)

### Step 1 — Create the Conda environment

```bash
conda create -n FinalProject python=3.11
conda activate FinalProject
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Add your API key

Open the `.env` file and replace the placeholder:

```
OPENAI_API_KEY=your-actual-key-here
```

> **Note:** The `.env` file is listed in `.gitignore` and will never be uploaded to GitHub.

---

## How to Run

### Run the full pipeline (both crews)

```bash
conda activate FinalProject
python flow.py
```

This will:
1. Run Crew 1 (Data Analyst) → validate outputs
2. Run Crew 2 (Data Scientist) → validate outputs
3. Print a summary with total runtime

### Run only Crew 1

```bash
python -m crews.analyst_crew
```

### Run only Crew 2

```bash
python -m crews.scientist_crew
```

### Launch the Streamlit dashboard

```bash
streamlit run app.py
```

---

## Crew 1 — Data Analyst Crew

**File:** `crews/analyst_crew.py`

Three agents run sequentially:

### Agent 1 — Data Ingestion Specialist
Loads `Walmart_Sales.csv` and validates:
- Dataset shape and column types
- Missing values per column
- Min/max of `Weekly_Sales` (flags any negatives)
- Date range and number of unique stores

### Agent 2 — Exploratory Data Analyst
Cleans the dataset and produces visual reports:
- Parses `Date` to datetime, sorts by Store + Date, drops duplicates
- Generates 4 charts (distribution, sales by store, sales over time, holiday vs. regular)
- Saves **`clean_data.csv`** and **`eda_report.html`**

### Agent 3 — Business Intelligence Analyst
Extracts business insights and defines the dataset contract:
- Top-performing stores, holiday effect, seasonal trends, correlations
- Saves **`insights.md`** and **`dataset_contract.json`**

---

## Crew 2 — Data Scientist Crew

**File:** `crews/scientist_crew.py`

Reads from `artifacts/clean_data.csv` and `artifacts/dataset_contract.json`.

### Agent 1 — Feature Engineer
Validates the dataset against the contract, then engineers new features:

| New Feature | Description |
|------------|-------------|
| Year | Extracted from Date |
| Month | Extracted from Date |
| Week | ISO week number |
| Is_Quarter_End | 1 if month in [3, 6, 9, 12] |
| Sales_Lag1 | Previous week's sales (per store) |
| Sales_Rolling4 | 4-week rolling average (per store) |

Saves **`features.csv`**.

### Agent 2 — Machine Learning Engineer
Trains and compares two regression models on an 80/20 train/test split:

| Model | Role |
|-------|------|
| Linear Regression | Baseline |
| Random Forest Regressor | Main model |

Metrics reported: **RMSE**, **MAE**, **R²**

Saves the best model as **`model.pkl`** and writes **`evaluation_report.md`**.

### Agent 3 — ML Documentation Lead
Writes a formal **`model_card.md`** covering:
- Model purpose and intended use
- Training data summary
- Performance metrics
- Known limitations
- Ethical considerations
- Usage code snippet

---

## CrewAI Flow

**File:** `flow.py`

The Flow connects both crews into a single automated pipeline with validation gates:

```
@start          run_analyst_crew
@listen         validate_analyst_outputs   ← checks 4 files exist
@listen         run_scientist_crew
@listen         validate_scientist_outputs ← checks 4 files exist
@listen         pipeline_done              ← prints summary + runtime
```

**Fail gracefully:** if any required output file is missing after a crew runs, the pipeline stops immediately and prints exactly which files are missing — Crew 2 never starts on incomplete data.

**State tracking:** a `WalmartState` object records the status of each crew (`pending / completed / failed`) and accumulates any errors for inspection.

---

## Streamlit App

**File:** `app.py`

An interactive dashboard with three tabs:

### Tab 1 — EDA Dashboard
- Key statistics from `clean_data.csv`
- All 4 EDA charts generated by Crew 1
- Business insights from `insights.md`

### Tab 2 — Model Results
- Side-by-side comparison table (Linear Regression vs. Random Forest)
- Metrics: RMSE, MAE, R²
- Feature importance chart (top 5 features)

### Tab 3 — Predict Sales
- Input form: Store, Year, Month, Week, Is_Quarter_End, Holiday flag, Temperature, Fuel Price, CPI, Unemployment, Last Week's Sales, 4-Week Rolling Avg
- Loads `model.pkl` and returns a real-time sales prediction
- Displays comparison against the store's historical average

---

## Output Artifacts

All files are saved to the `artifacts/` folder:

| File | Generated by | Description |
|------|-------------|-------------|
| `clean_data.csv` | Crew 1 | Cleaned dataset with parsed dates |
| `eda_report.html` | Crew 1 | EDA report with embedded charts |
| `insights.md` | Crew 1 | Business findings and recommendations |
| `dataset_contract.json` | Crew 1 | Schema, constraints, and assumptions |
| `features.csv` | Crew 2 | Dataset with engineered features |
| `model.pkl` | Crew 2 | Trained ML model (best performer) |
| `evaluation_report.md` | Crew 2 | Model comparison table and winner |
| `model_card.md` | Crew 2 | Formal model documentation |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [CrewAI](https://crewai.com) | Multi-agent orchestration |
| [Python 3.11](https://python.org) | Core language |
| [Pandas](https://pandas.pydata.org) | Data manipulation |
| [Scikit-Learn](https://scikit-learn.org) | ML models and metrics |
| [Matplotlib / Seaborn](https://seaborn.pydata.org) | Visualizations |
| [Streamlit](https://streamlit.io) | Interactive dashboard |
| [Joblib](https://joblib.readthedocs.io) | Model serialization |
| [Git + GitHub](https://github.com) | Version control |
