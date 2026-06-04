import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

# ── נתיבים ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_PATH     = BASE_DIR / "Walmart_Sales.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ── מודל משותף לכל ה-agents ─────────────────────────────────────────────────
llm = LLM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
)

# ════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Data Ingestion & Validation
# ════════════════════════════════════════════════════════════════════════════
ingestion_agent = Agent(
    role="Data Ingestion Specialist",
    goal="Load the Walmart sales dataset and validate its quality",
    backstory=(
        "You are an expert data engineer who ensures datasets are complete "
        "and trustworthy before any analysis begins. You write Python code "
        "to inspect data and report issues clearly."
    ),
    llm=llm,
    allow_code_execution=True,
    verbose=True,
)

task_ingest = Task(
    description=f"""
    Load the file at: {DATA_PATH}
    Using pandas, check and print a validation report that includes:
    1. Dataset shape (rows x columns)
    2. Column names and data types
    3. Missing values per column
    4. Min and max of Weekly_Sales — flag any negative values
    5. Date range (earliest and latest date)
    6. Number of unique stores

    Print the report clearly with section headers.
    """,
    agent=ingestion_agent,
    expected_output="A printed validation report confirming dataset quality with no critical issues found.",
)

# ════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Data Cleaning & EDA
# ════════════════════════════════════════════════════════════════════════════
eda_agent = Agent(
    role="Exploratory Data Analyst",
    goal="Clean the dataset and produce an EDA report with visualizations",
    backstory=(
        "You are a senior data analyst who transforms raw data into clean, "
        "analysis-ready datasets and produces insightful visual reports "
        "using pandas, matplotlib, and seaborn."
    ),
    llm=llm,
    allow_code_execution=True,
    verbose=True,
)

task_eda = Task(
    description=f"""
    Using pandas, matplotlib, and seaborn:

    STEP 1 — Clean the data:
    - Load {DATA_PATH}
    - Parse the Date column to datetime format
    - Sort by Store then Date
    - Drop any duplicate rows
    - Save the clean dataset to: {ARTIFACTS_DIR / 'clean_data.csv'} (index=False)

    STEP 2 — Create 4 charts and save each as a PNG inside {ARTIFACTS_DIR}:
    - chart1_sales_distribution.png : histogram of Weekly_Sales
    - chart2_sales_by_store.png     : bar chart of average Weekly_Sales per Store
    - chart3_sales_over_time.png    : line chart of total Weekly_Sales over time
    - chart4_holiday_vs_regular.png : box plot comparing Holiday_Flag 0 vs 1

    STEP 3 — Build an HTML report:
    Create {ARTIFACTS_DIR / 'eda_report.html'} that contains:
    - A title and brief dataset description
    - A statistics summary table (df.describe() rendered as HTML)
    - All 4 chart images embedded using <img src="..."> tags
    Use basic HTML (no external CSS needed).
    """,
    agent=eda_agent,
    expected_output=(
        "clean_data.csv saved, 4 PNG charts saved, and eda_report.html saved — "
        "all inside the artifacts/ folder."
    ),
)

# ════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Business Insights & Dataset Contract
# ════════════════════════════════════════════════════════════════════════════
insights_agent = Agent(
    role="Business Intelligence Analyst",
    goal="Extract business insights and define the dataset contract for downstream teams",
    backstory=(
        "You are a BI analyst who bridges raw data and business decisions. "
        "You write clear markdown summaries and formal data contracts so that "
        "the Data Scientist team knows exactly what to expect from the dataset."
    ),
    llm=llm,
    allow_code_execution=True,
    verbose=True,
)

task_contract = Task(
    description=f"""
    Load {ARTIFACTS_DIR / 'clean_data.csv'} and produce two output files:

    OUTPUT 1 — {ARTIFACTS_DIR / 'insights.md'}
    Write a markdown file with the following sections:
    - ## Key Findings  (top 5 data-driven findings, e.g. top store, holiday effect, seasonal trends)
    - ## Correlation Summary  (which features correlate most with Weekly_Sales)
    - ## Business Recommendations  (2-3 actionable recommendations)

    OUTPUT 2 — {ARTIFACTS_DIR / 'dataset_contract.json'}
    Write a JSON file that defines the dataset schema for the Data Scientist crew.
    Include for each column: type, nullable (true/false), and any constraints
    (range, allowed_values, format).
    Also include a top-level "assumptions" list and "required_columns" list.

    Example structure:
    {{
      "version": "1.0",
      "required_columns": ["Store", "Date", "Weekly_Sales", ...],
      "columns": {{
        "Store":        {{"type": "int64",   "nullable": false, "range": [1, 45]}},
        "Weekly_Sales": {{"type": "float64", "nullable": false, "min": 0}},
        "Holiday_Flag": {{"type": "int64",   "nullable": false, "allowed_values": [0, 1]}},
        "Date":         {{"type": "datetime","nullable": false, "format": "%Y-%m-%d"}}
      }},
      "assumptions": ["No missing values", "One row per store per week"]
    }}
    """,
    agent=insights_agent,
    expected_output="insights.md and dataset_contract.json saved to artifacts/.",
)

# ════════════════════════════════════════════════════════════════════════════
# CREW
# ════════════════════════════════════════════════════════════════════════════
analyst_crew = Crew(
    agents=[ingestion_agent, eda_agent, insights_agent],
    tasks=[task_ingest, task_eda, task_contract],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    result = analyst_crew.kickoff()
    print("\n✓ Analyst Crew finished.")
    print(result)
