import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from tools import code_tool

load_dotenv()

# ── paths ───────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_PATH     = BASE_DIR / "Walmart_Sales.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ── shared LLM for all agents ───────────────────────────────────────────────
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
    tools=[code_tool],
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
    tools=[code_tool],
)

task_eda = Task(
    description=f"""
    Using pandas, matplotlib, and seaborn:
    Note: use matplotlib.use('Agg') before importing pyplot to avoid display errors.

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
    context=[task_ingest],
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
    tools=[code_tool],
)

task_contract = Task(
    description=f"""
    Load {ARTIFACTS_DIR / 'clean_data.csv'} and produce two output files.

    IMPORTANT: Write ALL code in a single code block. Do NOT use f-strings or triple-quoted strings.
    Build strings by concatenating lines with + or by using a list and join().
    Use True/False (Python booleans), NOT true/false.

    STEP 1 — Load and prepare data, then compute values:
    - Load the CSV and immediately parse the Date column:
          df = pd.read_csv(path)
          df['Date'] = pd.to_datetime(df['Date'])
    - Compute:
        top_store:    store number with highest total Weekly_Sales (groupby + sum + idxmax)
        holiday_avg:  dict mapping Holiday_Flag (0,1) to mean Weekly_Sales
        month_avg:    dict mapping month number to mean Weekly_Sales (group by df['Date'].dt.month)
        top_corr:     top 3 column names most correlated with Weekly_Sales (excluding Weekly_Sales)
                      Use: df.corr(numeric_only=True)["Weekly_Sales"]

    STEP 2 — Write {ARTIFACTS_DIR / 'insights.md'} using a list of lines joined with newline.
    Example pattern:
        lines = []
        lines.append("## Key Findings")
        lines.append("1. Top Store: " + str(top_store))
        ...
        with open(path, "w") as f:
            f.write("\\n".join(lines))

    The file must contain:
    - ## Key Findings  (5 findings using the computed values)
    - ## Correlation Summary  (top 3 features correlated with Weekly_Sales)
    - ## Business Recommendations  (3 actionable recommendations)

    STEP 3 — Write {ARTIFACTS_DIR / 'dataset_contract.json'} using json.dump.
    Build the dict in Python (use True/False, not true/false), then dump it.
    Must include: version, required_columns, columns (with type/nullable/constraints), assumptions.
    required_columns must be: ["Store", "Date", "Weekly_Sales", "Holiday_Flag",
                               "Temperature", "Fuel_Price", "CPI", "Unemployment"]

    Print "insights.md saved" and "dataset_contract.json saved" when done.
    """,
    agent=insights_agent,
    context=[task_eda],
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
