import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from tools import code_tool

load_dotenv()

# ── paths ───────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

CLEAN_DATA       = ARTIFACTS_DIR / "clean_data.csv"
CONTRACT         = ARTIFACTS_DIR / "dataset_contract.json"
FEATURES_CSV     = ARTIFACTS_DIR / "features.csv"
MODEL_PKL        = ARTIFACTS_DIR / "model.pkl"
EVAL_REPORT      = ARTIFACTS_DIR / "evaluation_report.md"
MODEL_CARD       = ARTIFACTS_DIR / "model_card.md"

# ── shared LLM for all agents ───────────────────────────────────────────────
llm = LLM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
)

# ════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Feature Engineering
# ════════════════════════════════════════════════════════════════════════════
feature_agent = Agent(
    role="Feature Engineer",
    goal="Validate the cleaned dataset and engineer predictive features for ML modeling",
    backstory=(
        "You are a machine learning engineer who specializes in transforming "
        "cleaned datasets into feature-rich inputs for predictive models. "
        "You validate data contracts before touching any data."
    ),
    llm=llm,
    tools=[code_tool],
)

task_features = Task(
    description=f"""
    STEP 1 — Validate the dataset against the contract:
    - Load {CONTRACT} and read the required_columns list
    - Load {CLEAN_DATA} and confirm all required columns are present
    - Print "Contract validation passed" or raise an error with the missing columns

    STEP 2 — Engineer new features using pandas:
    - Parse Date to datetime if not already
    - Sort by Store then Date (ascending) — this is critical to prevent data leakage
    - Add: Year, Month, Week, Is_Quarter_End (1 if month in [3,6,9,12])
      For Week use EXACTLY this syntax (dt.week was removed in pandas 2.0):
          df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    - Add lag features using transform (required for correct index alignment).
      IMPORTANT: shift BEFORE rolling to avoid data leakage (do NOT include the current week in the window):
          df['Sales_Lag1'] = df.groupby('Store')['Weekly_Sales'].transform(
              lambda x: x.shift(1))
          df['Sales_Rolling4'] = df.groupby('Store')['Weekly_Sales'].transform(
              lambda x: x.shift(1).rolling(4, min_periods=1).mean())
      Sales_Rolling4 must be the 4-week trailing average of PAST weeks only (t-1, t-2, t-3, t-4).
    - Drop rows where Sales_Lag1 is NaN OR Sales_Rolling4 is NaN (first weeks per store)
    - Do NOT shuffle the data — preserve chronological order for a correct time-series split

    STEP 3 — Save to {FEATURES_CSV} (index=False)
    Print the shape and first 3 rows of the resulting dataframe.
    """,
    agent=feature_agent,
    expected_output=f"features.csv saved to {ARTIFACTS_DIR} with new engineered columns.",
)

# ════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Model Training & Evaluation
# ════════════════════════════════════════════════════════════════════════════
modeling_agent = Agent(
    role="Machine Learning Engineer",
    goal="Train and compare two predictive models for weekly sales forecasting",
    backstory=(
        "You are an ML engineer who builds, evaluates, and documents predictive "
        "models. You always compare a simple baseline against a more complex model "
        "and choose the winner based on objective metrics."
    ),
    llm=llm,
    tools=[code_tool],
)

task_modeling = Task(
    description=f"""
    Load {FEATURES_CSV} and train two regression models to predict Weekly_Sales.

    STEP 1 — Prepare data:
    Feature columns (X): Store, Year, Month, Week, Is_Quarter_End,
                         Holiday_Flag, Temperature, Fuel_Price, CPI,
                         Unemployment, Sales_Lag1, Sales_Rolling4
    Target (y): Weekly_Sales

    First, drop any rows with NaN in any feature or target column:
        feature_cols = ["Store","Year","Month","Week","Is_Quarter_End",
                        "Holiday_Flag","Temperature","Fuel_Price","CPI",
                        "Unemployment","Sales_Lag1","Sales_Rolling4"]
        df = df.dropna(subset=feature_cols + ["Weekly_Sales"])

    IMPORTANT — use a chronological split, NOT a random split.
    Feature engineering sorted by Store then Date. Before splitting, re-sort by Date ONLY
    (across all stores) so the 80/20 split is temporal, not store-based:
        df = df.sort_values("Date").reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test  = df.iloc[split_idx:]
        X_train, y_train = train[feature_cols], train["Weekly_Sales"]
        X_test,  y_test  = test[feature_cols],  test["Weekly_Sales"]
    This ensures the model trains on all stores' early weeks and tests on all stores' later weeks.

    STEP 2 — Train both models:
    Model A: LinearRegression  (from sklearn.linear_model)
    Model B: RandomForestRegressor(n_estimators=100, random_state=42)

    STEP 3 — Evaluate both on the test set:
    Calculate for each: RMSE, MAE, R²
    (use root_mean_squared_error for RMSE, mean_absolute_error for MAE, r2_score for R²
     — all from sklearn.metrics. Do NOT use mean_squared_error with squared=False, it is deprecated)

    STEP 4 — Save the better model (higher R²) to {MODEL_PKL} using joblib.dump

    STEP 5 — Write {EVAL_REPORT} as markdown.
    Build the content using a lines list + join (do NOT use triple-quoted strings or f-strings):
        lines = []
        lines.append("# Model Evaluation Report")
        lines.append("")
        lines.append("## Results")
        lines.append("| Model | RMSE | MAE | R² |")
        lines.append("|-------|------|-----|-----|")
        lines.append("| Linear Regression | " + str(round(rmse_a,2)) + " | ... | ... |")
        lines.append("| Random Forest | " + str(round(rmse_b,2)) + " | ... | ... |")
        ...
        with open(eval_path, "w") as f:
            f.write("\\n".join(lines))
    The file must contain: a results table (RMSE, MAE, R²), a Winner section, and Feature Importance (top 5 for RF).

    CRITICAL INSTRUCTION: Execute ALL steps (load, train, evaluate, save model, write report)
    in a SINGLE code block. Do NOT split into multiple separate code calls.
    Variables do not persist between code executions — everything must run together.
    """,
    agent=modeling_agent,
    context=[task_features],
    expected_output=(
        f"model.pkl saved to {ARTIFACTS_DIR}, "
        f"evaluation_report.md saved to {ARTIFACTS_DIR}."
    ),
)

# ════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Model Card
# ════════════════════════════════════════════════════════════════════════════
modelcard_agent = Agent(
    role="ML Documentation Lead",
    goal="Write a formal model card documenting the trained sales forecasting model",
    backstory=(
        "You are a responsible AI practitioner who documents ML models clearly "
        "so that business stakeholders understand what the model does, how well "
        "it performs, its limitations, and any ethical concerns."
    ),
    llm=llm,
    tools=[code_tool],
)

task_modelcard = Task(
    description=f"""
    Read {EVAL_REPORT} and {FEATURES_CSV} to gather facts, then write {MODEL_CARD}.

    The model card must include these sections:

    # Model Card — Walmart Weekly Sales Predictor

    ## Model Purpose
    What the model predicts, who would use it, and in what context.

    ## Training Data
    - Source dataset and date range
    - Number of records and stores
    - List of features used
    - How train/test split was done

    ## Performance Metrics
    A table with RMSE, MAE, R² for the winning model (from evaluation_report.md).

    ## Limitations
    At least 3 honest limitations, e.g.:
    - Time range of training data
    - Lag features cause first-week data loss per store
    - No store size or location data

    ## Ethical Considerations
    At least 2 points, e.g.:
    - Should not be used for staffing decisions without human review
    - Possible underrepresentation of certain store demographics

    ## How to Use
    A short Python snippet showing how to load model.pkl and predict for new data.
    """,
    agent=modelcard_agent,
    context=[task_modeling],
    expected_output=f"model_card.md saved to {ARTIFACTS_DIR}.",
)

# ════════════════════════════════════════════════════════════════════════════
# CREW
# ════════════════════════════════════════════════════════════════════════════
scientist_crew = Crew(
    agents=[feature_agent, modeling_agent, modelcard_agent],
    tasks=[task_features, task_modeling, task_modelcard],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    result = scientist_crew.kickoff()
    print("\n✓ Scientist Crew finished.")
    print(result)
