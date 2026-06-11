import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales AI Dashboard",
    page_icon="🛒",
    layout="wide",
)

# ── guard: pipeline must run first ───────────────────────────────────────────
required = [
    "clean_data.csv",
    "eda_report.html",
    "insights.md",
    "model.pkl",
    "evaluation_report.md",
    "model_card.md",
]
missing = [f for f in required if not (ARTIFACTS_DIR / f).exists()]
if missing:
    st.error("⚠️ Pipeline artifacts not found. Run `python flow.py` first.")
    st.code("\n".join(f"  ✗  {f}" for f in missing))
    st.stop()

# ── load data (cached) ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(ARTIFACTS_DIR / "clean_data.csv", parse_dates=["Date"])

@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS_DIR / "model.pkl")

df    = load_data()
model = load_model()

# ── header ───────────────────────────────────────────────────────────────────
st.title("🛒 Walmart Sales AI Dashboard")
st.caption("Powered by CrewAI · GPT-4o-mini · Scikit-Learn")
st.divider()

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Model Results", "🔮 Predict Sales"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Exploratory Data Analysis")

    # key metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",    f"{len(df):,}")
    col2.metric("Stores",           df["Store"].nunique())
    col3.metric("Avg Weekly Sales", f"${df['Weekly_Sales'].mean():,.0f}")
    col4.metric("Holiday Weeks",    f"{df['Holiday_Flag'].sum():,}")

    st.divider()

    # charts
    charts = [
        ("chart1_sales_distribution.png", "Sales Distribution"),
        ("chart2_sales_by_store.png",     "Average Sales by Store"),
        ("chart3_sales_over_time.png",    "Sales Over Time"),
        ("chart4_holiday_vs_regular.png", "Holiday vs Regular Weeks"),
    ]

    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(charts):
                fname, title = charts[i + j]
                path = ARTIFACTS_DIR / fname
                if path.exists():
                    col.subheader(title)
                    col.image(str(path), use_container_width=True)

    st.divider()

    # business insights
    insights_path = ARTIFACTS_DIR / "insights.md"
    if insights_path.exists():
        st.subheader("Business Insights")
        st.markdown(insights_path.read_text(encoding="utf-8"))

    # raw stats
    with st.expander("📋 Dataset Statistics"):
        st.dataframe(df.describe().round(2), use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL RESULTS
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Model Evaluation")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        eval_path = ARTIFACTS_DIR / "evaluation_report.md"
        if eval_path.exists():
            st.markdown(eval_path.read_text(encoding="utf-8"))

    with col_right:
        st.subheader("Model Info")
        model_name = type(model).__name__
        st.info(f"**Active model:** {model_name}")

        if hasattr(model, "n_estimators"):
            st.write(f"• Trees: {model.n_estimators}")
        if hasattr(model, "feature_importances_"):
            features = [
                "Store", "Year", "Month", "Week", "Is_Quarter_End",
                "Holiday_Flag", "Temperature", "Fuel_Price", "CPI",
                "Unemployment", "Sales_Lag1", "Sales_Rolling4",
            ]
            imp = pd.Series(model.feature_importances_, index=features)
            st.subheader("Top 5 Features")
            st.bar_chart(imp.nlargest(5))

    st.divider()

    # model card
    card_path = ARTIFACTS_DIR / "model_card.md"
    if card_path.exists():
        with st.expander("📄 Full Model Card"):
            st.markdown(card_path.read_text(encoding="utf-8"))

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — PREDICT SALES
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Predict Weekly Sales")
    st.write("Fill in the store details below to get a sales forecast.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            store   = st.selectbox("Store", options=sorted(df["Store"].unique()))
            month   = st.slider("Month", 1, 12, 6)
            week    = st.slider("Week", 1, 52, 26)
            year    = st.selectbox("Year", [2010, 2011, 2012, 2013])

        with col2:
            holiday       = st.checkbox("Holiday Week")
            is_quarter_end = st.checkbox("Quarter-End Week")
            temperature   = st.number_input("Temperature (°F)", 0.0, 120.0, 60.0, step=0.5)
            fuel_price    = st.number_input("Fuel Price ($/gal)", 2.0, 5.0, 3.5, step=0.01)

        with col3:
            cpi          = st.number_input("CPI", 120.0, 250.0, 211.0, step=0.1)
            unemployment = st.number_input("Unemployment (%)", 3.0, 15.0, 8.0, step=0.1)
            sales_lag1   = st.number_input(
                "Last Week's Sales ($)",
                min_value=0.0,
                value=float(df["Weekly_Sales"].mean()),
                step=1000.0,
                help="Enter the store's sales from the previous week",
            )
            sales_rolling4 = st.number_input(
                "4-Week Rolling Avg Sales ($)",
                min_value=0.0,
                value=float(df["Weekly_Sales"].mean()),
                step=1000.0,
                help="Average of the last 4 weeks of sales for this store",
            )

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "Store":          store,
            "Year":           year,
            "Month":          month,
            "Week":           week,
            "Is_Quarter_End": int(is_quarter_end),
            "Holiday_Flag":   int(holiday),
            "Temperature":    temperature,
            "Fuel_Price":     fuel_price,
            "CPI":            cpi,
            "Unemployment":   unemployment,
            "Sales_Lag1":     sales_lag1,
            "Sales_Rolling4": sales_rolling4,
        }])

        prediction = model.predict(input_data)[0]

        st.divider()
        st.metric(
            label=f"Predicted Weekly Sales — Store {store}",
            value=f"${prediction:,.0f}",
        )

        avg = df[df["Store"] == store]["Weekly_Sales"].mean()
        diff = prediction - avg
        direction = "above" if diff > 0 else "below"
        st.caption(
            f"This is ${abs(diff):,.0f} {direction} the historical average "
            f"for Store {store} (${avg:,.0f}/week)."
        )
