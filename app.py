import calendar

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

matplotlib.use("Agg")

BASE_DIR      = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

st.set_page_config(
    page_title="Walmart Sales AI Dashboard",
    page_icon="🛒",
    layout="wide",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
_BG     = "#0A0F1E"
_CARD   = "#111827"
_BORDER = "#1E2D45"
_MUTED  = "#94A3B8"
_DIM    = "#64748B"
_TEXT   = "#F1F5F9"
_BLUE   = "#2563EB"
_GREEN  = "#10B981"
_PURPLE = "#8B5CF6"
_AMBER  = "#F59E0B"
_RED    = "#EF4444"

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, .stMain,
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #0A0F1E !important;
    font-family: 'Inter', sans-serif !important;
    color: #F1F5F9 !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0F1E; }
::-webkit-scrollbar-thumb { background: #1E2D45; border-radius: 4px; }

[data-testid="stTabs"] > div:first-child { border-bottom: 1px solid #1E2D45 !important; gap: 0 !important; }
button[data-baseweb="tab"] {
    background: transparent !important; color: #94A3B8 !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.875rem !important;
    font-weight: 500 !important; padding: 10px 20px !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    border-radius: 0 !important; transition: color 0.2s ease, border-color 0.2s ease !important;
}
button[data-baseweb="tab"]:hover { color: #F1F5F9 !important; background: rgba(37,99,235,0.06) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563EB !important; border-bottom: 2px solid #2563EB !important; background: transparent !important;
}
[data-testid="stTabsContent"] { background: transparent !important; padding-top: 1.5rem !important; }

[data-testid="metric-container"] {
    background: #111827 !important; border: 1px solid #1E2D45 !important;
    border-left: 4px solid #2563EB !important; border-radius: 10px !important; padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 0.8rem !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
[data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 1.75rem !important; font-weight: 700 !important; font-variant-numeric: tabular-nums !important; }

[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important; color: #ffffff !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important; padding: 12px 28px !important;
    cursor: pointer !important; transition: box-shadow 0.2s ease !important;
}
[data-testid="stFormSubmitButton"] > button:hover { box-shadow: 0 0 20px rgba(37,99,235,0.45) !important; }
.stButton > button {
    background: #111827 !important; color: #F1F5F9 !important;
    border: 1px solid #1E2D45 !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div, [data-baseweb="input"] input,
[data-baseweb="select"] > div {
    background: #1A2235 !important; border: 1px solid #1E2D45 !important;
    border-radius: 8px !important; color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.875rem !important;
}
[data-testid="stExpander"] { background: #111827 !important; border: 1px solid #1E2D45 !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { color: #94A3B8 !important; font-size: 0.875rem !important; font-weight: 500 !important; padding: 12px 16px !important; }
[data-testid="stAlert"] { background: #1A2235 !important; border: 1px solid #1E2D45 !important; border-radius: 10px !important; color: #F1F5F9 !important; }
hr { border-color: #1E2D45 !important; margin: 1.5rem 0 !important; }
[data-testid="stCaptionContainer"], .stCaption, small { color: #64748B !important; font-size: 0.8rem !important; }
[data-baseweb="popover"] [data-baseweb="menu"], ul[data-testid="stSelectboxVirtualDropdown"] {
    background: #1A2235 !important; border: 1px solid #1E2D45 !important; border-radius: 8px !important;
}
li[data-baseweb="menu-item"] { background: transparent !important; color: #F1F5F9 !important; }
li[data-baseweb="menu-item"]:hover { background: rgba(37,99,235,0.12) !important; }
[data-testid="stForm"] { background: #111827 !important; border: 1px solid #1E2D45 !important; border-radius: 12px !important; padding: 24px !important; }
h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif !important; color: #F1F5F9 !important; }
p, li, span { color: #F1F5F9 !important; }
[data-testid="stTable"] table { width: 100%; border-collapse: collapse; }
[data-testid="stTable"] th {
    background: #1A2235 !important; color: #F1F5F9 !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    padding: 8px 12px !important; border-bottom: 1px solid #1E2D45 !important; text-align: right !important;
}
[data-testid="stTable"] td {
    color: #CBD5E1 !important; font-size: 0.82rem !important;
    padding: 7px 12px !important; border-bottom: 1px solid #1E2D45 !important; text-align: right !important;
}
[data-testid="stTable"] tr:first-child td { color: #94A3B8 !important; font-weight: 600 !important; }
[data-testid="stTable"] tbody tr:hover td { background: rgba(37,99,235,0.05) !important; }

/* ── code blocks ── */
pre, code, .stCodeBlock, [data-testid="stCode"] pre,
[data-testid="stMarkdownContainer"] pre,
[data-testid="stMarkdownContainer"] code {
    background: #0D1117 !important;
    color: #93C5FD !important;
    border: 1px solid #1E2D45 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ── guard ─────────────────────────────────────────────────────────────────────
required = ["clean_data.csv", "eda_report.html", "insights.md", "model.pkl",
            "evaluation_report.md", "model_card.md"]
missing = [f for f in required if not (ARTIFACTS_DIR / f).exists()]
if missing:
    st.error("Pipeline artifacts not found. Run `python flow.py` first.")
    st.code("\n".join(f"  ✗  {f}" for f in missing))
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(ARTIFACTS_DIR / "clean_data.csv", parse_dates=["Date"])

@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS_DIR / "model.pkl")

df    = load_data()
model = load_model()

# ── Pre-compute stats ─────────────────────────────────────────────────────────
total_sales      = df["Weekly_Sales"].sum()
avg_weekly       = df["Weekly_Sales"].mean()
top_store_id     = int(df.groupby("Store")["Weekly_Sales"].sum().idxmax())
top_store_total  = df.groupby("Store")["Weekly_Sales"].sum().max()
holiday_avg      = df[df["Holiday_Flag"] == 1]["Weekly_Sales"].mean()
regular_avg      = df[df["Holiday_Flag"] == 0]["Weekly_Sales"].mean()
holiday_lift_pct = (holiday_avg / regular_avg - 1) * 100
monthly_avg      = df.groupby(df["Date"].dt.month)["Weekly_Sales"].mean()
peak_month_num   = int(monthly_avg.idxmax())
low_month_num    = int(monthly_avg.idxmin())
peak_month_name  = calendar.month_name[peak_month_num]
low_month_name   = calendar.month_name[low_month_num]
date_range_str   = f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}"

# ── Parse evaluation report ───────────────────────────────────────────────────
eval_text = (ARTIFACTS_DIR / "evaluation_report.md").read_text(encoding="utf-8")

winner_model = None
for line in eval_text.split("\n"):
    if "better model is:" in line or "Winner:" in line:
        winner_model = line.split(":")[-1].strip().rstrip(".")
        break

rmse_lr = mae_lr = r2_lr = None
rmse_rf = mae_rf = r2_rf = None
for line in eval_text.split("\n"):
    if "Linear Regression" in line and "|" in line:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 4:
            try:
                rmse_lr = float(parts[1]); mae_lr = float(parts[2]); r2_lr = float(parts[3])
            except ValueError:
                pass
    if "Random Forest" in line and "|" in line:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 4:
            try:
                rmse_rf = float(parts[1]); mae_rf = float(parts[2]); r2_rf = float(parts[3])
            except ValueError:
                pass

# Winner's metrics
if winner_model and "Linear" in winner_model:
    rmse_w, mae_w, r2_w = rmse_lr, mae_lr, r2_lr
else:
    rmse_w, mae_w, r2_w = rmse_rf, mae_rf, r2_rf

# ── Plotly dark-theme helper ──────────────────────────────────────────────────
def _dark(fig, height=320):
    fig.update_layout(
        paper_bgcolor=_CARD, plot_bgcolor=_CARD,
        font=dict(color=_MUTED, family="Inter", size=11),
        height=height,
        margin=dict(l=12, r=12, t=28, b=12),
        xaxis=dict(gridcolor=_BORDER, zerolinecolor=_BORDER, tickfont=dict(color=_MUTED)),
        yaxis=dict(gridcolor=_BORDER, zerolinecolor=_BORDER, tickfont=dict(color=_MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_MUTED)),
        title=dict(font=dict(color=_MUTED, size=12)),
    )
    return fig

def _chart_card(fig, caption_text):
    st.markdown(
        f"<div style='background:{_CARD};border:1px solid {_BORDER};border-radius:12px;padding:4px 8px 0 8px;'>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(caption_text)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
model_display = winner_model or type(model).__name__
st.markdown(
    "<div style='display:flex;align-items:center;justify-content:space-between;"
    "padding:24px 0 18px 0;border-bottom:1px solid #1E2D45;margin-bottom:24px;'>"
    "<div style='display:flex;align-items:center;gap:14px;'>"
    "<div style='width:44px;height:44px;background:linear-gradient(135deg,#2563EB,#8B5CF6);"
    "border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:22px;'>&#x1F6D2;</div>"
    "<div>"
    "<h1 style='margin:0;padding:0;font-size:1.6rem;font-weight:700;line-height:1.2;"
    "color:#F1F5F9 !important;letter-spacing:-0.02em;'>Walmart Sales "
    "<span style=\"background:linear-gradient(90deg,#2563EB,#8B5CF6);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;\">AI</span> Dashboard</h1>"
    f"<p style='margin:2px 0 0 0;font-size:0.8rem;color:#64748B;'>"
    f"Data Analysis & Sales Forecasting · {date_range_str} · 45 Stores</p>"
    "</div></div>"
    "<div style='display:flex;flex-direction:column;align-items:flex-end;gap:6px;'>"
    f"<div style='background:rgba(37,99,235,0.1);border:1px solid rgba(37,99,235,0.3);"
    f"border-radius:20px;padding:5px 14px;font-size:0.75rem;color:#93BBFC;font-weight:500;'>"
    f"CrewAI · GPT-4o-mini · {model_display}</div>"
    "<div style='display:flex;gap:6px;align-items:center;'>"
    "<span style='width:7px;height:7px;border-radius:50%;background:#10B981;display:inline-block;"
    "box-shadow:0 0 6px #10B981;'></span>"
    "<span style='font-size:0.72rem;color:#10B981;font-weight:500;'>Model Active</span>"
    "</div></div></div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["  📊 Data Analysis  ", "  🤖 Model Results  ", "  🔮 Sales Forecast  "])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("### 📊 Data Overview")
    st.caption(f"Dataset covers {len(df):,} weekly observations across 45 Walmart stores from {date_range_str}.")

    # ── KPI cards ──────────────────────────────────────────────────────────────
    kpi_cols = st.columns(4)
    kpi_data = [
        ("Total Records",    f"{len(df):,}",        f"{df['Store'].nunique()} stores × ~143 weeks",                          _BLUE),
        ("Avg Weekly Sales", f"${avg_weekly:,.0f}", f"Range: ${df['Weekly_Sales'].min():,.0f} – ${df['Weekly_Sales'].max():,.0f}", _GREEN),
        ("Top Store",        f"Store {top_store_id}", f"${top_store_total/1e6:.1f}M total revenue",                            _PURPLE),
        ("Holiday Weeks",    f"{df[df['Holiday_Flag']==1]['Date'].nunique()}",
         "Unique holiday weeks (Super Bowl, Labor Day, Thanksgiving, Christmas)",                                               _AMBER),
    ]
    for col, (label, value, subtitle, color) in zip(kpi_cols, kpi_data):
        col.markdown(
            f"<div style='background:{_CARD};border:1px solid {_BORDER};border-left:4px solid {color};"
            f"border-radius:10px;padding:18px 20px;height:110px;display:flex;flex-direction:column;"
            f"justify-content:space-between;'>"
            f"<span style='font-size:0.68rem;font-weight:600;color:{_MUTED};text-transform:uppercase;"
            f"letter-spacing:0.08em;'>{label}</span>"
            f"<div style='font-size:1.7rem;font-weight:700;color:{_TEXT};font-variant-numeric:tabular-nums;"
            f"line-height:1;'>{value}</div>"
            f"<span style='font-size:0.72rem;color:{_DIM};'>{subtitle}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin:20px 0 0 0;'></div>", unsafe_allow_html=True)

    # ── Interactive Plotly Charts ──────────────────────────────────────────────
    col_a, col_b = st.columns(2, gap="medium")

    # Chart 1: Sales Distribution
    with col_a:
        fig1 = px.histogram(
            df, x="Weekly_Sales", nbins=60,
            labels={"Weekly_Sales": "Weekly Sales ($)", "count": "Weeks"},
            title="Weekly Sales Distribution",
            color_discrete_sequence=[_BLUE],
        )
        fig1.update_traces(marker_line_color=_CARD, marker_line_width=0.4)
        fig1.update_xaxes(tickprefix="$", tickformat=",.0f")
        _dark(fig1)
        _chart_card(fig1, "Most weeks fall between $500K and $2M. The right tail shows rare peak weeks — mostly during holidays.")

    # Chart 2: Average Sales by Store
    with col_b:
        store_avgs = df.groupby("Store")["Weekly_Sales"].mean().reset_index()
        store_avgs.columns = ["Store", "Avg_Sales"]
        bar_colors = [_GREEN if s == top_store_id else _BLUE for s in store_avgs["Store"]]
        fig2 = px.bar(
            store_avgs, x="Store", y="Avg_Sales",
            labels={"Avg_Sales": "Avg Weekly Sales ($)", "Store": "Store"},
            title="Average Sales by Store",
        )
        fig2.update_traces(marker_color=bar_colors)
        fig2.update_yaxes(tickprefix="$", tickformat=",.0f")
        _dark(fig2)
        _chart_card(fig2, f"Clear performance gaps across stores — Store {top_store_id} (green) leads by a wide margin.")

    col_c, col_d = st.columns(2, gap="medium")

    # Chart 3: Sales Trend Over Time
    with col_c:
        ts = (df.groupby(df["Date"].dt.to_period("M").dt.to_timestamp())["Weekly_Sales"]
              .mean().reset_index())
        ts.columns = ["Date", "Avg_Sales"]
        fig3 = px.area(
            ts, x="Date", y="Avg_Sales",
            labels={"Avg_Sales": "Avg Weekly Sales ($)", "Date": ""},
            title="Sales Trend Over Time",
            color_discrete_sequence=[_BLUE],
        )
        fig3.update_traces(line_width=2, fillcolor="rgba(37,99,235,0.15)")
        fig3.update_yaxes(tickprefix="$", tickformat=",.0f")
        _dark(fig3)
        _chart_card(fig3, "Consistent spike toward December each year (holiday season), followed by a drop in January–February.")

    # Chart 4: Holiday vs Regular Weeks
    with col_d:
        hol_df = pd.DataFrame({
            "Week Type": ["Regular Week", "Holiday Week"],
            "Average Sales": [regular_avg, holiday_avg],
        })
        fig4 = px.bar(
            hol_df, x="Week Type", y="Average Sales",
            color="Week Type",
            color_discrete_map={"Holiday Week": _AMBER, "Regular Week": _BLUE},
            title="Holiday vs Regular Weeks",
            text_auto="$,.0f",
        )
        fig4.update_traces(textposition="outside", textfont_color=_MUTED)
        fig4.update_yaxes(tickprefix="$", tickformat=",.0f")
        fig4.update_layout(showlegend=False)
        _dark(fig4)
        _chart_card(fig4, f"Holiday weeks average {holiday_lift_pct:.1f}% higher sales (${holiday_avg:,.0f} vs ${regular_avg:,.0f}).")

    st.markdown("---")

    # ── Business Insights ──────────────────────────────────────────────────────
    st.markdown("### 💡 Business Insights")

    insights = [
        {"icon": "🏆", "color": _PURPLE, "title": f"Store {top_store_id} — Top Performer",
         "text": f"Store {top_store_id} generated ${top_store_total/1e6:.1f}M in total revenue — the highest among all 45 stores. Analyzing its success factors (location, size, management) and applying them elsewhere could lift overall chain performance."},
        {"icon": "🎄", "color": _AMBER, "title": f"Holidays Boost Sales by {holiday_lift_pct:.1f}%",
         "text": f"Holiday weeks average ${holiday_avg:,.0f} vs ${regular_avg:,.0f} on regular weeks — a {holiday_lift_pct:.1f}% uplift. Investing in pre-holiday campaigns and inventory builds is expected to deliver strong ROI."},
        {"icon": "📅", "color": _GREEN, "title": f"Seasonality — Peak: {peak_month_name}, Slow: {low_month_name}",
         "text": f"Average weekly sales in {peak_month_name} reach ${monthly_avg[peak_month_num]:,.0f} versus ${monthly_avg[low_month_num]:,.0f} in {low_month_name}. Staff up and stock up before {peak_month_name}; plan promotions in {low_month_name} to stimulate demand."},
        {"icon": "📈", "color": _BLUE, "title": "External Factors Drive Sales",
         "text": "Fuel price, temperature, and local unemployment rate are the external indicators most correlated with weekly sales. Rising fuel costs likely reduce consumer purchasing power, negatively impacting store revenue."},
    ]

    ins_cols = st.columns(2)
    for i, ins in enumerate(insights):
        c, ic, ti, tx = ins["color"], ins["icon"], ins["title"], ins["text"]
        ins_cols[i % 2].markdown(
            f"<div style='background:{_CARD};border:1px solid {_BORDER};border-left:4px solid {c};"
            f"border-radius:10px;padding:16px 18px;margin-bottom:12px;'>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>"
            f"<span style='font-size:1rem;'>{ic}</span>"
            f"<span style='font-size:0.78rem;font-weight:600;color:{c};text-transform:uppercase;letter-spacing:0.05em;'>{ti}</span>"
            f"</div>"
            f"<p style='margin:0;font-size:0.875rem;line-height:1.65;color:#CBD5E1;'>{tx}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander("📋 Dataset Statistics"):
        st.table(df.describe().round(2))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("### 🤖 Model Performance")
    st.caption(f"Comparison between Linear Regression (baseline) and Random Forest. Winner: **{winner_model or 'best model'}** (lower RMSE).")

    # ── Summary metric cards ───────────────────────────────────────────────────
    if r2_w is not None:
        acc_pct = r2_w * 100
        err_pct = (rmse_w / avg_weekly) * 100 if avg_weekly else 0
        model_name = type(model).__name__
        model_desc = ("An ensemble of 100 decision trees — robust against overfitting."
                      if "Forest" in model_name
                      else "Finds the best-fit linear relationship across all 12 features.")
        summary_cards = [
            ("🎯", "R² Score",           f"{r2_w:.4f}",     f"Explains {acc_pct:.1f}% of sales variance (1.0 = perfect).",               _GREEN),
            ("📏", "Typical Error (RMSE)", f"${rmse_w:,.0f}", f"Predictions are typically within ±${rmse_w:,.0f} ({err_pct:.1f}% of avg).", _BLUE),
            ("🤖", "Winning Model",       model_name,        model_desc,                                                                   _PURPLE),
        ]
        m_cols = st.columns(3)
        for col, (icon, title, value, desc, color) in zip(m_cols, summary_cards):
            col.markdown(
                f"<div style='background:{_CARD};border:1px solid {_BORDER};border-top:3px solid {color};"
                f"border-radius:10px;padding:18px 20px;margin-bottom:16px;'>"
                f"<div style='font-size:1.4rem;margin-bottom:6px;'>{icon}</div>"
                f"<p style='margin:0 0 4px 0;font-size:0.72rem;font-weight:600;color:{_MUTED};"
                f"text-transform:uppercase;letter-spacing:0.07em;'>{title}</p>"
                f"<div style='font-size:1.5rem;font-weight:700;color:{color};font-variant-numeric:tabular-nums;"
                f"margin-bottom:6px;'>{value}</div>"
                f"<p style='margin:0;font-size:0.8rem;color:{_DIM};line-height:1.5;'>{desc}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Model Comparison Chart ─────────────────────────────────────────────────
    if rmse_lr and rmse_rf:
        comp_df = pd.DataFrame({
            "Model":  ["Linear Regression", "Random Forest"],
            "RMSE":   [rmse_lr, rmse_rf],
            "R²":     [r2_lr,   r2_rf],
            "Winner": [m == winner_model for m in ["Linear Regression", "Random Forest"]],
        })
        bar_clr = [_GREEN if w else "#334155" for w in comp_df["Winner"]]

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            x=comp_df["Model"], y=comp_df["RMSE"],
            marker_color=bar_clr,
            text=[f"${v:,.0f}" for v in comp_df["RMSE"]],
            textposition="outside", textfont_color=_MUTED,
            width=0.45,
        ))
        fig_cmp.update_layout(
            title="RMSE Comparison — Lower is Better",
            yaxis_title="RMSE ($)", yaxis_tickprefix="$", yaxis_tickformat=",.0f",
            showlegend=False,
        )
        _dark(fig_cmp, height=260)

        st.markdown(
            f"<div style='background:{_CARD};border:1px solid {_BORDER};border-radius:12px;padding:4px 8px 0 8px;'>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption(f"Green bar = winner. With a temporal (chronological) train/test split, {winner_model} achieved lower prediction error.")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2], gap="large")

    # ── Left: evaluation report + feature importance ───────────────────────────
    with col_left:
        st.markdown("#### Detailed Evaluation Report")

        fi_marker = "## Feature Importance"
        eval_part1, fi_raw = (eval_text.split(fi_marker, 1) if fi_marker in eval_text
                               else (eval_text, ""))

        st.markdown(
            f"<div style='background:{_CARD};border:1px solid {_BORDER};border-radius:12px;padding:20px 24px;'>",
            unsafe_allow_html=True,
        )
        st.markdown(eval_part1.strip())
        st.markdown("</div>", unsafe_allow_html=True)

        # Parse feature importance
        fi_data = []
        for line in fi_raw.split("\n"):
            line = line.strip().lstrip("0123456789. ")
            if "(" in line and ")" in line:
                name = line.split("(")[0].strip()
                try:
                    val = float(line.split("(")[1].rstrip(")").strip())
                    if name:
                        fi_data.append((name, val))
                except ValueError:
                    pass
            else:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        fi_data.append((parts[0], float(parts[1])))
                    except ValueError:
                        pass

        if fi_data:
            st.markdown("#### Feature Importance — What Drives Sales?")
            st.caption("Percentage of the Random Forest model's decisions based on each variable.")

            feature_labels = {
                "Sales_Rolling4": "4-Week Rolling Average",
                "Sales_Lag1":     "Previous Week's Sales",
                "Week":           "Week of Year",
                "Temperature":    "Temperature",
                "CPI":            "Consumer Price Index",
                "Holiday_Flag":   "Holiday Week",
                "Fuel_Price":     "Fuel Price",
                "Unemployment":   "Unemployment Rate",
                "Month":          "Month",
                "Store":          "Store Number",
                "Year":           "Year",
                "Is_Quarter_End": "Quarter-End Week",
            }
            bar_palette = [_BLUE, "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"]
            max_val = fi_data[0][1] if fi_data else 1

            for rank, (name, val) in enumerate(fi_data):
                bar_pct = int(val / max_val * 100)
                color   = bar_palette[rank % len(bar_palette)]
                label   = feature_labels.get(name, name)
                pct_str = f"{val*100:.1f}%"
                st.markdown(
                    f"<div style='padding:8px 0;border-bottom:1px solid {_BORDER};'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>"
                    f"<span style='font-size:0.85rem;font-weight:500;color:{_TEXT};'>{label}</span>"
                    f"<span style='font-size:0.82rem;font-weight:600;color:{color};font-variant-numeric:tabular-nums;'>{pct_str}</span>"
                    f"</div>"
                    f"<div style='background:{_BORDER};border-radius:4px;height:7px;'>"
                    f"<div style='width:{bar_pct}%;height:100%;background:{color};border-radius:4px;'></div>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Right: model details + matplotlib chart ────────────────────────────────
    with col_right:
        model_name = type(model).__name__

        st.markdown("#### Model Details")
        st.markdown(
            f"<div style='background:{_CARD};border:1px solid {_BORDER};border-radius:12px;padding:20px;margin-bottom:16px;'>"
            f"<p style='margin:0 0 12px 0;font-size:0.68rem;font-weight:600;color:{_MUTED};"
            f"text-transform:uppercase;letter-spacing:0.08em;'>Active Algorithm</p>"
            f"<div style='background:rgba(37,99,235,0.12);border:1px solid rgba(37,99,235,0.25);"
            f"border-radius:8px;padding:10px 14px;font-size:1rem;font-weight:600;color:#93BBFC;"
            f"margin-bottom:12px;'>{model_name}</div>",
            unsafe_allow_html=True,
        )

        details = []
        if hasattr(model, "n_estimators"):
            details.append(("Decision Trees", str(model.n_estimators)))
        if hasattr(model, "max_depth"):
            details.append(("Max Depth", str(model.max_depth) if model.max_depth else "Unlimited"))
        if hasattr(model, "random_state"):
            details.append(("Random Seed", str(model.random_state)))

        for k, v in details:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"border-top:1px solid {_BORDER};padding:10px 0;'>"
                f"<span style='font-size:0.82rem;color:{_MUTED};'>{k}</span>"
                f"<span style='font-size:0.85rem;font-weight:600;color:{_TEXT};'>{v}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        if hasattr(model, "feature_importances_"):
            features = (list(model.feature_names_in_) if hasattr(model, "feature_names_in_")
                        else ["Store","Year","Month","Week","Is_Quarter_End","Holiday_Flag",
                              "Temperature","Fuel_Price","CPI","Unemployment","Sales_Lag1","Sales_Rolling4"])
            imp  = pd.Series(model.feature_importances_, index=features)
            top5 = imp.nlargest(5).sort_values()

            fig_imp, ax = plt.subplots(figsize=(4, 2.6))
            fig_imp.patch.set_facecolor(_CARD)
            ax.set_facecolor(_CARD)
            colors = ["#1D4ED8","#2563EB","#3B82F6","#60A5FA","#93C5FD"]
            bars = ax.barh(top5.index, top5.values, color=colors, height=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor(_BORDER)
            ax.tick_params(colors=_MUTED, labelsize=7)
            ax.set_xlabel("Importance", fontsize=7, color=_MUTED)
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.005, bar.get_y() + bar.get_height()/2,
                        f"{w:.2f}", va="center", ha="left", fontsize=7, color="#CBD5E1")
            ax.set_xlim(0, top5.values.max() * 1.28)
            ax.grid(axis="x", color=_BORDER, linewidth=0.5, linestyle="--")
            ax.set_axisbelow(True)
            fig_imp.tight_layout(pad=0.6)

            st.markdown(
                f"<div style='background:{_CARD};border:1px solid {_BORDER};border-radius:12px;padding:14px;'>",
                unsafe_allow_html=True,
            )
            st.pyplot(fig_imp, use_container_width=True)
            plt.close(fig_imp)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    card_path = ARTIFACTS_DIR / "model_card.md"
    if card_path.exists():
        with st.expander("📄 Model Card — Full Documentation"):
            st.markdown(card_path.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICT SALES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown("### 🔮 Weekly Sales Forecast")
    st.caption("Select a store and input parameters to generate a machine-learning sales prediction.")

    store_stats = df.groupby("Store")["Weekly_Sales"].agg(["mean", "std"]).round(0)

    selected_store_for_fill = st.selectbox(
        "Select store to pre-fill defaults",
        options=sorted(df["Store"].unique()),
        key="store_select_top",
    )
    store_mean   = float(store_stats.loc[selected_store_for_fill, "mean"])
    store_df     = df[df["Store"] == selected_store_for_fill]
    default_temp = float(store_df["Temperature"].mean())
    default_fuel = float(store_df["Fuel_Price"].mean())
    default_cpi  = float(store_df["CPI"].mean())
    default_unemp = float(store_df["Unemployment"].mean())

    st.info(
        f"**Store {selected_store_for_fill}** · "
        f"Historical avg sales: **${store_mean:,.0f}**/week · "
        f"Avg temperature: **{default_temp:.1f}°F** · "
        f"Avg fuel price: **${default_fuel:.2f}**"
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.markdown("**🏪 Store & Date**")
            store  = st.selectbox("Store", options=sorted(df["Store"].unique()),
                                  index=list(sorted(df["Store"].unique())).index(selected_store_for_fill))
            year   = st.selectbox("Year", [2010, 2011, 2012, 2013])
            month  = st.selectbox("Month", list(range(1, 13)),
                                  format_func=lambda m: calendar.month_name[m], index=5)
            week   = st.slider("Week of Year", 1, 52, 26)

        with col2:
            st.markdown("**🌤️ Market Conditions**")
            holiday        = st.checkbox("Holiday / Major Promotion Week")
            is_quarter_end = st.checkbox("Quarter-End Week (Mar / Jun / Sep / Dec)")
            temperature    = st.number_input("Temperature (°F)", -20.0, 120.0, default_temp, step=0.5,
                                             help="Average regional temperature (historical range: -2°F to 100°F)")
            fuel_price     = st.number_input("Fuel Price ($/gal)", 2.0, 5.0, default_fuel, step=0.01,
                                             help="Regional average fuel price — affects consumer purchasing power")

        with col3:
            st.markdown("**📊 Economic Indicators & Historical Sales**")
            cpi          = st.number_input("Consumer Price Index (CPI)", 120.0, 250.0, default_cpi, step=0.1,
                                           help="Cost-of-living index. Higher = more expensive environment")
            unemployment = st.number_input("Unemployment (%)", 3.0, 15.0, default_unemp, step=0.1,
                                           help="Regional unemployment rate — impacts consumer spending")
            sales_lag1   = st.number_input("Last Week's Sales ($)", min_value=0.0, value=store_mean,
                                           step=1000.0, help="Most influential feature in the model")
            sales_rolling4 = st.number_input("4-Week Rolling Average Sales ($)", min_value=0.0,
                                              value=store_mean, step=1000.0,
                                              help="Average of this store's sales over the past 4 weeks")

        submitted = st.form_submit_button("🔮 Generate Forecast", use_container_width=True)

    # ── Result ────────────────────────────────────────────────────────────────
    if submitted:
        input_data = pd.DataFrame([{
            "Store": store, "Year": year, "Month": month, "Week": week,
            "Is_Quarter_End": int(is_quarter_end), "Holiday_Flag": int(holiday),
            "Temperature": temperature, "Fuel_Price": fuel_price,
            "CPI": cpi, "Unemployment": unemployment,
            "Sales_Lag1": sales_lag1, "Sales_Rolling4": sales_rolling4,
        }])

        with st.spinner("Running model inference..."):
            prediction = model.predict(input_data)[0]

        store_avg  = df[df["Store"] == store]["Weekly_Sales"].mean()
        diff       = prediction - store_avg
        pct        = (diff / store_avg) * 100 if store_avg else 0
        direction  = "above avg" if diff > 0 else "below avg"
        diff_color = _GREEN if diff > 0 else _RED
        arrow      = "▲" if diff > 0 else "▼"

        st.markdown("<div style='margin:24px 0 0 0;'></div>", unsafe_allow_html=True)

        res_col, gauge_col = st.columns([1, 1], gap="large")

        # ── Text result card ──
        with res_col:
            st.markdown(
                f"<div style='background:linear-gradient(135deg,{_CARD} 0%,#1A2235 100%);"
                f"border:1px solid {_BORDER};border-top:3px solid {_BLUE};"
                f"border-radius:14px;padding:28px 32px;text-align:center;height:100%;'>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='margin:0 0 4px 0;font-size:0.72rem;font-weight:600;color:{_MUTED};"
                f"text-transform:uppercase;letter-spacing:0.1em;'>"
                f"Sales Forecast — Store {store} · {calendar.month_name[month]} {year}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:3rem;font-weight:800;color:{_TEXT};font-variant-numeric:tabular-nums;"
                f"letter-spacing:-0.03em;line-height:1.1;margin:10px 0 14px 0;'>${prediction:,.0f}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='display:inline-flex;align-items:center;gap:6px;"
                f"background:rgba(255,255,255,0.04);border:1px solid {_BORDER};"
                f"border-radius:20px;padding:6px 16px;margin-bottom:22px;'>"
                f"<span style='font-size:1rem;color:{diff_color};'>{arrow}</span>"
                f"<span style='font-size:0.88rem;font-weight:600;color:{diff_color};'>"
                f"${abs(diff):,.0f} ({abs(pct):.1f}%)</span>"
                f"<span style='font-size:0.82rem;color:{_DIM};'>&nbsp;{direction} for Store {store}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='display:flex;justify-content:center;gap:40px;"
                f"border-top:1px solid {_BORDER};padding-top:18px;'>"
                f"<div style='text-align:center;'><p style='margin:0;font-size:0.7rem;color:{_DIM};"
                f"text-transform:uppercase;'>Store {store} Avg</p>"
                f"<p style='margin:4px 0 0;font-size:1.05rem;font-weight:600;color:{_MUTED};'>"
                f"${store_avg:,.0f}</p></div>"
                f"<div style='text-align:center;'><p style='margin:0;font-size:0.7rem;color:{_DIM};"
                f"text-transform:uppercase;'>Forecast</p>"
                f"<p style='margin:4px 0 0;font-size:1.05rem;font-weight:600;color:{_TEXT};'>"
                f"${prediction:,.0f}</p></div>"
                f"<div style='text-align:center;'><p style='margin:0;font-size:0.7rem;color:{_DIM};"
                f"text-transform:uppercase;'>Variance</p>"
                f"<p style='margin:4px 0 0;font-size:1.05rem;font-weight:600;color:{diff_color};'>"
                f"{pct:+.1f}%</p></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Gauge chart ──
        with gauge_col:
            max_sales = float(df["Weekly_Sales"].max())
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                delta={"reference": store_avg, "relative": True,
                       "valueformat": ".1%", "font": {"size": 14, "color": diff_color}},
                number={"prefix": "$", "valueformat": ",.0f",
                        "font": {"color": _TEXT, "size": 28, "family": "Inter"}},
                title={"text": f"Predicted vs Store {store} Average",
                       "font": {"color": _MUTED, "size": 12}},
                gauge={
                    "axis": {"range": [0, max_sales],
                             "tickprefix": "$", "tickformat": ",.0f",
                             "tickcolor": _MUTED, "tickfont": {"color": _MUTED, "size": 9},
                             "nticks": 5},
                    "bar":  {"color": _BLUE, "thickness": 0.22},
                    "bgcolor": _CARD,
                    "bordercolor": _BORDER,
                    "borderwidth": 1,
                    "steps": [
                        {"range": [0, regular_avg],  "color": "#1A2235"},
                        {"range": [regular_avg, max_sales], "color": "#1E2D45"},
                    ],
                    "threshold": {
                        "line":      {"color": _GREEN, "width": 3},
                        "thickness": 0.8,
                        "value":     store_avg,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor=_CARD,
                font={"family": "Inter"},
                height=300,
                margin=dict(l=30, r=30, t=60, b=20),
            )
            st.markdown(
                f"<div style='background:{_CARD};border:1px solid {_BORDER};border-radius:14px;padding:4px;'>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption(f"Green line = Store {store} historical average (${store_avg:,.0f}). Blue arc = predicted sales.")
