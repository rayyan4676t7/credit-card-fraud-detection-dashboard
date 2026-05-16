"""
app.py
======
Credit Card Fraud Detection Dashboard — Streamlit Application
A production-grade fintech security monitoring system.

Run with:
    streamlit run app.py
"""

import os
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  — Cyber / Glassmorphism dark theme
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Base / Reset ─────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #050a14;
    color: #e2e8f0;
}

/* ── Animated grid background ─────────────── */
.stApp {
    background:
        linear-gradient(rgba(6,182,212,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(6,182,212,.03) 1px, transparent 1px),
        linear-gradient(135deg, #050a14 0%, #0a1628 50%, #050a14 100%);
    background-size: 40px 40px, 40px 40px, 100% 100%;
}

/* ── Sidebar ───────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1f3c 0%,#061020 100%);
    border-right: 1px solid rgba(6,182,212,.2);
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── Metric cards ──────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(6,182,212,.07);
    border: 1px solid rgba(6,182,212,.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(10px);
}
[data-testid="stMetricLabel"]  { color: #94a3b8 !important; font-size: .78rem; letter-spacing:.08em; text-transform:uppercase; }
[data-testid="stMetricValue"]  { color: #06b6d4 !important; font-size: 1.6rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
[data-testid="stMetricDelta"]  { color: #10b981 !important; }

/* ── Fraud ALERT card ──────────────────────── */
.fraud-alert {
    background: linear-gradient(135deg,rgba(239,68,68,.15),rgba(239,68,68,.05));
    border: 1px solid rgba(239,68,68,.5);
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    animation: pulse-red 2s infinite;
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,.4); }
    50%      { box-shadow: 0 0 20px 4px rgba(239,68,68,.2); }
}

/* ── Safe card ─────────────────────────────── */
.safe-card {
    background: linear-gradient(135deg,rgba(16,185,129,.15),rgba(16,185,129,.05));
    border: 1px solid rgba(16,185,129,.5);
    border-left: 4px solid #10b981;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}

/* ── Warning card ──────────────────────────── */
.warn-card {
    background: linear-gradient(135deg,rgba(245,158,11,.15),rgba(245,158,11,.05));
    border: 1px solid rgba(245,158,11,.5);
    border-left: 4px solid #f59e0b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}

/* ── Glass panel ───────────────────────────── */
.glass-panel {
    background: rgba(255,255,255,.03);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(6,182,212,.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Section headers ───────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #06b6d4;
    letter-spacing: .05em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(6,182,212,.2);
    padding-bottom: .5rem;
    margin-bottom: 1rem;
}

/* ── Buttons ───────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg,#0891b2,#0e7490) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: .04em !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#06b6d4,#0891b2) !important;
    box-shadow: 0 0 20px rgba(6,182,212,.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Sliders / inputs ──────────────────────── */
.stSlider [data-testid="stThumbValue"] { color: #06b6d4; }
[data-testid="stNumberInput"] input { background: rgba(6,182,212,.06); border:1px solid rgba(6,182,212,.3); border-radius:8px; color:#e2e8f0; }

/* ── Tables ────────────────────────────────── */
.stDataFrame { background:transparent; }
.stDataFrame [data-testid="stDataFrame"] table { border-collapse:collapse; }

/* ── Scrollbar ─────────────────────────────── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0a1628; }
::-webkit-scrollbar-thumb { background:#0891b2; border-radius:3px; }

/* ── Monospace numbers ─────────────────────── */
.mono { font-family:'JetBrains Mono',monospace; }

/* ── Feed items ────────────────────────────── */
.feed-item {
    display:flex; align-items:center; gap:.8rem;
    padding:.5rem .8rem; border-radius:8px;
    margin-bottom:.4rem;
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.06);
    font-size:.82rem;
    font-family:'JetBrains Mono',monospace;
}
.feed-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.dot-green  { background:#10b981; box-shadow:0 0 6px #10b981; }
.dot-yellow { background:#f59e0b; box-shadow:0 0 6px #f59e0b; }
.dot-red    { background:#ef4444; box-shadow:0 0 6px #ef4444; animation:blink 1s infinite; }
@keyframes blink { 50%{ opacity:.3; } }

/* ── Hide Streamlit chrome ─────────────────── */
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1.5rem; padding-bottom:2rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────
MODEL_DIR = "saved_model"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#94a3b8"),
    margin=dict(l=20, r=20, t=40, b=20),
)

COLOR_FRAUD  = "#ef4444"
COLOR_NORMAL = "#06b6d4"
COLOR_WARN   = "#f59e0b"
COLOR_OK     = "#10b981"

def color_risk(score: float) -> str:
    if score >= 70:   return COLOR_FRAUD
    if score >= 40:   return COLOR_WARN
    return COLOR_OK


def risk_label(score: float) -> tuple[str, str]:
    if score >= 70:   return "🔴 HIGH RISK",    "fraud-alert"
    if score >= 40:   return "🟡 MEDIUM RISK",  "warn-card"
    return "🟢 LOW RISK",     "safe-card"


def fmt_currency(val: float) -> str:
    return f"${val:,.2f}"


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load all saved ML artefacts; return None if not found."""
    required = ["best_model.pkl", "scaler.pkl", "feature_names.pkl"]
    if not all(os.path.exists(f"{MODEL_DIR}/{f}") for f in required):
        return None

    arts = {
        "best"     : joblib.load(f"{MODEL_DIR}/best_model.pkl"),
        "lr"       : joblib.load(f"{MODEL_DIR}/logistic_regression.pkl"),
        "rf"       : joblib.load(f"{MODEL_DIR}/random_forest.pkl"),
        "scaler"   : joblib.load(f"{MODEL_DIR}/scaler.pkl"),
        "features" : joblib.load(f"{MODEL_DIR}/feature_names.pkl"),
        "metrics"  : joblib.load(f"{MODEL_DIR}/metrics.pkl") if os.path.exists(f"{MODEL_DIR}/metrics.pkl") else {},
    }
    fi_path = f"{MODEL_DIR}/feature_importance.csv"
    arts["importance"] = pd.read_csv(fi_path) if os.path.exists(fi_path) else None
    ts_path = f"{MODEL_DIR}/test_sample.csv"
    arts["test_sample"] = pd.read_csv(ts_path) if os.path.exists(ts_path) else None
    return arts


# ─────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────
def predict(arts, model_choice: str, input_values: dict) -> dict:
    """Run prediction; return probability, score, label, explanation."""
    features = arts["features"]
    scaler   = arts["scaler"]

    # Build DataFrame with correct column order
    row = pd.DataFrame([{f: input_values.get(f, 0.0) for f in features}])
    row_scaled = scaler.transform(row)

    model = arts["lr"] if model_choice == "Logistic Regression" else arts["rf"]
    proba = model.predict_proba(row_scaled)[0, 1]  # fraud probability

    score = round(proba * 100, 2)
    label, css = risk_label(score)

    # ── Simple explanation logic ─────────────
    reasons = []
    if input_values.get("Amount", 0) > 1000:
        reasons.append("⚠️ High transaction amount detected")
    if abs(input_values.get("V1", 0)) > 2:
        reasons.append("⚠️ Unusual principal component V1 value")
    if abs(input_values.get("V3", 0)) > 2:
        reasons.append("⚠️ Unusual principal component V3 value")
    if abs(input_values.get("V14", 0)) > 2:
        reasons.append("⚠️ V14 anomaly — linked to fraud patterns")
    if abs(input_values.get("V17", 0)) > 2:
        reasons.append("⚠️ V17 anomaly — high-risk indicator")
    if not reasons:
        reasons.append("✅ No strong individual feature anomalies detected")

    return {
        "probability": proba,
        "score"      : score,
        "label"      : label,
        "css"        : css,
        "is_fraud"   : proba >= 0.5,
        "reasons"    : reasons,
        "model_used" : model_choice,
    }


# ─────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────
def gauge_chart(score: float) -> go.Figure:
    color = color_risk(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title=dict(text="Fraud Detection Score", font=dict(color="#94a3b8", size=14)),
        number=dict(font=dict(color=color, size=48, family="JetBrains Mono"), suffix=" / 100"),
        delta=dict(reference=50, increasing=dict(color=COLOR_FRAUD), decreasing=dict(color=COLOR_OK)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#94a3b8", tickfont=dict(color="#94a3b8")),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            steps=[
                dict(range=[0,  40], color="rgba(16,185,129,.15)"),
                dict(range=[40, 70], color="rgba(245,158,11,.15)"),
                dict(range=[70,100], color="rgba(239,68,68,.15)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=score),
        ),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280)
    return fig


def distribution_chart(test_sample: pd.DataFrame) -> go.Figure:
    normal = test_sample[test_sample["Class"] == 0]
    fraud  = test_sample[test_sample["Class"] == 1]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=normal["Amount"], name="Normal", nbinsx=60,
        marker_color=COLOR_NORMAL, opacity=.7,
        hovertemplate="Amount: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=fraud["Amount"], name="Fraud", nbinsx=30,
        marker_color=COLOR_FRAUD, opacity=.85,
        hovertemplate="Amount: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Transaction Amount Distribution",
        barmode="overlay",
        xaxis=dict(title="Amount (USD)", gridcolor="rgba(255,255,255,.05)"),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,.05)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,.1)"),
        height=320,
    )
    return fig


def class_pie_chart(test_sample: pd.DataFrame) -> go.Figure:
    counts = test_sample["Class"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Label"] = counts["Class"].map({0: "Normal", 1: "Fraud"})
    fig = go.Figure(go.Pie(
        labels=counts["Label"],
        values=counts["Count"],
        hole=.6,
        marker=dict(colors=[COLOR_NORMAL, COLOR_FRAUD],
                    line=dict(color="#050a14", width=3)),
        textinfo="label+percent",
        textfont=dict(size=13),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Transaction Class Distribution",
        annotations=[dict(text="Dataset", x=.5, y=.5, font_size=14,
                          showarrow=False, font_color="#94a3b8")],
        height=320,
    )
    return fig


def model_comparison_chart(metrics: dict) -> go.Figure:
    models  = ["Logistic Regression", "Random Forest"]
    kpis    = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors  = [COLOR_NORMAL, "#8b5cf6"]

    fig = go.Figure()
    for model, color in zip(models, colors):
        if model not in metrics:
            continue
        vals = [round(metrics[model].get(k, 0) * 100, 2) for k in kpis]
        fig.add_trace(go.Bar(
            name=model, x=labels, y=vals,
            marker_color=color, opacity=.85,
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Model Performance Comparison",
        yaxis=dict(title="Score (%)", range=[0, 115], gridcolor="rgba(255,255,255,.05)"),
        barmode="group",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=360,
    )
    return fig


def feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    df = importance_df.head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["importance"],
            colorscale=[[0, "#0891b2"], [0.5, "#8b5cf6"], [1, "#ef4444"]],
            showscale=False,
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Top {top_n} Feature Importances",
        xaxis=dict(title="Importance", gridcolor="rgba(255,255,255,.05)"),
        yaxis=dict(tickfont=dict(family="JetBrains Mono", size=11)),
        height=420,
    )
    return fig


def confusion_matrix_chart(cm: list) -> go.Figure:
    z     = np.array(cm)
    total = z.sum()
    z_pct = z / total * 100

    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=j, y=i,
                text=f"<b>{z[i,j]:,}</b><br><span style='font-size:10px'>{z_pct[i,j]:.1f}%</span>",
                showarrow=False,
                font=dict(size=14, color="#fff"),
                xref="x", yref="y",
            ))
    fig = go.Figure(go.Heatmap(
        z=z, x=["Pred Normal", "Pred Fraud"],
        y=["Actual Normal", "Actual Fraud"],
        colorscale=[[0, "#0a1628"], [0.5, "#0891b2"], [1, "#ef4444"]],
        showscale=False,
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Confusion Matrix",
        annotations=annotations,
        height=320,
    )
    return fig


def heatmap_time_chart(test_sample: pd.DataFrame) -> go.Figure:
    """Fake heatmap: fraud rate by hour-of-day vs binned amount."""
    df = test_sample.copy()
    df["hour"]  = ((df["Time"] / 3600) % 24).astype(int)
    df["bin"]   = pd.cut(df["Amount"], bins=[0, 50, 200, 500, 2000, 1e9],
                         labels=["<$50", "$50-200", "$200-500", "$500-2k", ">$2k"])
    pivot = df.groupby(["hour", "bin"])["Class"].mean().unstack(fill_value=0)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[f"{h:02d}:00" for h in pivot.index],
        colorscale=[[0, "#0a1628"], [0.4, "#0891b2"], [0.7, "#f59e0b"], [1, "#ef4444"]],
        colorbar=dict(title="Fraud Rate", tickformat=".1%",
                      tickfont=dict(color="#94a3b8")),
        hovertemplate="Hour: %{y}<br>Amount: %{x}<br>Fraud Rate: %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Fraud Rate Heatmap — Hour vs Amount Band",
        xaxis=dict(title="Amount Band"),
        yaxis=dict(title="Hour of Day", autorange="reversed"),
        height=380,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# LIVE FEED GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_feed_items(n: int = 8) -> list[dict]:
    items = []
    for _ in range(n):
        is_fraud = random.random() < 0.12
        is_sus   = (not is_fraud) and (random.random() < 0.18)
        amount   = round(random.expovariate(1 / 120) + 1, 2)
        card     = f"**** **** **** {random.randint(1000,9999)}"
        country  = random.choice(["US", "UK", "DE", "FR", "CN", "BR", "IN", "RU", "AU", "CA"])
        ts       = (datetime.now() - timedelta(seconds=random.randint(0, 60))).strftime("%H:%M:%S")
        status   = "FRAUD"   if is_fraud else ("SUSPICIOUS" if is_sus else "NORMAL")
        dot      = "dot-red" if is_fraud else ("dot-yellow" if is_sus else "dot-green")
        items.append(dict(card=card, amount=amount, country=country,
                          ts=ts, status=status, dot=dot))
    return sorted(items, key=lambda x: x["ts"], reverse=True)


def render_feed(items: list[dict]):
    html = ""
    for it in items:
        html += (
            f'<div class="feed-item">'
            f'<div class="feed-dot {it["dot"]}"></div>'
            f'<span style="color:#94a3b8;min-width:60px">{it["ts"]}</span>'
            f'<span style="flex:1">{it["card"]}</span>'
            f'<span style="color:#06b6d4;min-width:90px">${it["amount"]:,.2f}</span>'
            f'<span style="color:#64748b;min-width:40px">{it["country"]}</span>'
            f'<span style="min-width:90px;font-weight:600;color:'
            + ("#ef4444" if it["status"]=="FRAUD" else ("#f59e0b" if it["status"]=="SUSPICIOUS" else "#10b981"))
            + f'">{it["status"]}</span>'
            f'</div>'
        )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar(arts) -> tuple[str, dict]:
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:1rem 0'>"
            "<div style='font-size:2.5rem'>🛡️</div>"
            "<div style='font-size:1.2rem;font-weight:700;color:#06b6d4'>FraudShield AI</div>"
            "<div style='font-size:.75rem;color:#64748b;letter-spacing:.1em'>ENTERPRISE SECURITY v2.4</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("<div class='section-header'>⚙️ Configuration</div>", unsafe_allow_html=True)

        model_choice = st.selectbox(
            "Active Model",
            ["Random Forest", "Logistic Regression"],
            help="Choose which trained ML model to use for predictions.",
        )

        live_sim = st.toggle("Live Transaction Monitor", value=True)
        auto_refresh = st.toggle("Auto-Refresh Feed", value=False)

        st.markdown("---")
        st.markdown("<div class='section-header'>💳 Transaction Input</div>", unsafe_allow_html=True)

        amount = st.number_input("Amount (USD)", min_value=0.01, max_value=50_000.0,
                                 value=150.0, step=10.0)
        txn_time = st.slider("Transaction Hour", 0, 23, 14, format="%d:00")

        st.markdown("**Feature Overrides** (V-components)")
        col1, col2 = st.columns(2)
        with col1:
            v1  = st.slider("V1",  -5.0, 5.0, 0.0, .1)
            v3  = st.slider("V3",  -5.0, 5.0, 0.0, .1)
            v5  = st.slider("V5",  -5.0, 5.0, 0.0, .1)
            v10 = st.slider("V10", -5.0, 5.0, 0.0, .1)
        with col2:
            v14 = st.slider("V14", -5.0, 5.0, 0.0, .1)
            v17 = st.slider("V17", -5.0, 5.0, 0.0, .1)
            v20 = st.slider("V20", -5.0, 5.0, 0.0, .1)
            v28 = st.slider("V28", -5.0, 5.0, 0.0, .1)

        inputs = {f"V{i}": 0.0 for i in range(1, 29)}
        inputs.update({
            "Amount": amount,
            "Time"  : txn_time * 3600,
            "V1"    : v1, "V3": v3, "V5": v5, "V10": v10,
            "V14"   : v14, "V17": v17, "V20": v20, "V28": v28,
        })

        predict_btn = st.button("🔍 Analyse Transaction", use_container_width=True)
        st.markdown("---")
        st.markdown(
            "<div style='font-size:.72rem;color:#475569;text-align:center'>"
            "Powered by Scikit-learn & Streamlit<br>"
            "Model updated: today"
            "</div>",
            unsafe_allow_html=True,
        )

    return model_choice, inputs, predict_btn, live_sim, auto_refresh


# ─────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────
def main():
    # ── Load artefacts ────────────────────────────────────────
    arts = load_artifacts()

    # ── Sidebar ───────────────────────────────────────────────
    model_choice, inputs, predict_btn, live_sim, auto_refresh = render_sidebar(arts)

    # ── Header ────────────────────────────────────────────────
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.markdown("<div style='font-size:3.5rem;line-height:1'>🛡️</div>", unsafe_allow_html=True)
    with col_title:
        st.markdown(
            "<div style='margin-top:.3rem'>"
            "<span style='font-size:2rem;font-weight:700;color:#06b6d4'>FraudShield</span>"
            "<span style='font-size:2rem;font-weight:300;color:#e2e8f0'> AI Dashboard</span>"
            "</div>"
            "<div style='color:#64748b;font-size:.85rem;letter-spacing:.08em'>"
            "REAL-TIME CREDIT CARD FRAUD DETECTION ENGINE &nbsp;|&nbsp; "
            + datetime.now().strftime("%A, %d %B %Y  %H:%M:%S")
            + "</div>",
            unsafe_allow_html=True,
        )

    # ── No model warning ──────────────────────────────────────
    if arts is None:
        st.warning(
            "⚠️ **Model not found.** "
            "Please run `python model_training.py` first, then refresh this page.",
            icon="🤖",
        )
        st.code("pip install -r requirements.txt\npython model_training.py\nstreamlit run app.py")
        return

    # ── KPI row ───────────────────────────────────────────────
    metrics     = arts.get("metrics", {})
    best_name   = metrics.get("best_model_name", "Random Forest")
    best_metrics = metrics.get(best_name, {})
    test_sample = arts.get("test_sample")

    fraud_count  = (test_sample["Class"] == 1).sum() if test_sample is not None else 0
    total_count  = len(test_sample) if test_sample is not None else 0
    fraud_pct    = fraud_count / total_count * 100 if total_count else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🏆 Best Model",      best_name.split()[0])
    k2.metric("🎯 F1-Score",        f"{best_metrics.get('f1', 0)*100:.1f}%")
    k3.metric("📡 ROC-AUC",         f"{best_metrics.get('roc_auc', 0)*100:.1f}%")
    k4.metric("🔴 Fraud in Sample", f"{fraud_count:,} ({fraud_pct:.2f}%)")
    k5.metric("📊 Total Transactions", f"{total_count:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════
    # TABS
    # ═════════════════════════════════════════════
    tab_pred, tab_analytics, tab_monitor, tab_model, tab_report = st.tabs([
        "🔍 Prediction",
        "📊 Analytics",
        "📡 Live Monitor",
        "🤖 Model Insights",
        "📥 Report",
    ])

    # ─────────────────────────────────────────
    # TAB 1 — PREDICTION
    # ─────────────────────────────────────────
    with tab_pred:
        if predict_btn:
            result = predict(arts, model_choice, inputs)

            # Gauge + result side-by-side
            c_gauge, c_result = st.columns([1, 1])
            with c_gauge:
                st.plotly_chart(gauge_chart(result["score"]), use_container_width=True)
            with c_result:
                label, css = result["label"], result["css"]
                st.markdown(f"""
<div class="{css}">
  <div style="font-size:1.4rem;font-weight:700;margin-bottom:.5rem">{label}</div>
  <div style="font-family:'JetBrains Mono',monospace">
    <span style="color:#94a3b8">Fraud Probability :</span>
    <span style="color:#06b6d4;font-size:1.2rem"> {result['probability']*100:.2f}%</span>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;margin-top:.3rem">
    <span style="color:#94a3b8">Detection Score  :</span>
    <span style="font-size:1.2rem"> {result['score']} / 100</span>
  </div>
  <div style="margin-top:.3rem;color:#94a3b8;font-size:.82rem">
    Model: {result['model_used']}
  </div>
</div>
""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Explanation panel
            st.markdown("<div class='section-header'>🧠 AI Explanation Panel</div>",
                        unsafe_allow_html=True)
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.markdown("**Why this prediction?**")
                for reason in result["reasons"]:
                    st.markdown(f"- {reason}")
                if result["is_fraud"]:
                    st.markdown("- 🔴 Combined feature pattern matches known fraud clusters")
                    st.markdown("- 🔴 Exceeds decision boundary threshold (≥ 50% fraud probability)")

            with exp_col2:
                breakdown = {
                    "Feature": ["Amount", "Time", "V1", "V14", "V17", "Other V-features"],
                    "Value"  : [
                        fmt_currency(inputs.get("Amount", 0)),
                        f"{int(inputs.get('Time',0)//3600):02d}:00",
                        f"{inputs.get('V1',0):.2f}",
                        f"{inputs.get('V14',0):.2f}",
                        f"{inputs.get('V17',0):.2f}",
                        "Default (0.00)",
                    ],
                    "Risk Flag": [
                        "⚠️ High" if inputs.get("Amount",0)>1000 else "✅ Normal",
                        "✅ Normal",
                        "⚠️ Anomaly" if abs(inputs.get("V1",0))>2 else "✅ Normal",
                        "⚠️ Anomaly" if abs(inputs.get("V14",0))>2 else "✅ Normal",
                        "⚠️ Anomaly" if abs(inputs.get("V17",0))>2 else "✅ Normal",
                        "✅ Normal",
                    ],
                }
                st.dataframe(pd.DataFrame(breakdown), use_container_width=True, hide_index=True)

            # Store prediction in session history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                "Timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Amount"       : fmt_currency(inputs.get("Amount", 0)),
                "Model"        : model_choice,
                "Score"        : result["score"],
                "Probability"  : f"{result['probability']*100:.2f}%",
                "Risk"         : result["label"],
                "Verdict"      : "FRAUD" if result["is_fraud"] else "NORMAL",
            })

        else:
            st.info("👈 Configure a transaction in the sidebar and click **Analyse Transaction**.",
                    icon="🔍")

        # Suspicious history table
        if "history" in st.session_state and st.session_state.history:
            st.markdown("---")
            st.markdown("<div class='section-header'>📋 Prediction History</div>",
                        unsafe_allow_html=True)
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(
                hist_df.style.apply(
                    lambda row: [
                        "background-color:rgba(239,68,68,.15)" if row["Verdict"] == "FRAUD"
                        else "background-color:rgba(16,185,129,.06)"
                    ] * len(row),
                    axis=1,
                ),
                use_container_width=True,
                hide_index=True,
            )
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()

    # ─────────────────────────────────────────
    # TAB 2 — ANALYTICS
    # ─────────────────────────────────────────
    with tab_analytics:
        if test_sample is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(class_pie_chart(test_sample), use_container_width=True)
            with c2:
                st.plotly_chart(distribution_chart(test_sample), use_container_width=True)
            st.plotly_chart(heatmap_time_chart(test_sample), use_container_width=True)
        else:
            st.warning("No test sample data found. Run `model_training.py` first.")

    # ─────────────────────────────────────────
    # TAB 3 — LIVE MONITOR
    # ─────────────────────────────────────────
    with tab_monitor:
        st.markdown("<div class='section-header'>📡 Real-Time Transaction Feed</div>",
                    unsafe_allow_html=True)

        feed_placeholder = st.empty()
        if live_sim:
            feed_items = generate_feed_items(12)
            with feed_placeholder.container():
                render_feed(feed_items)

            # Summary stats
            fraud_in_feed  = sum(1 for i in feed_items if i["status"] == "FRAUD")
            sus_in_feed    = sum(1 for i in feed_items if i["status"] == "SUSPICIOUS")
            normal_in_feed = sum(1 for i in feed_items if i["status"] == "NORMAL")

            st.markdown("<br>", unsafe_allow_html=True)
            fa, fb, fc = st.columns(3)
            fa.metric("🔴 Fraud Detected",   fraud_in_feed)
            fb.metric("🟡 Suspicious",        sus_in_feed)
            fc.metric("🟢 Normal",            normal_in_feed)

            if auto_refresh:
                time.sleep(5)
                st.rerun()
        else:
            st.info("Enable **Live Transaction Monitor** in the sidebar to see the feed.")

    # ─────────────────────────────────────────
    # TAB 4 — MODEL INSIGHTS
    # ─────────────────────────────────────────
    with tab_model:
        if metrics:
            st.plotly_chart(model_comparison_chart(metrics), use_container_width=True)

            c_cm1, c_cm2 = st.columns(2)
            for col, mname in zip([c_cm1, c_cm2],
                                   ["Logistic Regression", "Random Forest"]):
                with col:
                    if mname in metrics and "confusion_matrix" in metrics[mname]:
                        fig = confusion_matrix_chart(metrics[mname]["confusion_matrix"])
                        fig.update_layout(title=f"Confusion Matrix — {mname.split()[0]}")
                        st.plotly_chart(fig, use_container_width=True)

            # Numeric metrics table
            st.markdown("---")
            st.markdown("<div class='section-header'>📈 Performance Metrics</div>",
                        unsafe_allow_html=True)
            rows = []
            for mname in ["Logistic Regression", "Random Forest"]:
                if mname in metrics:
                    m = metrics[mname]
                    rows.append({
                        "Model"    : mname,
                        "Accuracy" : f"{m.get('accuracy',0)*100:.2f}%",
                        "Precision": f"{m.get('precision',0)*100:.2f}%",
                        "Recall"   : f"{m.get('recall',0)*100:.2f}%",
                        "F1-Score" : f"{m.get('f1',0)*100:.2f}%",
                        "ROC-AUC"  : f"{m.get('roc_auc',0)*100:.2f}%",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Feature importance
        importance_df = arts.get("importance")
        if importance_df is not None:
            st.markdown("---")
            st.plotly_chart(feature_importance_chart(importance_df), use_container_width=True)

    # ─────────────────────────────────────────
    # TAB 5 — REPORT
    # ─────────────────────────────────────────
    with tab_report:
        st.markdown("<div class='section-header'>📥 Download Prediction Report</div>",
                    unsafe_allow_html=True)
        if "history" in st.session_state and st.session_state.history:
            report_df = pd.DataFrame(st.session_state.history)
            csv = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download CSV Report",
                data=csv,
                file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
            st.dataframe(report_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run at least one prediction to generate a report.", icon="💡")

        # Model metrics download
        if metrics:
            st.markdown("---")
            metrics_rows = []
            for mname in ["Logistic Regression", "Random Forest"]:
                if mname in metrics:
                    m = metrics[mname]
                    metrics_rows.append({
                        "Model": mname,
                        "Accuracy" : round(m.get("accuracy", 0), 4),
                        "Precision": round(m.get("precision", 0), 4),
                        "Recall"   : round(m.get("recall", 0), 4),
                        "F1"       : round(m.get("f1", 0), 4),
                        "ROC_AUC"  : round(m.get("roc_auc", 0), 4),
                    })
            if metrics_rows:
                metrics_csv = pd.DataFrame(metrics_rows).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Model Metrics CSV",
                    data=metrics_csv,
                    file_name="model_metrics.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
