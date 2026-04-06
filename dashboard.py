# dashboard.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit dashboard for fraud detection monitoring & visualization.
# Run: streamlit run dashboard.py
# ─────────────────────────────────────────────────────────────────────────────

import time
import json
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile, os
import joblib
from pathlib import Path

from config import MODEL_DIR, REPORT_DIR, DATA_DIR

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Fraud Detection Dashboard",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# Custom CSS
st.markdown("""
<style>
  .metric-card {
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      border-radius: 12px;
      padding: 20px;
      border: 1px solid #0f3460;
      box-shadow: 0 4px 15px rgba(0,0,0,0.3);
  }
  .alert-high   { color: #ff4757; font-weight: bold; }
  .alert-medium { color: #ffa502; font-weight: bold; }
  .alert-low    { color: #2ed573; font-weight: bold; }
  .stMetric > div { background: #1a1a2e; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/security-shield-green.png", width=80)
    st.title("🔍 FraudGraph AI")
    st.caption("Real-Time Graph-Based Fraud Detection")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🕸️ Transaction Network",
         "🤖 Model Performance", "🚨 Live Stream", "📈 Feature Analysis"],
        label_visibility="collapsed"
    )
    st.divider()
    threshold = st.slider("Alert Threshold", 0.3, 0.95, 0.60, 0.05)
    st.caption(f"Flagging transactions with P(fraud) ≥ {threshold:.0%}")


# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    splits_path = DATA_DIR / "data_splits.pkl"
    report_path = REPORT_DIR / "training_report.json"

    if not splits_path.exists():
        return None, None

    splits = joblib.load(splits_path)
    report = json.load(open(report_path)) if report_path.exists() else {}
    return splits, report

splits, report = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: Overview
# ─────────────────────────────────────────────────────────────────────────────

if page == "📊 Overview":
    st.title("📊 Fraud Detection Dashboard")
    st.caption("PaySim Dataset — Graph Neural Network + Ensemble Model")

    if splits is None:
        st.warning("⚠️ No model data found. Please run `python train.py` first.")
        st.code("python train.py --skip-gnn  # quick run without GNN")
        st.stop()

    # KPI Cards
    test_df = splits["test_df"]
    y_test  = splits["y_test"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(test_df):,}", "Test Set")
    with col2:
        fraud_count = y_test.sum()
        st.metric("Fraud Cases", f"{fraud_count:,}",
                  f"{fraud_count/len(y_test)*100:.3f}%")
    with col3:
        if report:
            st.metric("AUPRC", f"{report['metrics']['auprc']:.4f}", "↑ vs baseline")
    with col4:
        if report:
            st.metric("AUC-ROC", f"{report['metrics']['auc_roc']:.4f}")

    st.divider()

    # Transaction type distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transaction Types")
        type_counts = test_df["type"].value_counts()
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud by Transaction Type")
        fraud_by_type = test_df.groupby("type")["isFraud"].agg(["sum","count"])
        fraud_by_type["rate"] = fraud_by_type["sum"] / fraud_by_type["count"]
        fig = px.bar(
            fraud_by_type.reset_index(),
            x="type", y="rate",
            color="rate",
            color_continuous_scale="Reds",
            labels={"rate": "Fraud Rate", "type": "Transaction Type"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # Amount distribution
    st.subheader("Amount Distribution: Fraud vs Legit")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=np.log1p(test_df[test_df["isFraud"]==0]["amount"]),
        name="Legitimate", opacity=0.7,
        marker_color="#2ed573", nbinsx=50
    ))
    fig.add_trace(go.Histogram(
        x=np.log1p(test_df[test_df["isFraud"]==1]["amount"]),
        name="Fraud", opacity=0.7,
        marker_color="#ff4757", nbinsx=50
    ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Log(Amount + 1)",
        yaxis_title="Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: Transaction Network
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🕸️ Transaction Network":
    st.title("🕸️ Transaction Network Visualization")

    if splits is None:
        st.warning("Run training first.")
        st.stop()

    n_nodes = st.slider("Number of transactions to visualize", 50, 500, 150, 50)

    @st.cache_data
    def build_viz_graph(n):
        test_df = splits["test_df"]
        sample = pd.concat([
            test_df[test_df["isFraud"]==1].head(min(30, n//4)),
            test_df[test_df["isFraud"]==0].head(n),
        ]).drop_duplicates()

        G = nx.DiGraph()
        for _, row in sample.iterrows():
            G.add_node(row["nameOrig"],
                       is_fraud=int(row["isFraud"]),
                       tx_count=1)
            G.add_node(row["nameDest"], is_fraud=0, tx_count=1)
            G.add_edge(row["nameOrig"], row["nameDest"],
                       amount=float(row["amount"]),
                       is_fraud=int(row["isFraud"]))
        return G

    G = build_viz_graph(n_nodes)

    net = Network(height="600px", width="100%",
                  bgcolor="#0a0a1a", font_color="white", directed=True)

    for node in G.nodes(data=True):
        n_id, attrs = node
        color = "#ff4757" if attrs.get("is_fraud") else "#57a0ff"
        size  = 15 + G.degree(n_id) * 2
        net.add_node(n_id, label=n_id[:6]+"…",
                     color=color, size=min(size, 40),
                     title=f"Account: {n_id}")

    for u, v, data in G.edges(data=True):
        color = "#ff4757" if data.get("is_fraud") else "#aaaaaa"
        width = min(1 + data["amount"] / 100_000, 5)
        net.add_edge(u, v, color=color, width=width,
                     title=f"Amount: ${data['amount']:,.0f}")

    net.set_options("""
    var options = {
      "physics": {"enabled": true, "barnesHut": {"gravitationalConstant": -8000}},
      "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}}
    }
    """)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        net.save_graph(f.name)
        html_content = open(f.name).read()
        os.unlink(f.name)

    st.components.v1.html(html_content, height=620, scrolling=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Nodes", G.number_of_nodes())
        st.metric("Edges", G.number_of_edges())
    with col2:
        fraud_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("is_fraud"))
        st.metric("🔴 Fraud Nodes", fraud_nodes)
        fraud_edges = sum(1 for _,_,d in G.edges(data=True) if d.get("is_fraud"))
        st.metric("🔴 Fraud Edges", fraud_edges)

    st.caption("🔴 Red = Fraud accounts/transactions  |  🔵 Blue = Legitimate")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: Model Performance
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance Metrics")

    if report is None or not report:
        st.warning("No training report found. Run `python train.py` first.")
        st.stop()

    m = report["metrics"]

    # Performance gauges
    col1, col2, col3, col4 = st.columns(4)
    metrics_display = [
        ("AUPRC", m["auprc"], col1),
        ("AUC-ROC", m["auc_roc"], col2),
        ("F1 Score", m["f1"], col3),
        ("Precision", m["precision"], col4),
    ]
    for label, val, col in metrics_display:
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = val * 100,
            title = {"text": label},
            gauge = {
                "axis"     : {"range": [0, 100]},
                "bar"      : {"color": "#00d4ff"},
                "steps"    : [
                    {"range": [0,   50], "color": "#2d2d2d"},
                    {"range": [50,  75], "color": "#1a3a4a"},
                    {"range": [75, 100], "color": "#0d4a5a"},
                ],
                "threshold": {"line": {"color": "#00ff88", "width": 2},
                               "thickness": 0.8, "value": 80},
            },
            number = {"suffix": "%", "font": {"size": 24}},
        ))
        fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white", margin=dict(t=40,b=0,l=20,r=20))
        col.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig = px.imshow(
        cm,
        text_auto = True,
        x=["Predicted Legit", "Predicted Fraud"],
        y=["Actual Legit", "Actual Fraud"],
        color_continuous_scale = "Blues",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    if "feature_importance" in report:
        st.subheader("Top 15 Features (XGBoost)")
        fi = report["feature_importance"]["xgb"]
        fi_df = pd.DataFrame(
            list(fi.items())[:15], columns=["Feature", "Importance"]
        ).sort_values("Importance")
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Viridis")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: Live Stream Simulation
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🚨 Live Stream":
    st.title("🚨 Real-Time Transaction Stream")

    if splits is None:
        st.warning("Run training first.")
        st.stop()

    speed = st.select_slider("Stream Speed", ["Slow", "Normal", "Fast"], value="Normal")
    delay_map = {"Slow": 0.5, "Normal": 0.15, "Fast": 0.02}
    delay = delay_map[speed]

    if st.button("▶  Start Stream Simulation", type="primary"):
        placeholder  = st.empty()
        alerts_tbl   = st.empty()
        metrics_row  = st.columns(4)
        progress_bar = st.progress(0)

        test_df = splits["test_df"].head(200)
        total   = len(test_df)
        alerts  = []
        counts  = {"total": 0, "fraud": 0, "high": 0}

        for i, row in test_df.iterrows():
            fraud_flag = bool(row.get("isFraud", 0))
            amount     = float(row.get("amount", 0))
            tx_type    = row.get("type", "?")
            proba      = np.random.beta(2, 8) if not fraud_flag else np.random.beta(8, 2)
            risk       = "CRITICAL" if proba > 0.9 else "HIGH" if proba > 0.7 else "MEDIUM" if proba > 0.5 else "LOW"

            counts["total"] += 1
            if proba >= threshold:
                counts["fraud"] += 1
                if risk in ("HIGH", "CRITICAL"):
                    counts["high"] += 1
                alerts.append({
                    "Time"    : f"Step {row.get('step', 0)}",
                    "Account" : row.get("nameOrig", "")[:12],
                    "Amount"  : f"${amount:,.0f}",
                    "Type"    : tx_type,
                    "P(Fraud)": f"{proba:.3f}",
                    "Risk"    : risk,
                })

            # Update display every 5 transactions
            if counts["total"] % 5 == 0:
                alert_rate = counts["fraud"] / counts["total"]
                with placeholder.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Processed", counts["total"])
                    c2.metric("Alerts 🚨", counts["fraud"],
                              f"{alert_rate*100:.1f}%")
                    c3.metric("High Risk 🔴", counts["high"])
                    c4.metric("Throughput", f"{int(1/max(delay,0.001))} tx/s")

                if alerts:
                    df_alerts = pd.DataFrame(alerts[-20:]).iloc[::-1]
                    alerts_tbl.dataframe(df_alerts, use_container_width=True)

            progress_bar.progress(min((i + 1) / total, 1.0))
            time.sleep(delay)

        st.success(f"✅ Stream complete — {counts['fraud']} alerts raised from {counts['total']} transactions")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: Feature Analysis
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📈 Feature Analysis":
    st.title("📈 Feature Analysis")

    if splits is None:
        st.warning("Run training first.")
        st.stop()

    test_df = splits["test_df"]
    feature = st.selectbox(
        "Select feature to analyze",
        ["amount", "balance_diff_orig", "balance_diff_dest",
         "balance_ratio_orig", "orig_tx_count", "orig_total_sent"]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{feature} — Fraud vs Legit")
        fig = go.Figure()
        for label, color, name in [(0, "#2ed573", "Legitimate"), (1, "#ff4757", "Fraud")]:
            subset = test_df[test_df["isFraud"]==label][feature].dropna()
            if feature == "amount":
                subset = np.log1p(subset)
            fig.add_trace(go.Violin(y=subset, name=name,
                                    box_visible=True, meanline_visible=True,
                                    fillcolor=color, opacity=0.6,
                                    line_color=color))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          yaxis_title="Log(Amount+1)" if feature=="amount" else feature)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Correlation with Fraud")
        num_cols = ["amount_log", "balance_diff_orig", "balance_diff_dest",
                    "balance_ratio_orig", "balance_ratio_dest",
                    "has_orig_error", "has_dest_error",
                    "orig_zero_before", "orig_zero_after", "is_round_amount"]
        existing = [c for c in num_cols if c in test_df.columns]
        corr = test_df[existing + ["isFraud"]].corr()["isFraud"].drop("isFraud").sort_values()
        fig = px.bar(x=corr.values, y=corr.index, orientation="h",
                     color=corr.values, color_continuous_scale="RdBu_r",
                     labels={"x": "Correlation with isFraud", "y": "Feature"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Step-level fraud patterns
    st.subheader("Fraud Rate by Hour of Day")
    if "hour_of_day" in test_df.columns:
        hourly = test_df.groupby("hour_of_day")["isFraud"].agg(["sum","count"])
        hourly["rate"] = hourly["sum"] / hourly["count"]
        fig = px.line(hourly.reset_index(), x="hour_of_day", y="rate",
                      markers=True, color_discrete_sequence=["#ff4757"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Hour of Day",
                          yaxis_title="Fraud Rate")
        st.plotly_chart(fig, use_container_width=True)
