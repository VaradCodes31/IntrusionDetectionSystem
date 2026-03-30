import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="🛡️ NetSage-IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- THEME & CSS ---------------- #
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

    :root {
        --primary: #00f2ff;
        --secondary: #7000ff;
        --bg-dark: #0a0b10;
        --card-bg: rgba(255, 255, 255, 0.05);
        --danger: #ff0055;
        --success: #00ff88;
    }

    .main {
        background-color: var(--bg-dark);
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 2px;
        color: var(--primary);
    }

    .stMetric {
        background: var(--card-bg);
        border: 1px solid rgba(0, 242, 255, 0.2);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }

    .stButton>button {
        background: linear-gradient(45deg, var(--secondary), var(--primary));
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 5px;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        transition: 0.3s;
    }

    .stButton>button:hover {
        box-shadow: 0 0 15px var(--primary);
        transform: scale(1.05);
    }

    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
    }

    .log-entry {
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.85rem;
        padding: 5px 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .threat-high { color: var(--danger); font-weight: bold; }
    .threat-low { color: var(--success); }

    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------- UTILS ---------------- #
def load_resources():
    try:
        model = joblib.load("models/xgboost_model.pkl")
        # Try loading dynamic encoder, fallback to hardcoded if not found
        try:
            le = joblib.load("models/label_encoder.pkl")
            label_map = {i: label for i, label in enumerate(le.classes_)}
        except:
            label_map = {
                0: "BENIGN", 1: "Bot", 2: "DDoS", 3: "DoS GoldenEye", 4: "DoS Hulk",
                5: "DoS Slowhttptest", 6: "DoS slowloris", 7: "FTP-Patator",
                8: "Heartbleed", 9: "Infiltration", 10: "PortScan", 11: "SSH-Patator",
                12: "Web Attack Brute Force", 13: "Web Attack Sql Injection", 14: "Web Attack XSS"
            }
        return model, label_map
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        return None, None

model, label_map = load_resources()

def preprocess_data(df, model):
    df.columns = df.columns.str.strip()
    X = df.copy()
    if "Label" in X.columns:
        X = X.drop("Label", axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)
    expected_features = model.get_booster().feature_names
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
    return X[expected_features]

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/security-shield.png", width=100)
    st.title("CSOC Terminal")
    st.markdown("---")
    app_mode = st.radio("📡 SELECT OPERATION MODE", ["Live Monitor", "Batch Analysis", "System Health"])
    
    st.markdown("---")
    st.info("System Status: **ACTIVE**")
    st.markdown(f"**Last Sync:** {datetime.now().strftime('%H:%M:%S')}")
    
    if st.button("RESET ENGINE"):
        st.rerun()

# ---------------- MAIN CONTENT ---------------- #

if app_mode == "Live Monitor":
    st.title("📡 NetSage-IDS: LIVE MONITOR")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Placeholders for metrics
    p1 = col1.empty()
    p2 = col2.empty()
    p3 = col3.empty()
    p4 = col4.empty()
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown('<div class="glass-card"><h3>🛰️ Activity Stream</h3></div>', unsafe_allow_html=True)
        log_placeholder = st.empty()
    
    with c2:
        st.markdown('<div class="glass-card"><h3>🚨 Alert Console</h3></div>', unsafe_allow_html=True)
        alert_placeholder = st.empty()
        st.markdown('<div class="glass-card"><h3>📉 Distribution</h3></div>', unsafe_allow_html=True)
        chart_placeholder = st.empty()

    # Simulation Logic
    if "live_data" not in st.session_state:
        st.session_state.live_data = []
        st.session_state.alerts = []
        st.session_state.stats = {"total": 0, "attacks": 0, "threat_level": 0}

    # Load sample for simulation
    sample_df = pd.read_csv("sample_test.csv")
    
    run_sim = st.checkbox("START SIMULATION", value=True)
    
    if run_sim:
        while True:
            # Pick a random row
            row = sample_df.sample(1)
            X_live = preprocess_data(row, model)
            pred = model.predict(X_live)[0]
            label = label_map[int(pred)]
            
            # Update state
            st.session_state.stats["total"] += 1
            label = label_map.get(int(pred), "ANOMALY")
            
            if label != "BENIGN":
                st.session_state.stats["attacks"] += 1
                st.session_state.alerts.insert(0, f"{datetime.now().strftime('%H:%M:%S')} - WARNING: {label} detected!")
            
            st.session_state.live_data.insert(0, {
                "Time": datetime.now().strftime('%H:%M:%S'),
                "Event": label,
                "Status": "BLOCKED" if label != "BENIGN" else "PASSED",
                "Traffic": np.random.randint(100, 1000)
            })
            
            # Limit logs
            st.session_state.live_data = st.session_state.live_data[:15]
            st.session_state.alerts = st.session_state.alerts[:10]
            
            # Update Dashboard
            p1.metric("PACKETS SCANNED", st.session_state.stats["total"], delta=1)
            p2.metric("THREATS NEUTRALIZED", st.session_state.stats["attacks"], delta=1 if label != "BENIGN" else 0, delta_color="inverse")
            p3.metric("CURRENT RISK", "HIGH" if label != "BENIGN" else "LOW", delta=label if label != "BENIGN" else "OK")
            p4.metric("ENGINE LATENCY", f"{np.random.randint(5, 45)}ms")
            
            with log_placeholder.container():
                st.table(pd.DataFrame(st.session_state.live_data))
            
            with alert_placeholder.container():
                for a in st.session_state.alerts[:5]:
                    st.error(a)
            
            with chart_placeholder.container():
                if len(st.session_state.live_data) > 0:
                    counts = pd.DataFrame([d["Event"] for d in st.session_state.live_data], columns=["Attack"]).value_counts().reset_index()
                    fig = px.pie(counts, values='count', names='Attack', hole=.4, color_discrete_sequence=['#00f2ff', '#7000ff', '#00ff88', '#ff0055', '#505050'])
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, width='stretch')

            time.sleep(1.5)
            # st.rerun() # Not technically needed inside the loop with st.empty, but often safer

elif app_mode == "Batch Analysis":
    st.title("📂 BATCH DATA AUDIT")
    uploaded_file = st.sidebar.file_uploader("Upload Network Traffic (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        X = preprocess_data(df, model)
        preds = model.predict(X)
        decoded = [label_map[int(p)] for p in preds]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown('<div class="glass-card"><h3>📊 Audit Results</h3></div>', unsafe_allow_html=True)
            res_df = df.copy()
            res_df["Detection"] = decoded
            st.dataframe(res_df.head(20), width='stretch')
            
        with col2:
            st.markdown('<div class="glass-card"><h3>📈 Breakdown</h3></div>', unsafe_allow_html=True)
            counts = pd.Series(decoded).value_counts()
            fig = px.bar(counts, color=counts.index, color_discrete_sequence=['#00f2ff', '#7000ff', '#00ff88', '#ff0055', '#505050'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig, width='stretch')
            
            attacks = sum([1 for p in decoded if p != "BENIGN"])
            st.metric("Total Attacks Found", attacks)
            if attacks > 0:
                st.warning("⚠️ Critical threats detected in the uploaded batch.")

        st.markdown("---")
        st.subheader("🧠 SHAP Model Explainability")
        if st.checkbox("Generate SHAP Breakdown (First Row)"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.iloc[[0]])
            pred_class = int(preds[0])
            
            if isinstance(shap_values, list):
                sv = shap_values[pred_class][0]
            else:
                sv = shap_values[0][:, pred_class]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#0a0b10')
            ax.set_facecolor('#0a0b10')
            shap.plots.bar(shap.Explanation(values=sv, data=X.iloc[0]), show=False)
            plt.xticks(color="white")
            plt.yticks(color="white")
            st.pyplot(fig)

elif app_mode == "System Health":
    st.title("🛠️ SYSTEM DIAGNOSTICS")
    st.markdown('<div class="glass-card"><h3>💻 Engine Specifications</h3></div>', unsafe_allow_html=True)
    st.write(f"**Model Type:** XGBoost Classifier")
    st.write(f"**Label Encoding:** Dynamic via label_encoder.pkl")
    st.write(f"**Features Tracked:** {len(model.get_booster().feature_names)}")
    
    st.markdown("---")
    st.markdown('<div class="glass-card"><h3>📡 Cluster Status</h3></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.success("Node 01: ONLINE")
    c2.success("Node 02: ONLINE")
    c3.error("Node 03: OFFLINE (Scheduled Maintenance)")