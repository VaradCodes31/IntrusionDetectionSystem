import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="IDS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- DARK THEME (CUSTOM CSS) ---------------- #
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #
st.title("🛡️ Cybersecurity IDS Dashboard")
st.markdown("### 🔍 Explainable Intrusion Detection System")

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_model.pkl")

model = load_model()

# ---------------- LABEL MAP ---------------- #
label_map = {
    0: "BENIGN",
    1: "Bot",
    2: "DDoS",
    3: "DoS GoldenEye",
    4: "DoS Hulk",
    5: "DoS Slowhttptest",
    6: "DoS slowloris",
    7: "FTP-Patator",
    8: "Heartbleed",
    9: "Infiltration",
    10: "PortScan",
    11: "SSH-Patator",
    12: "Web Attack Brute Force",
    13: "Web Attack Sql Injection",
    14: "Web Attack XSS"
}

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])

# ---------------- TABS ---------------- #
tab1, tab2, tab3 = st.tabs(["🚨 Prediction", "🧠 SHAP Explainability", "📊 Insights"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---------------- PREPROCESS ---------------- #
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

    X = X[expected_features]

    predictions = model.predict(X)
    decoded = [label_map[int(p)] for p in predictions]

    # ---------------- TAB 1: PREDICTION ---------------- #
    with tab1:
        st.subheader("🚨 Detection Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Uploaded Data")
            st.dataframe(df.head())

        with col2:
            st.markdown("### ⚠️ Predictions")

            for i, pred in enumerate(decoded[:10]):
                if pred == "BENIGN":
                    st.success(f"Row {i}: {pred}")
                else:
                    st.error(f"Row {i}: 🚨 {pred}")

    # ---------------- TAB 2: SHAP ---------------- #
    with tab2:
        st.subheader("🧠 SHAP Explanation")

        sample = X.iloc[[0]]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        pred_class = int(predictions[0])

        if isinstance(shap_values, list):
            sv = shap_values[pred_class][0]
        else:
            sv = shap_values[0][:, pred_class]

        st.markdown(f"### 🔍 Prediction: {label_map[pred_class]}")

        fig, ax = plt.subplots()
        shap.plots.bar(
            shap.Explanation(values=sv, data=sample.iloc[0]),
            show=False
        )
        st.pyplot(fig)

    # ---------------- TAB 3: INSIGHTS ---------------- #
    with tab3:
        st.subheader("📊 Attack Insights")

        pred_series = pd.Series(decoded)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Attack Distribution")
            st.bar_chart(pred_series.value_counts())

        with col2:
            st.markdown("### 📌 Summary")

            total = len(decoded)
            attacks = sum([1 for p in decoded if p != "BENIGN"])

            st.metric("Total Samples", total)
            st.metric("Detected Attacks", attacks)

            if attacks > 0:
                st.error("⚠️ Threats Detected in Network Traffic")
            else:
                st.success("✅ Network Traffic is Safe")

else:
    st.info("⬅️ Upload a dataset from the sidebar to begin.")