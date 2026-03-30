# 🛡️ NetSage-IDS: Project Information Document

## 1. Project Overview
**NetSage-IDS** is a state-of-the-art, AI-powered Intrusion Detection System (IDS) designed to identify, categorize, and explain network security threats in real-time. Built on a high-performance Gradient Boosting framework (XGBoost), it provides sub-millisecond detection across 15 distinct attack categories, ranging from DoS/DDoS to sophisticated web-based injections and botnet activity.

---

## 2. Technical Stack
*   **Language**: Python 3.11+
*   **Machine Learning**: XGBoost (Extreme Gradient Boosting), Scikit-Learn
*   **Interface**: Streamlit (Cyber-Ops Terminal Aesthetic)
*   **Explainability**: SHAP (Shapley Additive Explanations), LIME
*   **Visualization**: Plotly, Matplotlib, Seaborn
*   **Persistence**: Joblib (Model & Encoder serialization)

---

## 3. The Technical Pipeline

### A. Data Ingestion & Loading
The system is designed to consume high-fidelity network traffic logs (compatible with the CIC-IDS2017 dataset format).
-   **Path**: `data/raw/`
-   **Protocol**: Automatic concatenation of multiple CSV files with dynamic header cleaning (stripping whitespace and special characters).

### B. Preprocessing & Integrity
A rigorous cleaning pipeline ensures the model trains only on valid, non-redundant data:
1.  **Cleaning**: Automatic handling of `NaN` and `Inf` values frequently found in network capture tools.
2.  **Conversion**: Safe conversion of 70+ network features (packet lengths, inter-arrival times, flags) into numeric formats.
3.  **Leakage Protection**: A strict **Stratified Split** protocol. The `LabelEncoder` is fit strictly on the training set to prevent information leakage from labels found only in the test set.

### C. Detection Engine (XGBoost)
The core logic utilizes a multi-class **XGBoost Classifier**:
-   **Objective**: `multi:softprob` for high-precision probability estimation.
-   **Configuration**: 100 Estimators, Depth of 6, and a learning rate of 0.1, optimized for the M1/M2 Apple Silicon architecture.
-   **Output**: 15 classification indices mapped to specific threat labels.

### D. Explainability Layer (XAI)
NetSage-IDS moves beyond "black-box" detection:
-   **Global SHAP**: Identifies which network features (e.g., Destination Port, Fwd Packet Length) are the most critical across the entire enterprise.
-   **Local SHAP**: For every single alert, the system can display a contribution plot showing the exact packet flags that triggered the warning.

---

## 4. Operational Interface (CSOC Dashboard)

The dashboard provides a "Cyber Security Operations Center" (CSOC) experience with three distinct modules:

### 📡 Live Monitor Mode
Designed for real-time surveillance simulation:
-   **Packet Stream**: A live rolling log of network events.
-   **Neutralization Log**: Immediate visual feedback for blocked threats.
-   **Risk Gauges**: Real-time metrics showing scanning latency and cumulative risk levels.

### 📂 Batch Analysis Mode
For forensic auditing of historical data:
-   **Bulk Audit**: Upload entire days of network traffic for rapid threat scanning.
-   **Threat Distribution**: Automatic generation of interactive Plotly breakdowns of all detected anomalies.
-   **Deep Dive**: SHAP-powered forensic analysis of individual logs.

### 🛠️ System Health
A diagnostics terminal that monitors the status of the detection engine, node availability, and model synchronization.

---

## 5. Security & Robustness Features
*   **Anomaly Fallback**: The system includes a "Safety Catch" that flags unexpected model outputs as "ANOMALY" instead of crashing, ensuring maximum uptime.
*   **Encoder Sync**: Centrally managed Label Encoding to ensure that the Training Pipeline and the Dashboard utilize perfectly synchronized classification maps.
*   **Responsive UI**: Optimized for high-resolution security monitors with a dark-mode, neon-accented glassmorphism aesthetic.

---

## 6. Future Roadmap
1.  **Real-Time Pcap Hooking**: Transitioning from CSV simulation to live `scapy`-based packet sniffing.
2.  **Ensemble Expansion**: Integrating LightGBM and Random Forest for a "Collective Defense" voting mechanism.
3.  **Active Response**: Automated firewall rule generation (IPTables/UFW) upon threat detection.
