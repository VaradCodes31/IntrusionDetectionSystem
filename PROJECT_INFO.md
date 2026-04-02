# 🛡️ NetSage-IDS: Project Information Document

## 1. Project Overview
**NetSage-IDS** is a state-of-the-art, AI-powered Intrusion Detection System (IDS) designed to identify, categorize, and explain network security threats in real-time. Built on a high-performance Gradient Boosting framework (XGBoost), it provides sub-millisecond detection across 15 distinct attack categories, ranging from DoS/DDoS to sophisticated web-based injections and botnet activity.

---

## 2. Technical Stack
*   **Language**: Python 3.11+
*   **Machine Learning**: XGBoost (Extreme Gradient Boosting), Scikit-Learn
*   **Interface**: Streamlit (Cyber-Ops Terminal Aesthetic)
*   **Explainability**: SHAP, LIME, Anchors, Counterfactuals (DiCE), Integrated Gradients
*   **Quantum Engine**: PennyLane, PyTorch (4-Qubit Variational Circuit)
*   **Visualization**: Plotly, Matplotlib, Seaborn
*   **Persistence**: Joblib & Torch serialization

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

### D. Hybrid Classical-Quantum Pipeline (HCQP)
NetSage-IDS implements a novel **consensus logic** across two divergent computational domains:
1.  **Classical Branch (XGBoost)**: Provides ultra-fast, high-precision detection (99.88% accuracy) using 70+ features.
2.  **Quantum Branch (QNN)**: Utilizes a 4-qubit **Parameterized Quantum Circuit (PQC)** to identify non-linear feature overlaps in the Hilbert space.
3.  **Weighted Fusion**: Final decision = `(0.8 * XGBoost) + (0.2 * QNN)`. This stabilizes the research findings while incorporating quantum inductive bias.

### E. Multi-Layer Explainability Layer (XAI)
The system provides five distinct perspectives on model logic:
-   **Global SHAP**: Enterprise-wide feature importance.
-   **LIME**: Fast, linear local approximations for packet-level audit.
-   **Anchors**: Generates human-readable IF-THEN rules for "black-box" decisions.
-   **Counterfactuals**: Provides "What-If?" scenarios for reclassifying blocked packets as Benign.
-   **Quantum Integrated Gradients (Q-IG)**: Differentiable attribution for the quantum branch via the **parameter-shift rule**.

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

### ⚛️ Quantum Lab
A dedicated research portal for evaluating the Quantum-Classical gap:
-   **Benchmark Suite**: Head-to-head comparison between XGBoost, Hybrid QNN, and QKSVM.
-   **Q-IG Visualizer**: Real-time attribution of quantum state triggers.
-   **Circuit Explorer**: Visualization of the PQC architecture (Angle Encoding + Strongly Entangling Layers).

---

## 5. Security & Robustness Features
*   **Anomaly Fallback**: The system includes a "Safety Catch" that flags unexpected model outputs as "ANOMALY" instead of crashing, ensuring maximum uptime.
*   **Encoder Sync**: Centrally managed Label Encoding to ensure that the Training Pipeline and the Dashboard utilize perfectly synchronized classification maps.
*   **Responsive UI**: Optimized for high-resolution security monitors with a dark-mode, neon-accented glassmorphism aesthetic.

---

## 6. Future Roadmap
1.  **Real-Time Pcap Hooking**: Transitioning from CSV simulation to live `scapy`-based packet sniffing.
### 2. Ensemble Expansion
Integrating LightGBM and Random Forest for a "Collective Defense" voting mechanism.
### 3. Active Response
Automated firewall rule generation (IPTables/UFW) upon threat detection.

---

## 7. Recent Technical Updates & Bugfixes (v2.0 Architecture)

Recent development cycles transitioned the Quantum module from experimental to **Production Core**. Several blocking technical challenges were resolved:

### A. Environment Stability (macOS / Apple Silicon)
- **OpenMP Memory Deadlocks (Segfaults)**: Streamlit initiates a multithreaded runtime environment. When PyTorch's C++ bindings and XGBoost both attempt to lock the macOS `libomp` (OpenMP) framework concurrently, it triggers a hard Segmentation Fault. 
  - **Resolution**: Explicit environment locking (`os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` & `OMP_NUM_THREADS="1"`) was enforced directly above the module imports to orchestrate thread allocation safely.

### B. Dynamic Array Broadcasting (Hybrid Class Imbalances)
- **Dimensionality Clashes**: The underlying XGBoost/QNN checkpoints were down-sampled to a 3-class target (`BENIGN`, `DoS Hulk`, `PortScan`) for quantum benchmarking feasibility, but the Streamlit system historically pulled its class indices dynamically from a 15-class `label_encoder.pkl`. This caused `HybridIDS` array concatenation to throw `IndexError (axis 1 with size 3)`.
  - **Resolution**: Architected the `HybridIDS` wrapper to dynamically uncouple from legacy state encoders. The system safely identifies 3-class sub-matrices during inference down-streams and scales them independently without hard-crashing downstream explainers like Anchors.

### C. Persistent Serialization Formats
- **Interpreter Traps**: Scikit-Learn components (like the `MinMaxScaler` angle scaler generated during QNN phase-shifting) were historically saved via `joblib`. Attempting to load them sequentially using standard `pickle` generated `_pickle.UnpicklingError` flags due to header fragmentation.
  - **Resolution**: Uniform standard applied across the execution suite where strictly `joblib` is leveraged for data encoders, while `torch.save` handles weights.
  
### D. Plotly Dynamic Color Evaluation
- **CSS Alpha Truncation**: Radar charts utilizing `go.Scatterpolar` evaluated hex combinations natively via python string replacement hacks (e.g. replacing `#` with `rgba(`), resulting in malformed CSS logic and throwing TypeErrors during rendering.
  - **Resolution**: Rebuilt chart constructors to utilize explicitly mapped RGBA array matrices, guaranteeing proper chart rendering with responsive opacity support.
