# 🛡️ NetSage-IDS: Explainable Intrusion Detection System

An advanced **Machine Learning-based Intrusion Detection System** enhanced with **Explainable AI (XAI)** techniques to provide transparent, interpretable, and reliable cyber-attack detection.

---

## 🚀 Project Overview

Traditional IDS systems often act as **black boxes**, making it difficult to understand *why* a network activity is flagged as malicious. 

**NetSage-IDS** solves this problem by integrating a high-performance detection engine with a professional **Cyber Security Operations Center (CSOC)** terminal.

* 🔍 **Machine Learning (XGBoost)** for high-performance attack detection
* 🧠 **Explainable AI (SHAP)** for model interpretability
* 📊 **CSOC Dashboard (Streamlit)** for live monitoring and batch analysis

---

## 🆕 Recent Updates (v2.0)

*   **Hybrid Classical-Quantum Pipeline (HCQP)**: Successfully merged the PennyLane QNN pilot into a unified Scikit-learn compliant ensemble model alongside XGBoost.
*   **Multi-Layer Model-Agnostic XAI**: Integrated LIME, Anchors (Alibi), and Counterfactuals (DiCE) directly into the Hybrid prediction pipeline.
*   **macOS / M-Silicon Optimizations**: Resolved critical thread-locking OpenMP constraints causing segmentation faults between PyTorch and XGBoost in Streamlit.
*   **Dynamic Matrix Alignment**: Hardened the fusion engine to gracefully map subset probabilities (3-class specific trainings) against variable label encoders.
*   **Quantum Integrated Gradients**: Released Q-IG via the parameter-shift rule for evaluating quantum circuit decision trajectories.

---

## 🎯 Key Features

* 🚨 **Multi-class Attack Detection** (15 classes)
* 📡 **Live Traffic Monitor** with **Hybrid Consensus**
* ⚛️ **Quantum-Enhanced Detection** (Hybrid QNN branch)
* 📂 **Batch Audit Mode** (Bulk historical log analysis)
* 🧠 **Multi-Engine Explainability** (SHAP, LIME, Q-IG)
* 🧪 **Quantum Integrated Gradients (Q-IG)** for circuit attribution
* 🛡️ **Anomaly Fallback** & Risk Scoring

---

## 🧠 Novel Contributions

### 1️⃣ Attack Feature Profiling

* Identifies **top contributing features** for each attack type
* Generates **attack signatures using SHAP values**

Example:

* **DDoS →** High packet rates & flow intensity
* **PortScan →** SYN flags & probing behavior

---

### 2️⃣ Hybrid Classical-Quantum Pipeline (HCQP)

*   **Dual-Stream Inference**: Combines high-capacity XGBoost with a 4-qubit **Variational Quantum Circuit**.
*   **Consensus Scoring**: Weighted fusion (80/20) provides higher confidence in adversarial detection.

### 3️⃣ Quantum Integrated Gradients (Q-IG)

*   First-of-its-kind implementation of **differentiable quantum attribution** in an IDS.
*   Uses the **parameter-shift rule** to trace quantum state triggers back to specific network flags.

---

## 🗂️ Project Structure

```
IntrusionDetectionSystem/
│
├── main.py                     # Centralized Execution Pipeline
├── PROJECT_INFO.md             # Comprehensive Technical Documentation
│
├── data/
│   └── raw/                    # Raw CICIDS2017 CSV files
│
├── preprocessing/
│   ├── data_loader.py          # Data ingestion logic
│   ├── data_cleaning.py        # Log sanitization
│   └── feature_engineering.py   # Leakage-free feature preparation
│
├── models/
│   ├── train_xgboost.py        # XGBoost training logic
│   ├── model_utils.py          # Persistence helpers (Encoder/Model)
│   └── xgboost_model.pkl       # Pre-trained detection engine
│
├── explainability/
│   ├── shap_global.py          # Global feature importance
│   ├── shap_local.py           # Per-event breakdown
│   └── explanation_profiles.py # Attack signature profiling
│
├── evaluation/
│   ├── metrics.py              # Performance reporting
│   └── performance_report.py
│
├── experiments/
│   └── stability_test.py       # Robustness evaluation
│
├── dashboard/
│   └── app.py                  # NetSage-IDS CSOC Terminal
│
├── results/
│   └── plots/                  # Generated XAI visualizations
│
├── requirements.txt            # Project dependencies
└── README.md
```

---

## 📊 Dataset

* **CICIDS2017 Dataset**
* Contains real-world network traffic with labeled attacks

🔗 https://www.kaggle.com/datasets/naeem41/cicids2017-dataset

---

## ⚙️ Tech Stack

| Category         | Tools                    |
| ---------------- | ------------------------ |
| Machine Learning | XGBoost, Scikit-learn, PyTorch |
| Quantum Engine   | PennyLane (PQC / Hilbert Simulator) |
| Data Processing  | Pandas, NumPy            |
| Explainability   | SHAP, LIME, Integrated Gradients |
| Visualization    | Plotly, Matplotlib       |
| Dashboard        | Streamlit (CSOC Theme)   |

---

## 🧪 Model Performance

* ✅ Accuracy: **~99.88%**
* 🛡️ **Data Integrity**: Enforced leak-free encoding (fit only on training data)
* 📈 Strong performance across all 15 major attack categories

---

## 🖥️ Dashboard Features

* 📡 **Live Monitor**: Real-time packet simulation and instantaneous risk scoring.
* 📂 **Batch Audit**: Large-scale analysis of historical network captures via CSV upload.
* 🧠 **Forensic Explainability**: On-demand SHAP breakdowns for every flagged event.
* 📊 **Interactive Plotly Charts**: High-density visual insights into attack distributions.
* 🛠️ **System Diagnostics**: Health-monitoring for the detection engine and feature-space.

---

## ▶️ How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/IntrusionDetectionSystem.git
cd IntrusionDetectionSystem
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Central Pipeline (Train/Evaluate)

```bash
python main.py
```

### 5️⃣ Launch NetSage-IDS Terminal

```bash
streamlit run dashboard/app.py
```

---

## 🧠 Future Enhancements

* 🌐 Real-time network traffic integration
* 📡 Live monitoring dashboard
* ☁️ Cloud deployment

---

## 👨‍💻 Author

**Varad Alshi**
BTech Computer Science

---

## 📌 Conclusion

**NetSage-IDS** demonstrates how **Explainable AI can bridge the gap between model performance and human trust** in cybersecurity systems. By moving beyond black-box detection, it provides security analysts with the "why" behind every alert, making network defense more transparent, interpretable, and reliable.

---

## ⭐ If you found this useful, consider giving it a star!
