# 🛡️ Explainable Intrusion Detection System (IDS)

An advanced **Machine Learning-based Intrusion Detection System** enhanced with **Explainable AI (XAI)** techniques to provide transparent, interpretable, and reliable cyber-attack detection.

---

## 🚀 Project Overview

Traditional IDS systems often act as **black boxes**, making it difficult to understand *why* a network activity is flagged as malicious.

This project solves that problem by integrating:

* 🔍 **Machine Learning (XGBoost)** for high-performance attack detection
* 🧠 **Explainable AI (SHAP)** for model interpretability
* 📊 **Interactive Dashboard (Streamlit)** for visualization and analysis

---

## 🎯 Key Features

* 🚨 **Multi-class Attack Detection** (15 classes)
* 🧠 **Global Explainability (SHAP Summary)**
* 🔍 **Local Explainability (Per-sample explanation)**
* 🧬 **Attack Feature Profiling (Novel Contribution)**
* 🧪 **Explanation Stability Testing (Novel Contribution)**
* 🖥️ **Interactive Cybersecurity Dashboard**
* 📊 **Real-time Predictions & Visual Insights**

---

## 🧠 Novel Contributions

### 1️⃣ Attack Feature Profiling

* Identifies **top contributing features** for each attack type
* Generates **attack signatures using SHAP values**

Example:

* **DDoS →** High packet rates & flow intensity
* **PortScan →** SYN flags & probing behavior

---

### 2️⃣ Explanation Stability Testing

* Evaluates consistency of SHAP explanations under slight input changes
* Ensures **model reliability and robustness**

---

## 🗂️ Project Structure

```
IntrusionDetectionSystem/
│
├── data/
│   └── raw/
│
├── preprocessing/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│
├── models/
│   ├── train_xgboost.py
│   └── xgboost_model.pkl
│
├── explainability/
│   ├── shap_global.py
│   ├── shap_local.py
│   ├── explanation_profiles.py
│
├── evaluation/
│   ├── metrics.py
│   └── performance_report.py
│
├── experiments/
│   └── stability_test.py
│
├── dashboard/
│   └── app.py
│
├── results/
│   └── plots/
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* **CICIDS2017 Dataset**
* Contains real-world network traffic with labeled attacks

🔗 https://www.kaggle.com/datasets/naeem41/cicids2017-dataset

---

## ⚙️ Tech Stack

| Category         | Tools         |
| ---------------- | ------------- |
| Machine Learning | XGBoost       |
| Data Processing  | Pandas, NumPy |
| Explainability   | SHAP          |
| Visualization    | Matplotlib    |
| Dashboard        | Streamlit     |
| Evaluation       | Scikit-learn  |

---

## 🧪 Model Performance

* ✅ Accuracy: **~99.88%**
* ⚠️ Minor class imbalance impact on rare attacks
* 📈 Strong performance across major attack categories

---

## 🖥️ Dashboard Features

* 📂 Upload network traffic data (CSV)
* 🚨 Real-time attack detection
* 🧠 SHAP-based explanations
* 📊 Attack distribution visualization
* 🟢🔴 Color-coded alerts (Safe vs Threat)
* 📌 Insights and summary metrics

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
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the pipeline (optional)

```bash
python preprocessing/data_loader.py
```

### 5️⃣ Launch dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🧠 Future Enhancements

* 🌐 Real-time network traffic integration
* 📡 Live monitoring dashboard
* 🤖 Deep Learning / Hybrid models
* ⚛️ Quantum Machine Learning (experimental)
* ☁️ Cloud deployment

---

## 👨‍💻 Author

**Varad Alshi**
BTech Computer Science

---

## 📌 Conclusion

This project demonstrates how **Explainable AI can bridge the gap between model performance and human trust** in cybersecurity systems.

It transforms a traditional IDS into a:

> 🔐 **Transparent, Interpretable, and Reliable Security Solution**

---

## ⭐ If you found this useful, consider giving it a star!
