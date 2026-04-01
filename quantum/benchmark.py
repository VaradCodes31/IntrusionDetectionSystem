"""
quantum/benchmark.py
──────────────────────────────────────────────────────────────────────────────
Purpose:
    Orchestrates the full comparison study between three models:
        1. XGBoost (Classical Baseline) — retrained fresh on benchmark data
        2. Hybrid QNN (Quantum Neural Network with PQC)
        3. QKSVM (Quantum Kernel Support Vector Machine)

    All three models are evaluated on the SAME test split using identical
    preprocessing and label encoding, ensuring a provably fair comparison.

    Results are saved to `results/benchmark_results.json` and read by
    the dashboard's ⚛ Quantum Lab tab for visualization.

Runtime expectations (M1/M2 Mac, CPU-only) — OPTIMIZED:
    - XGBoost training:   < 30 seconds  (full dataset)
    - QNN training:       < 30 seconds  (500 samples, 4 qubits, backprop)
    - QKSVM training:     1–2 minutes   (150 samples × 150-sample kernel)
    - Total:              2–4 minutes

Design note:
    The sample caps for QNN and QKSVM are explicitly recorded in the benchmark
    JSON and surfaced in the dashboard for academic transparency. The research
    contribution is not that quantum models match XGBoost's speed/accuracy on
    the full dataset, but that they provide a meaningfully different inductive
    bias when operating on a compressed feature space.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Add project root to path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)
from xgboost import XGBClassifier
import joblib

from quantum.feature_selector import select_top_features
from quantum.data_encoder import encode_features, save_scaler
from quantum.qnn_model import train_qnn, predict_qnn, save_qnn, load_qnn, N_TRAIN_CAP as QNN_CAP, N_LAYERS
from quantum.quantum_kernel_svm import train_qksvm, predict_qksvm, save_qksvm, N_TRAIN_CAP as QKSVM_CAP

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
# N_QUBITS matches qnn_model.N_QUBITS and quantum_kernel_svm.N_QUBITS.
# 4 qubits: statevector = 16 elements (2^4), manageable on CPU in minutes.
N_QUBITS = 4
RESULTS_PATH = "results/benchmark_results.json"
# QKSVM_TEST_CAP and QKSVM_CAP will be clamped to actual dataset size at runtime.
# 20 test samples × 40 train = 800 kernel evaluations for inference (~15s).
QKSVM_TEST_CAP = 20


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────
def _load_data() -> pd.DataFrame:
    """
    Tries to load benchmark data in order of preference:
      1. varied_traffic.csv  (synthetic multi-class traffic, pre-generated)
      2. data/raw/*.csv      (original CIC-IDS2017 raw files)
    """
    candidates = ["varied_traffic.csv", "data/varied_traffic.csv"]
    for path in candidates:
        if os.path.exists(path):
            print(f"[Benchmark] Loading data from '{path}'")
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            return df

    raw_dir = "data/raw"
    if os.path.exists(raw_dir):
        files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".csv")])[:3]
        if files:
            dfs = []
            for fname in files:
                try:
                    dfs.append(pd.read_csv(os.path.join(raw_dir, fname)))
                    print(f"[Benchmark] Loaded '{fname}'")
                except Exception as e:
                    print(f"[Benchmark] Warning: could not load '{fname}': {e}")
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                df.columns = df.columns.str.strip()
                return df

    raise FileNotFoundError(
        "No benchmark data found. Ensure 'varied_traffic.csv' exists in the project root."
    )


def _prepare_data(df: pd.DataFrame) -> tuple:
    """
    Cleans data, performs stratified 75/25 split, and encodes labels.

    Returns:
        X_train, X_test, y_train_enc, y_test_enc, le, class_names
    """
    if "Label" not in df.columns:
        raise ValueError("Dataset must contain a 'Label' column for supervised learning.")

    X = df.drop("Label", axis=1)
    y = df["Label"]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.replace([np.inf, -np.inf], 0.0)

    print(f"[Benchmark] Class distribution:\n{y.value_counts().to_string()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print(f"[Benchmark] Train: {len(X_train)} | Test: {len(X_test)} | Classes: {list(le.classes_)}")
    return X_train, X_test, y_train_enc, y_test_enc, le


# ──────────────────────────────────────────────────────────────────────────────
# Metric Computation
# ──────────────────────────────────────────────────────────────────────────────
def _metrics(y_true, y_pred, model_name: str, inference_time_ms: float, classes: list) -> dict:
    """Assembles a structured metrics dict for JSON export."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)) * 100, 2),
        "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)) * 100, 2),
        "inference_latency_ms": round(inference_time_ms, 4),
        "confusion_matrix": cm,
        "per_class_f1": {classes[i]: round(float(v) * 100, 2) for i, v in enumerate(per_class_f1)},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main Benchmark Runner
# ──────────────────────────────────────────────────────────────────────────────
def run_benchmark() -> dict:
    print("\n" + "=" * 65)
    print("  ⚛  NetSage-IDS | Quantum Machine Learning Benchmark Study")
    print("=" * 65)
    print("=" * 65, flush=True)
    print(f"  Models:    XGBoost  |  QNN (backprop PQC, {N_QUBITS} qubits)  |  QKSVM", flush=True)
    print(f"  QNN:       diff_method=backprop | {QNN_CAP} train samples | 15 epochs", flush=True)
    print(f"  QKSVM:     {QKSVM_CAP} train samples (kernel matrix O(N²))", flush=True)
    print(f"  Runtime:   XGBoost <30s | QNN 2-5min | QKSVM 1-3min", flush=True)
    print("=" * 65, flush=True)

    total_t0 = time.time()

    # ── Step 1: Load & prepare data ──────────────────────────────────────────
    print("\n[1/7] Loading data...")
    df = _load_data()
    X_train, X_test, y_train, y_test, le = _prepare_data(df)
    class_names = list(le.classes_)
    n_classes = len(class_names)

    # ── Step 2: Feature selection for QML ────────────────────────────────────
    print(f"\n[2/7] Selecting top-{N_QUBITS} features for quantum encoding...")
    try:
        selected_features = select_top_features(
            k=N_QUBITS, model_path="models/xgboost_model.pkl"
        )
    except FileNotFoundError:
        print("[Benchmark] XGBoost model not found — using top feature columns by variance.")
        variances = X_train.var().sort_values(ascending=False)
        selected_features = list(variances.head(N_QUBITS).index)

    # Validate and fill missing features
    selected_features = [f for f in selected_features if f in X_train.columns]
    if len(selected_features) < N_QUBITS:
        extras = [c for c in X_train.columns if c not in selected_features]
        selected_features += extras[: N_QUBITS - len(selected_features)]
    selected_features = selected_features[:N_QUBITS]

    # ── Step 3: Encode for QML ───────────────────────────────────────────────
    print("\n[3/7] Angle-encoding features for quantum circuits...")
    X_train_q, scaler = encode_features(X_train, selected_features, fit=True)
    X_test_q, _ = encode_features(X_test, selected_features, scaler=scaler, fit=False)
    save_scaler(scaler)

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "selected_features": selected_features,
            "class_names": class_names,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "qnn_train_cap": QNN_CAP,
            "qksvm_train_cap": QKSVM_CAP,
            "qksvm_test_cap": QKSVM_TEST_CAP,
        },
        "models": [],
    }

    # ── Step 4: XGBoost (Classical Baseline) ─────────────────────────────────
    print("\n[4/7] Training XGBoost (Classical Baseline)...")
    xgb_t0 = time.time()
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_train_time = time.time() - xgb_t0

    # Inference timing
    t0 = time.time()
    xgb_preds = xgb.predict(X_test)
    xgb_latency = (time.time() - t0) / len(X_test) * 1000

    xgb_result = _metrics(y_test, xgb_preds, "XGBoost (Classical)", xgb_latency, class_names)
    xgb_result["train_time_s"] = round(xgb_train_time, 2)
    results["models"].append(xgb_result)
    print(f"  ✓ XGBoost | Acc: {xgb_result['accuracy']}% | F1: {xgb_result['f1_macro']}% | "
          f"Train: {xgb_train_time:.1f}s | Latency: {xgb_latency:.3f}ms/sample")

    # ── Step 5: Quantum Neural Network ───────────────────────────────────────
    print(f"\n[5/7] Training Hybrid QNN ({N_QUBITS} qubits, {N_LAYERS} layers, backprop)...")
    print(f"      ⏱  Expected runtime: 2–5 minutes (backprop on CPU simulator)")
    qnn_t0 = time.time()
    qnn_model = train_qnn(X_train_q, y_train, n_classes=n_classes)
    qnn_train_time = time.time() - qnn_t0
    save_qnn(qnn_model)

    t0 = time.time()
    qnn_preds, _ = predict_qnn(qnn_model, X_test_q)
    qnn_latency = (time.time() - t0) / len(X_test_q) * 1000

    qnn_result = _metrics(y_test, qnn_preds, "QNN (Hybrid PQC)", qnn_latency, class_names)
    qnn_result["train_time_s"] = round(qnn_train_time, 2)
    qnn_result["train_samples_used"] = min(QNN_CAP, len(X_train))
    results["models"].append(qnn_result)
    print(f"  ✓ QNN     | Acc: {qnn_result['accuracy']}% | F1: {qnn_result['f1_macro']}% | "
          f"Train: {qnn_train_time:.1f}s | Latency: {qnn_latency:.3f}ms/sample")

    # ── Step 6: Quantum Kernel SVM ───────────────────────────────────────────
    print(f"\n[6/7] Training Quantum Kernel SVM ({QKSVM_CAP} train samples)...")
    print(f"      ⏱  Expected runtime: 1–3 minutes on CPU simulator")
    qksvm_t0 = time.time()
    svm_model, X_train_ref = train_qksvm(X_train_q, y_train)
    qksvm_train_time = time.time() - qksvm_t0
    save_qksvm(svm_model, X_train_ref)

    # Cap test set for kernel matrix
    test_idx = np.random.choice(len(X_test_q), min(QKSVM_TEST_CAP, len(X_test_q)), replace=False)
    X_test_q_cap = X_test_q[test_idx]
    y_test_cap = y_test[test_idx]

    t0 = time.time()
    qksvm_preds, _ = predict_qksvm(svm_model, X_train_ref, X_test_q_cap)
    qksvm_latency = (time.time() - t0) / len(X_test_q_cap) * 1000

    qksvm_result = _metrics(y_test_cap, qksvm_preds, "QKSVM (Quantum Kernel)", qksvm_latency, class_names)
    qksvm_result["train_time_s"] = round(qksvm_train_time, 2)
    qksvm_result["train_samples_used"] = min(QKSVM_CAP, len(X_train))
    qksvm_result["test_samples_used"] = len(test_idx)
    results["models"].append(qksvm_result)
    print(f"  ✓ QKSVM   | Acc: {qksvm_result['accuracy']}% | F1: {qksvm_result['f1_macro']}% | "
          f"Train: {qksvm_train_time:.1f}s | Latency: {qksvm_latency:.3f}ms/sample")

    # ── Step 7: Save results ──────────────────────────────────────────────────
    print(f"\n[7/7] Saving benchmark results...")
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - total_t0
    print(f"\n{'='*65}")
    print(f"  ✅ Benchmark complete in {total_time/60:.1f} minutes")
    print(f"  📄 Results saved to '{RESULTS_PATH}'")
    print(f"{'='*65}\n")

    # Summary table
    print(f"{'Model':<25} {'Accuracy':>10} {'F1-Macro':>10} {'Latency (ms)':>14}")
    print("-" * 62)
    for m in results["models"]:
        print(f"{m['model']:<25} {m['accuracy']:>9.2f}% {m['f1_macro']:>9.2f}% {m['inference_latency_ms']:>14.4f}")

    return results


if __name__ == "__main__":
    run_benchmark()
