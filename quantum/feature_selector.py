"""
quantum/feature_selector.py
──────────────────────────────────────────────────────────────────────────────
Purpose:
    Selects the top-K most important features from the trained XGBoost model
    based on its built-in feature_importances_. These K features are then
    angle-encoded into K qubits for the QML pipeline.

Theory:
    Encoding all 70+ features into qubits is not computationally feasible on
    a classical simulator. By selecting the most informative features (as ranked
    by XGBoost's gradient-based feature importance), we compress the input to
    the highest-signal dimensions while keeping the qubit count manageable.

Default: K=8 (8 qubits). Outputs selected features to a JSON file for
consistency across training, benchmarking, and dashboard inference.
"""

import json
import os
import joblib
import numpy as np


def select_top_features(
    k: int = 4,
    model_path: str = "models/xgboost_model.pkl",
    output_path: str = "quantum/selected_features.json",
) -> list:
    """
    Loads the trained XGBoost model and selects the top-K features by
    feature importance (weight). Saves the selection to a JSON file.

    Args:
        k:           Number of features (qubits) to select.
        model_path:  Path to the saved XGBoost model.
        output_path: Where to write the selection JSON.

    Returns:
        List of selected feature name strings.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"XGBoost model not found at '{model_path}'. "
            "Run main.py first to train and save the model."
        )

    model = joblib.load(model_path)
    feature_names = model.get_booster().feature_names
    importances = model.feature_importances_

    # Sort descending by importance
    sorted_idx = np.argsort(importances)[::-1]
    top_k_idx = sorted_idx[:k]
    top_k_features = [feature_names[i] for i in top_k_idx]
    top_k_importances = [float(importances[i]) for i in top_k_idx]

    # Persist selection so all downstream modules use the same K features
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    record = {
        "k": k,
        "features": top_k_features,
        "importances": top_k_importances,
    }
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n[QML] Top-{k} features selected for quantum encoding:")
    for feat, imp in zip(top_k_features, top_k_importances):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<45} {imp:.4f}  {bar}")

    return top_k_features


def load_selected_features(path: str = "quantum/selected_features.json") -> list:
    """
    Loads the previously saved feature selection from JSON.

    Args:
        path: Path to the JSON file written by select_top_features().

    Returns:
        List of feature name strings.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No feature selection found at '{path}'. "
            "Call select_top_features() first."
        )
    with open(path, "r") as f:
        data = json.load(f)
    return data["features"]
