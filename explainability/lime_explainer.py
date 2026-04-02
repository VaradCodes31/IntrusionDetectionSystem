"""
explainability/lime_explainer.py
──────────────────────────────────────────────────────────────────────────────
Purpose:
    Implements LIME (Local Interpretable Model-agnostic Explanations) for
    fast, per-packet, local explanations in NetSage-IDS.

Theory:
    LIME (Ribeiro et al., 2016) addresses the interpretability gap by building
    a locally-faithful linear surrogate model around any specific prediction:

        ξ(x) = argmin_{g ∈ G} L(f, g, πₓ) + Ω(g)

    Where:
        f   = the black-box model (XGBoost)
        g   = a locally-fitted linear model (the explanation)
        πₓ  = a proximity measure (Gaussian kernel around input x)
        Ω(g) = complexity penalty (number of non-zero coefficients)

    Algorithm for tabular data:
        1. Take the input sample x (one network packet row)
        2. Generate N perturbed neighbors by sampling from the feature
           distribution learned from the training set
        3. Weight each neighbor by its distance from x (πₓ)
        4. Query XGBoost for predictions on all neighbors
        5. Fit a weighted Lasso regression in this local neighborhood
        6. The Lasso coefficients become the explanation: positive = pushed
           toward attack class, negative = pushed toward benign

    Key difference from SHAP:
        LIME approximates locally; it is NOT guaranteed to be consistent
        (i.e., two similar inputs can get very different LIME explanations).
        It trades consistency for speed: LIME is model-agnostic and fast,
        making it suitable for real-time per-packet explanation in the dashboard.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lime import lime_tabular


# ──────────────────────────────────────────────────────────────────────────────
# Explainer Factory
# ──────────────────────────────────────────────────────────────────────────────
def create_lime_explainer(
    X_train: pd.DataFrame,
    class_names: list,
) -> lime_tabular.LimeTabularExplainer:
    """
    Creates and returns a LimeTabularExplainer fitted on the training
    distribution. This explainer is then reused across multiple explain_instance
    calls without re-fitting.

    Args:
        X_train:     Training feature DataFrame (used to learn feature distributions).
        class_names: List of class label strings (e.g. ['BENIGN', 'DoS Hulk', 'PortScan']).

    Returns:
        A fitted LimeTabularExplainer instance.
    """
    return lime_tabular.LimeTabularExplainer(
        training_data=X_train.values.astype(np.float64),
        feature_names=list(X_train.columns),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Explain Single Instance
# ──────────────────────────────────────────────────────────────────────────────
def lime_explain_instance(
    explainer: lime_tabular.LimeTabularExplainer,
    row: np.ndarray,
    model_predict_fn,
    predicted_class_idx: int,
    top_features: int = 10,
    num_samples: int = 500,
) -> dict:
    """
    Generates a LIME explanation for a single network packet.

    Args:
        explainer:          The LimeTabularExplainer (from create_lime_explainer).
        row:                1D numpy array of feature values for the packet.
        model_predict_fn:   A function that takes a 2D array and returns
                            probabilities — typically model.predict_proba.
        predicted_class_idx: The predicted class index to explain.
        top_features:       Number of features to include in the explanation.
        num_samples:        Synthetic neighbors to generate (higher = more stable).

    Returns:
        Dict with 'features' (list of (name, weight) tuples),
        'predicted_label' index, and 'local_score' of the explanation.
    """
    exp = explainer.explain_instance(
        data_row=row.astype(np.float64),
        predict_fn=model_predict_fn,
        labels=(predicted_class_idx,),
        num_features=top_features,
        num_samples=num_samples,
    )

    feature_weights = exp.as_list(label=predicted_class_idx)
    # LIME's local_pred indices match the 'labels' parameter provided to explain_instance.
    # Since we only passed (predicted_class_idx,), local_pred has size 1.
    local_score = exp.local_pred[0] if hasattr(exp, "local_pred") and len(exp.local_pred) > 0 else None
    score_str = exp.score if hasattr(exp, "score") else None

    return {
        "features": feature_weights,   # [(feature_condition_str, weight), ...]
        "predicted_class_idx": predicted_class_idx,
        "local_prediction_score": float(local_score) if local_score is not None else None,
        "surrogate_r2": float(score_str) if score_str is not None else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────
def plot_lime_explanation(
    lime_result: dict,
    class_name: str,
    save_path: str = None,
) -> plt.Figure:
    """
    Renders the LIME explanation as a horizontal bar chart in the CSOC aesthetic.

    Colors: green bars = features pushing toward predicted class,
            red bars   = features pushing away from it.

    Args:
        lime_result:  Output of lime_explain_instance().
        class_name:   Human-readable predicted class label.
        save_path:    If provided, saves the figure to this path.

    Returns:
        matplotlib Figure object (for Streamlit's st.pyplot()).
    """
    features = lime_result["features"]
    if not features:
        return None

    names = [f[0] for f in features]
    weights = [f[1] for f in features]
    colors = ["#00ff88" if w >= 0 else "#ff0055" for w in weights]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    fig.patch.set_facecolor("#0a0b10")
    ax.set_facecolor("#0a0b10")

    bars = ax.barh(range(len(names)), weights, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color="#c0c0c0", fontsize=9)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("LIME Feature Weight", color="#00f2ff", fontsize=10)
    ax.set_title(
        f"LIME Explanation  →  Predicted: {class_name}",
        color="#00f2ff", fontsize=12, fontweight="bold", pad=12,
    )
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    green_patch = mpatches.Patch(color="#00ff88", label="Promotes this class")
    red_patch = mpatches.Patch(color="#ff0055", label="Suppresses this class")
    ax.legend(handles=[green_patch, red_patch], facecolor="#0a0b10",
              labelcolor="white", fontsize=8, loc="lower right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig
