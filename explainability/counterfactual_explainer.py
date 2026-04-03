"""
explainability/counterfactual_explainer.py
──────────────────────────────────────────────────────────────────────────────
Purpose:
    Implements Counterfactual Explanations for "What-If?" forensic analysis
    in NetSage-IDS. For any blocked packet, the system shows which features
    need to change — and by how much — for it to be reclassified as BENIGN.

Theory:
    Counterfactuals answer: "What is the minimal change to this input
    that would flip the model's prediction?"

    Formally:
        x' = argmin ||x − x'||  such that  f(x') ≠ f(x)

    This is an optimization problem over the feature space. The solution x'
    is called the counterfactual: it is the closest point on the OTHER SIDE
    of the model's decision boundary.

    NetSage-IDS application:
        For a detected "PortScan" packet:
            Original:       Port=22, SYN_Count=84, Flow_Duration=0.1s  → PortScan
            Counterfactual: Port=443, SYN_Count=3, Flow_Duration=2.1s  → BENIGN

        This tells the analyst (and the Active Response module):
            "The model treats Port 22, high SYN counts, and short flow
            durations as the primary discriminators for this attack class."

    DiCE (Diverse Counterfactual Explanations):
        We use DiCE rather than a single nearest counterfactual because
        diverse counterfactuals reveal MULTIPLE paths to reclassification,
        helping analysts understand the true shape of the decision boundary,
        not just the closest escape route.

    Diversity constraint:
        DiCE enforces distance between generated counterfactuals using a
        determinantal point process (DPP), ensuring they explore different
        feature subspaces.

Dependencies:
    dice-ml >= 0.11 (pip install dice-ml)
    No TensorFlow required — uses the scikit-learn backend.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_dice_explainer(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_names: list,
    model,
    target_col: str = "_target_",
):
    """
    Creates and returns a DiCE explainer for the given model.

    Args:
        X_train:      Training feature DataFrame.
        y_train:      Encoded integer class labels for training data.
        feature_names: List of feature column names.
        model:        Trained classifier with a predict() method.
        target_col:   Internal name for the target column (not a feature).

    Returns:
        dice_explainer: A configured dice_ml.Dice object.
        feature_names:  Cleaned list of feature names (for later reference).
    """
    try:
        import dice_ml
    except ImportError:
        raise ImportError(
            "dice-ml is required for Counterfactual Explanations. "
            "Install with: pip install dice-ml>=0.11"
        )

    # Build the training dataframe with target column appended
    train_df = pd.DataFrame(X_train, columns=feature_names).copy()
    train_df[target_col] = y_train.astype(int)

    # DiCE Data object — describes the dataset structure
    data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=feature_names,
        outcome_name=target_col,
    )

    # DiCE Model wrapper — wraps the sklearn estimator
    class _SklearnWrapper:
        """Thin wrapper so DiCE can call model.predict() correctly."""
        def __init__(self, clf, feat_names, tgt):
            self.clf = clf
            self.feat_names = feat_names
            self.tgt = tgt

        def predict(self, X):
            if isinstance(X, pd.DataFrame):
                X = X[self.feat_names].values
            return self.clf.predict(X)

        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                X = X[self.feat_names].values
            if hasattr(self.clf, "predict_proba"):
                return self.clf.predict_proba(X)
            # Fallback if no predict_proba exists
            preds = self.predict(X)
            try:
                classes = self.clf.classes_
            except AttributeError:
                classes = np.unique(preds)
            probas = np.zeros((len(X), len(classes)))
            for i, p in enumerate(preds):
                idx = np.where(classes == p)[0]
                if len(idx) > 0:
                    probas[i, idx[0]] = 1.0
                else:
                    probas[i, 0] = 1.0
            return probas

    wrapped_model = dice_ml.Model(
        model=_SklearnWrapper(model, feature_names, target_col),
        backend="sklearn",
        model_type="classifier",
    )

    explainer = dice_ml.Dice(data, wrapped_model, method="genetic")
    return explainer, feature_names


def generate_counterfactuals(
    explainer,
    row: pd.Series,
    feature_names: list,
    desired_class: int = 0,
    n_cfs: int = 3,
    target_col: str = "_target_",
    method: str = None,
) -> dict:
    """
    Generates N diverse counterfactual examples for a single input packet.

    Args:
        explainer:      DiCE explainer (from create_dice_explainer).
        row:            pandas Series or 1D array of the input sample.
        feature_names:  Feature column names.
        desired_class:  Target class for the counterfactual (default 0 = BENIGN).
        n_cfs:          Number of diverse counterfactuals to generate.
        target_col:     Name of the target column (must match create_dice_explainer).

    Returns:
        Dict with 'original', 'counterfactuals' (list of dicts), and
        'changed_features' (list of feature names that differ).
    """
    try:
        import dice_ml
    except ImportError:
        raise ImportError("dice-ml is required. Install with: pip install dice-ml>=0.11")

    # Build input DataFrame
    if isinstance(row, pd.Series):
        row_values = row[feature_names].values
    elif isinstance(row, np.ndarray):
        row_values = row
    else:
        row_values = np.array(row)

    input_df = pd.DataFrame([row_values], columns=feature_names)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # If a specific method is requested for this instance, we honor it
            # Genetic is generally more robust for tabular data than random/kd-tree
            cf_result = explainer.generate_counterfactuals(
                input_df,
                total_CFs=n_cfs,
                desired_class=desired_class,
                features_to_vary="all",
                verbose=False,
            )
        except Exception as e:
            return {
                "original": dict(zip(feature_names, row_values)),
                "counterfactuals": [],
                "changed_features": [],
                "error": str(e),
            }

    try:
        cf_df = cf_result.cf_examples_list[0].final_cfs_df
        if cf_df is None or len(cf_df) == 0:
            raise ValueError("No counterfactuals generated")
        # Drop target column if present
        if target_col in cf_df.columns:
            cf_df = cf_df.drop(columns=[target_col])
    except Exception as e:
        return {
            "original": dict(zip(feature_names, row_values)),
            "counterfactuals": [],
            "changed_features": [],
            "error": str(e),
        }

    original_dict = dict(zip(feature_names, row_values))
    counterfactuals = []
    all_changed = set()

    for _, cf_row in cf_df.iterrows():
        cf_dict = {}
        changed = []
        for feat in feature_names:
            orig_val = original_dict.get(feat, 0.0)
            cf_val = float(cf_row.get(feat, orig_val))
            changed_flag = not np.isclose(orig_val, cf_val, rtol=1e-3, atol=1e-5)
            cf_dict[feat] = {
                "original": round(float(orig_val), 4),
                "counterfactual": round(cf_val, 4),
                "changed": changed_flag,
            }
            if changed_flag:
                changed.append(feat)
                all_changed.add(feat)
        counterfactuals.append({"features": cf_dict, "changed": changed})

    return {
        "original": original_dict,
        "counterfactuals": counterfactuals,
        "changed_features": sorted(all_changed),
    }


def build_cf_comparison_table(cf_result: dict) -> pd.DataFrame:
    """
    Builds a clean comparison DataFrame showing original vs counterfactual values
    for only the features that changed across at least one counterfactual.

    Args:
        cf_result: Output of generate_counterfactuals().

    Returns:
        DataFrame with columns: Feature | Original | CF-1 | CF-2 | CF-3 ...
    """
    changed_feats = cf_result.get("changed_features", [])
    if not changed_feats or not cf_result.get("counterfactuals"):
        return pd.DataFrame({"Info": ["No counterfactuals were generated for this instance."]})

    original = cf_result["original"]
    cfs = cf_result["counterfactuals"]

    rows = []
    for feat in changed_feats:
        row = {
            "Feature": feat,
            "Original Value": round(float(original.get(feat, 0)), 4),
        }
        for i, cf in enumerate(cfs, start=1):
            cf_val = cf["features"].get(feat, {}).get("counterfactual", original.get(feat, 0))
            row[f"CF-{i}"] = round(float(cf_val), 4)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_cf_comparison(
    cf_result: dict,
    predicted_class: str,
    desired_class_name: str = "BENIGN",
    save_path: str = None,
) -> plt.Figure:
    """
    Renders a visual comparison of original vs counterfactual values
    for the changed features, in the CSOC aesthetic.
    """
    df = build_cf_comparison_table(cf_result)
    if len(df) == 0 or "Info" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 2))
        fig.patch.set_facecolor("#0a0b10")
        ax.set_facecolor("#0a0b10")
        ax.text(0.5, 0.5, "No counterfactuals generated.", ha="center", va="center",
                color="#888888", fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        return fig

    cf_cols = [c for c in df.columns if c.startswith("CF-")]

    # Filter to top 12 changed features by max difference to keep the chart readable
    diffs = []
    orig_vals_all = df["Original Value"].values
    
    for idx, row in df.iterrows():
        orig = row["Original Value"]
        # Handle cases where CF columns might not be present or valid
        if not cf_cols:
            diffs.append(0)
            continue
        max_d = max(abs(row[c] - orig) for c in cf_cols)
        diffs.append(max_d)

    df["max_diff"] = diffs
    df = df.sort_values("max_diff", ascending=False).head(12)

    changed_feats = list(df["Feature"])
    orig_vals = list(df["Original Value"])

    x = np.arange(len(changed_feats))
    width = 0.8 / (1 + len(cf_cols))

    fig, ax = plt.subplots(figsize=(max(8, len(changed_feats) * 0.8), 5))
    fig.patch.set_facecolor("#0a0b10")
    ax.set_facecolor("#0a0b10")

    # Normalize for visual comparison
    all_vals = orig_vals + [v for col in cf_cols for v in df[col].tolist()]
    vmax = max(abs(v) for v in all_vals) or 1.0

    ax.bar(x - width * len(cf_cols) / 2, [v / vmax for v in orig_vals],
           width=width, label=f"Original ({predicted_class})",
           color="#ff0055", alpha=0.85, edgecolor="none")

    cf_colors = ["#00f2ff", "#7000ff", "#00ff88"]
    for i, col in enumerate(cf_cols):
        cf_vals = list(df[col])
        ax.bar(x - width * len(cf_cols) / 2 + width * (i + 1), [v / vmax for v in cf_vals],
               width=width, label=f"{col} (→ {desired_class_name})",
               color=cf_colors[i % len(cf_colors)], alpha=0.85, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(changed_feats, rotation=45, ha="right", color="#c0c0c0", fontsize=9)
    ax.set_ylabel("Normalized Value", color="#00f2ff", fontsize=10)
    ax.set_title(
        f"Counterfactual: Top Changed Features ({predicted_class} → {desired_class_name})",
        color="#00f2ff", fontsize=11, fontweight="bold", pad=10,
    )
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#0a0b10", labelcolor="white", fontsize=8, framealpha=0.8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig
