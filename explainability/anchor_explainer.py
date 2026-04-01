"""
explainability/anchor_explainer.py
──────────────────────────────────────────────────────────────────────────────
Purpose:
    Implements Anchor explanations for rule-based, human-readable
    "if-then" justifications of intrusion detection decisions.

Theory:
    Anchors (Ribeiro et al., 2018) change the XAI question from
    "how much did each feature contribute?" (SHAP/LIME) to:
    "What is the minimal set of conditions that LOCKS IN this prediction?"

    Formally, an anchor A is a rule such that:

        E[f(z) = f(x) | z satisfies A] ≥ τ

    Where:
        τ   = precision threshold (default: 0.95)
        z   ~ Distribution of samples satisfying rule A
        f(x) = the model's prediction on input x

    Properties:
        - Precision: fraction of rule-satisfying samples that get the same class
        - Coverage:  fraction of feature space where the rule is applicable
        - Anchors are "if and only if" style: they state sufficient conditions

    Example output for a PortScan detection:
        IF  Destination_Port ≤ 1024
        AND SYN_Flag_Count > 50
        AND Flow_Duration < 0.5s
        THEN → PortScan  (Precision: 0.97, Coverage: 0.23)

    This is fundamentally different from SHAP/LIME:
        - SHAP gives numerical attribution weights
        - LIME gives a local linear model
        - Anchors give a logical rule that a non-technical analyst can read

    Best use case in NetSage-IDS:
        Populate the Neutralization Log with the anchor rule when a packet
        is blocked — so the analyst knows the exact logical reason, not just
        a score.

Dependencies:
    alibi (pip install alibi>=0.9.0)
    Alibi's AnchorTabular is a battle-tested implementation of the
    original KDD 2018 paper.
"""

import os
import numpy as np
import pandas as pd


def create_anchor_explainer(
    X_train: np.ndarray,
    feature_names: list,
    class_names: list,
    model_predict_fn,
    categorical_names: dict = None,
) -> object:
    """
    Creates and fits an AnchorTabular explainer on the training distribution.

    The explainer learns the feature distributions and discretization
    boundaries from X_train. These are later used to generate anchor rules
    that respect realistic thresholds (e.g., "Port ≤ 1024" not "Port ≤ 1024.37").

    Args:
        X_train:          Training feature array (n_samples, n_features).
        feature_names:    List of feature name strings.
        class_names:      List of class label strings.
        model_predict_fn: Function returning class indices: model.predict(X) → array.
        categorical_names: Optional dict mapping feature index → list of category names.

    Returns:
        A fitted alibi AnchorTabular explainer.
    """
    try:
        from alibi.explainers import AnchorTabular
    except ImportError:
        raise ImportError(
            "alibi is required for Anchor explanations. "
            "Install with: pip install alibi>=0.9.0"
        )

    explainer = AnchorTabular(
        predictor=model_predict_fn,
        feature_names=feature_names,
        categorical_names=categorical_names or {},
        seed=42,
    )

    print("[Anchors] Fitting explainer on training data...")
    explainer.fit(X_train, disc_perc=(25, 50, 75))
    print("[Anchors] Explainer ready.")
    return explainer


def anchor_explain_instance(
    explainer,
    row: np.ndarray,
    threshold: float = 0.95,
    max_anchor_size: int = 5,
) -> dict:
    """
    Generates an anchor rule for a single network packet prediction.

    Args:
        explainer:       Fitted AnchorTabular explainer.
        row:             1D numpy array of feature values.
        threshold:       Required precision (default: 0.95 = 95% confident rule).
        max_anchor_size: Maximum number of conditions in the rule.

    Returns:
        Dict with:
            'rule':      Human-readable rule string
            'precision': Probability the rule correctly predicts the class
            'coverage':  Fraction of dataset where this rule applies
            'features':  List of individual condition strings
    """
    exp = explainer.explain(
        row,
        threshold=threshold,
        max_anchor_size=max_anchor_size,
        coverage_samples=1000,
        beam_size=1,
        stop_on_first=True,
        verbose=False,
    )

    conditions = list(exp.anchor)
    precision = float(exp.precision) if hasattr(exp, "precision") else 0.0
    coverage = float(exp.coverage) if hasattr(exp, "coverage") else 0.0

    if conditions:
        rule_str = "\n    AND ".join(conditions)
        rule_str = f"IF  {rule_str}"
    else:
        rule_str = "No anchor found (model decision boundary too complex locally)"

    return {
        "rule": rule_str,
        "conditions": conditions,
        "precision": round(precision * 100, 1),
        "coverage": round(coverage * 100, 1),
    }


def format_anchor_for_display(anchor_result: dict, predicted_class: str) -> str:
    """
    Formats the anchor result as a styled text block for Streamlit display.

    Args:
        anchor_result:   Output of anchor_explain_instance().
        predicted_class: Human-readable prediction label.

    Returns:
        Multi-line string suitable for st.code() or st.text().
    """
    conds = anchor_result["conditions"]
    precision = anchor_result["precision"]
    coverage = anchor_result["coverage"]

    lines = []
    lines.append("━" * 55)
    lines.append("  DECISION RULE (Anchor Explanation)")
    lines.append("━" * 55)
    if conds:
        lines.append(f"  IF   {conds[0]}")
        for c in conds[1:]:
            lines.append(f"  AND  {c}")
        lines.append(f"  ──────────────────────────────────")
        lines.append(f"  THEN → {predicted_class.upper()}")
    else:
        lines.append("  No deterministic anchor found for this instance.")
    lines.append("")
    lines.append(f"  Precision: {precision}%   (certainty of this rule)")
    lines.append(f"  Coverage:  {coverage}%   (% of traffic space this covers)")
    lines.append("━" * 55)
    return "\n".join(lines)
