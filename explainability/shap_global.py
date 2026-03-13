import shap
import pandas as pd
import matplotlib.pyplot as plt


def shap_global_explanation(model, X_train):

    print("Running SHAP global explainability...")

    # Use a small sample (SHAP is expensive)
    sample = X_train.sample(2000, random_state=42)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(sample)

    print("Generating SHAP summary plot...")

    shap.summary_plot(shap_values, sample, show=False)

    plt.tight_layout()
    plt.savefig("results/plots/shap_global_summary.png")

    print("SHAP summary plot saved to results/plots/")