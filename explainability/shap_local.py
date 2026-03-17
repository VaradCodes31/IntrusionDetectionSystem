import shap
import matplotlib.pyplot as plt
import numpy as np


def shap_local_explanation(model, X_test):

    print("Running SHAP local explanation...")

    # Pick one sample
    sample = X_test.sample(1, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Get prediction
    prediction = model.predict(sample)[0]
    print(f"Model prediction class: {prediction}")

    print("Generating SHAP force plot...")

    # 🔥 Handle both SHAP formats
    if isinstance(shap_values, list):
        sv = shap_values[prediction][0]
        base_val = explainer.expected_value[prediction]
    else:
        # shape: (1, features, classes)
        sv = shap_values[0][:, prediction]
        base_val = explainer.expected_value[prediction]

    shap.force_plot(
        base_val,
        sv,
        sample.iloc[0],
        matplotlib=True,
        show=False
    )

    plt.tight_layout()
    plt.savefig("results/plots/shap_local_force.png")

    print("Local SHAP explanation saved to results/plots/")