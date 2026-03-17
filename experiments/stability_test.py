import shap
import numpy as np


def explanation_stability(model, X_test):

    print("Running explanation stability test...")

    # Take one sample
    sample = X_test.sample(1, random_state=42)

    # Create small noise
    noise = np.random.normal(0, 0.01, sample.shape)

    perturbed_sample = sample + noise

    explainer = shap.TreeExplainer(model)

    shap_original = explainer.shap_values(sample)
    shap_perturbed = explainer.shap_values(perturbed_sample)

    # Handle SHAP format
    if isinstance(shap_original, list):
        shap_original = shap_original[0][0]
        shap_perturbed = shap_perturbed[0][0]
    else:
        shap_original = shap_original[0]
        shap_perturbed = shap_perturbed[0]

    # Compute difference
    stability_score = np.mean(np.abs(shap_original - shap_perturbed))

    print(f"Stability Score: {stability_score}")

    return stability_score