import shap
import matplotlib.pyplot as plt


def shap_local_explanation(model, X_test):

    print("Running SHAP local explanation...")

    # Pick one sample
    sample = X_test.sample(1, random_state=42)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(sample)

    # Get predicted class
    prediction = model.predict(sample)[0]

    print(f"Model prediction class: {prediction}")

    print("Generating SHAP force plot...")

    shap.force_plot(
        explainer.expected_value[prediction],
        shap_values[prediction][0],
        sample.iloc[0],
        matplotlib=True,
        show=False
    )

    plt.tight_layout()
    plt.savefig("results/plots/shap_local_force.png")

    print("Local SHAP explanation saved to results/plots/")