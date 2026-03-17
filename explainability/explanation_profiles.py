import shap
import pandas as pd


def attack_feature_profiling(model, X_train, y_train, label_encoder):

    print("Running attack-wise feature profiling...")

    # Sample for SHAP
    sample = X_train.sample(2000, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    feature_names = sample.columns

    profiles = {}

    for class_idx in range(len(label_encoder.classes_)):

        class_name = label_encoder.inverse_transform([class_idx])[0]

        # Handle SHAP format
        if isinstance(shap_values, list):
            class_shap = shap_values[class_idx]
        else:
            # shape: (samples, features, classes)
            class_shap = shap_values[:, :, class_idx]

        # Mean absolute importance
        mean_shap = abs(class_shap).mean(axis=0)

        top_features = pd.Series(mean_shap, index=feature_names)\
                        .sort_values(ascending=False)\
                        .head(5)

        profiles[class_name] = list(top_features.index)

        print(f"\n{class_name} Top Features:")
        print(top_features)

    print("\nAttack profiling completed.")
    return profiles