import os
import sys
import pandas as pd

# Project Root Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.data_loader import load_dataset
from preprocessing.data_cleaning import clean_data
from preprocessing.feature_engineering import prepare_features
from models.train_xgboost import train_model, save_model
from models.model_utils import save_object
from evaluation.metrics import evaluate_model
from explainability.shap_global import shap_global_explanation
from explainability.shap_local import shap_local_explanation
from explainability.explanation_profiles import attack_feature_profiling
from experiments.stability_test import explanation_stability

def main():
    print("--- 🛡️ Intrusion Detection System: Pipeline Start ---")
    
    # 1. Load Data
    try:
        df = load_dataset()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Clean Data
    print("Cleaning data...")
    df = clean_data(df)

    # 3. Feature Engineering & Train-Test Split (fixed for leakage)
    X_train, X_test, y_train, y_test, le = prepare_features(df)

    # 4. Save LabelEncoder for Dashboard Use
    save_object(le, "label_encoder.pkl")

    # 5. Model Training
    model = train_model(X_train, y_train)

    # 6. Save Model
    save_model(model)

    # 7. Evaluation & Explainability
    print("Running evaluation and explainability reports...")
    evaluate_model(model, X_test, y_test)
    
    # Optional: these might take time depending on dataset size
    # shap_global_explanation(model, X_train)
    # shap_local_explanation(model, X_test)
    # attack_feature_profiling(model, X_train, y_train, le)
    # explanation_stability(model, X_test)

    print("--- ✅ Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()
