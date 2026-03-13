import os
import sys
import pandas as pd

# Project Root Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import evaluate_model
from explainability.shap_global import shap_global_explanation
from explainability.shap_local import shap_local_explanation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_PATH = "data/raw"

def load_dataset():
    csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/raw/")
    
    df_list = []
    
    for file in csv_files:
        file_path = os.path.join(DATA_PATH, file)
        print(f"Loading {file}...")
        df = pd.read_csv(file_path)
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.columns = combined_df.columns.str.strip()

    print("All files loaded successfully.")
    return combined_df


from data_cleaning import clean_data
from feature_engineering import prepare_features
from models.train_xgboost import train_model, save_model

if __name__ == "__main__":
    df = load_dataset()

    df = clean_data(df)

    X_train, X_test, y_train, y_test, le = prepare_features(df)

    model = train_model(X_train, y_train)

    save_model(model)

    evaluate_model(model, X_test, y_test)
    shap_global_explanation(model, X_train)
    shap_local_explanation(model, X_test)