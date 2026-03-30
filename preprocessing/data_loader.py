import os
import sys
import pandas as pd

# Project Root Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import evaluate_model
from explainability.shap_global import shap_global_explanation
from explainability.shap_local import shap_local_explanation
from explainability.explanation_profiles import attack_feature_profiling
from experiments.stability_test import explanation_stability

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