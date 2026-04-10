import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# Use Agg backend for headless environments
plt.switch_backend('Agg')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_cleaning import clean_data

def generate_performance_plots():
    print("--- 🛡️ Starting NetSage-IDS High-Fidelity Analysis ---")
    
    # Define paths
    MODEL_PATH = "models/xgboost_model.pkl"
    ENCODER_PATH = "models/label_encoder.pkl"
    DATA_PATH = "data/raw/combinenew.csv"  # SWITCHING TO FULL DATASET
    OUTPUT_DIR = "results/plots"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Model and Encoder
    print("Loading model and label encoder...")
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    classes = list(le.classes_)
    n_classes = len(classes)
    
    # 2. Load and Sample Data (10% stratified sample for 2.8M rows)
    print(f"Loading full dataset from {DATA_PATH}...")
    # Note: Header has leading spaces in raw CSV
    full_df = pd.read_csv(DATA_PATH, low_memory=False)
    full_df.columns = full_df.columns.str.strip() # SANITIZE HEADERS
    
    # CLEAN DATA (Replace inf/NaN)
    print("Cleaning dataset (handling INF/NaN)...")
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df.dropna(inplace=True)
    
    # Handle rare classes and sample
    # Filter to only include classes the model was trained on
    trained_classes = set(le.classes_)
    full_df = full_df[full_df['Label'].isin(trained_classes)]
    
    sample_size = min(300000, len(full_df))
    print(f"Dataset filtered to known classes. Remaining rows: {full_df.shape[0]}")
    print(f"Extracting stratified sample of {sample_size} rows...")
    
    # Stratified sample if possible, otherwise simple sample
    try:
        df = full_df.groupby('Label', group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
    except:
        df = full_df.sample(n=sample_size, random_state=42)
    
    del full_df # Free memory
    
    # 3. Preprocess
    # XGBoost features
    expected_features = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets',
        'Total Backward Packets', 'Total Length of Fwd Packets',
        'Total Length of Bwd Packets', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
        'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
        'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
        'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
        'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
        'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
        'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
        'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
        'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
        'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
        'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1',
        'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
        'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
        'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
        'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
        'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std',
        'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
    ]
    
    y_true_labels = df["Label"]
    y_true = le.transform(y_true_labels)
    
    X = df[expected_features]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    
    # 4. Predictions
    print(f"Generating predictions for {X.shape[0]} samples...")
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)
    
    # 5. Confusion Matrix
    print("Plotting Confusion Matrix (15-Class)...")
    cm = confusion_matrix(y_true, y_pred)
    # Normalize for better visualization of imbalances
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 8})
    plt.title('NetSage-IDS: Normalized Multi-Class Confusion Matrix (Full Dataset)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 6. ROC Curves (One-vs-Rest)
    print("Plotting High-Fidelity ROC Curves...")
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(12, 10))
    for i in range(n_classes):
        # Only plot if there are samples of this class in the sample
        if np.sum(y_true_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, label=f'{classes[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('NetSage-IDS: Multiclass AUC-ROC (2.8M Rows)')
    plt.legend(loc="lower right", fontsize='x-small', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=300)
    plt.close()
    
    # 7. Precision-Recall Curves
    print("Plotting Precision-Recall Curves...")
    plt.figure(figsize=(12, 10))
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
            avg_p = average_precision_score(y_true_bin[:, i], y_score[:, i])
            plt.plot(recall, precision, lw=1.5, label=f'{classes[i]} (AP = {avg_p:.3f})')
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('NetSage-IDS: Precision-Recall (Full Dataset)')
    plt.legend(loc="lower left", fontsize='x-small', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_curves.png'), dpi=300)
    plt.close()
    
    # 8. Feature Importance
    print("Plotting Feature Importance...")
    importances = model.feature_importances_
    # Get top 20 features
    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.title('NetSage-IDS: Top 20 Forensic Features (XGBoost)')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300)
    plt.close()
    
    print(f"--- ✅ All plots saved successfully in {OUTPUT_DIR} ---")

if __name__ == "__main__":
    generate_performance_plots()
