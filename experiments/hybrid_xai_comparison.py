import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from lime import lime_tabular

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum.hybrid_ensemble import HybridIDS
from quantum.qml_xai import calculate_quantum_ig

def run_comparative_study(num_samples=10):
    print(f"--- HCD Comparative XAI Study (n={num_samples}) ---")
    
    # 1. Load Data & Model
    df = pd.read_csv('varied_traffic.csv')
    overlap_df = df[df['Research_Cluster'] == 'OVERLAP'].sample(num_samples, random_state=42)
    X = overlap_df.drop(['Label', 'Research_Cluster'], axis=1)
    
    hybrid = HybridIDS()
    
    # 2. Setup LIME (Classical Side) - Works on FULL feature space
    all_features = list(X.columns)
    q_features = hybrid.q_features
    
    explainer_lime = lime_tabular.LimeTabularExplainer(
        X.values,
        feature_names=all_features,
        class_names=list(hybrid.le.classes_),
        mode='classification'
    )
    
    correlations = []
    
    for i in range(num_samples):
        sample = X.iloc[i]
        
        # A. Classical Ranking (LIME on full 78 features)
        exp = explainer_lime.explain_instance(sample.values, hybrid.classical_model.predict_proba, num_features=78)
        full_lime_map = {name: abs(val) for name, val in exp.as_list()}
        classical_rank = [full_lime_map.get(f, 0) for f in q_features]
        
        # B. Quantum Ranking (Q-IG on 4 features)
        sample_q = pd.DataFrame([sample[q_features]])
        
        # Get the 3-class index for the Quantum model's attribution
        true_label = overlap_df.iloc[i]['Label']
        q_target_idx = hybrid.qnn_classes.index(true_label)
        
        q_ig = calculate_quantum_ig(hybrid.qnn, torch.tensor(hybrid.scaler.transform(sample_q), dtype=torch.float32), target_class_idx=q_target_idx)
        quantum_rank = np.abs(q_ig).flatten()
        
        # C. Correlation
        corr, _ = spearmanr(classical_rank, quantum_rank)
        correlations.append(corr)
        
        print(f"Sample {i+1}: Spearman Correlation = {corr:.4f}")

    avg_corr = np.nanmean(correlations)
    print(f"\nAverage Cross-Model Logic Alignment: {avg_corr:.4f}")
    return avg_corr

if __name__ == "__main__":
    run_comparative_study(5)
