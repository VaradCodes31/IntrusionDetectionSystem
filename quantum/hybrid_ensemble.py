"""
quantum/hybrid_ensemble.py
──────────────────────────────────────────────────────────────────────────────
Purpose:
    Implements the Hybrid Classical-Quantum Pipeline (HCQP). 
    Fuses predictions from the baseline XGBoost and the research Quantum 
    Neural Network (QNN) according to the consensus strategy.

Design:
    The HybridIDS class follows the Scikit-learn estimator API (predict/predict_proba)
    to ensure seamless integration with LIME, Anchors, and Counterfactuals.

Fusion Strategy:
    P(final) = α * P(classical) + (1-α) * P(quantum)
    Where α is the classical weight (default=0.8).
"""

import os
import json
import pickle
import numpy as np
import torch
import pandas as pd
from quantum.qnn_model import load_qnn

class HybridIDS:
    def __init__(self, 
                 classical_path="models/xgboost_model.pkl",
                 quantum_path="quantum/qnn_weights.pt",
                 scaler_path="quantum/angle_scaler.pkl",
                 features_path="quantum/selected_features.json",
                 label_encoder_path="models/label_encoder.pkl",
                 classical_weight=0.8):
        
        self.classical_weight = classical_weight
        self.quantum_weight = 1.0 - classical_weight
        
        # 1. Load Classical Model
        import joblib
        self.classical_model = joblib.load(classical_path)
            
        # 2. Load Label Encoder (for 15-class mapping)
        with open(label_encoder_path, 'rb') as f:
            self.le = pickle.load(f)
            
        # 3. Load QNN Configuration
        with open(features_path, 'r') as f:
            config = json.load(f)
            self.q_features = config["features"]
            
        # Both models were retrained on a 3-class subset 
        self.qnn_classes = ["BENIGN", "DoS Hulk", "PortScan"]
            
        # 4. Load QNN Weights 
        self.qnn = load_qnn(n_classes=len(self.qnn_classes), path=quantum_path)
        self.qnn.eval()            

        
        # 5. Load Angle Scaler (for QNN input normalization [0, pi])
        import joblib
        self.scaler = joblib.load(scaler_path)


    def _preprocess_for_quantum(self, X):
        """Extracts and scales the 4 specific features required by the QNN."""
        if isinstance(X, np.ndarray):
            # If X is a numpy array, we assume it has all features in matching order
            # This is risky; we prefer DataFrames.
            raise ValueError("HybridIDS requires a pandas DataFrame to ensure feature alignment.")
            
        # Extract the specific 4 features
        X_q = X[self.q_features].copy()
        
        # Scale to [0, pi] for angle encoding
        X_scaled = self.scaler.transform(X_q)
        return torch.tensor(X_scaled, dtype=torch.float32)

    def predict_proba(self, X):
        """
        Calculates weighted consensus probabilities.
        Returns: (n_samples, 15) array of probabilities.
        """
        # Ensure Dataframe
        if not isinstance(X, pd.DataFrame):
            feature_names = self.classical_model.feature_names_in_
            X = pd.DataFrame(X, columns=feature_names)

        # 1. Classical Probabilities (15-class)
        p_classical = self.classical_model.predict_proba(X)
        
        # 2. Quantum Probabilities (3-class)
        X_q = self._preprocess_for_quantum(X)
        with torch.no_grad():
            logits = self.qnn(X_q)
            p_quantum_small = torch.softmax(logits, dim=1).numpy()
            
        # Map 3-class QNN output to the full 15-class space
        p_quantum = np.zeros_like(p_classical)
        for i, class_label in enumerate(self.qnn_classes):
             if class_label in self.le.classes_:
                 class_idx = list(self.le.classes_).index(class_label)
                 p_quantum[:, class_idx] = p_quantum_small[:, i]
            
        # 3. Weighted Consensus Logic
        # For each sample, identify the top predicted class from XGBoost
        p_final = np.zeros_like(p_classical)
        for i in range(len(p_classical)):
            top_class_idx = np.argmax(p_classical[i])
            top_class_label = self.le.classes_[top_class_idx]
            
            if top_class_label in self.qnn_classes:
                # Use Hybrid fusion (80/20)
                p_final[i] = (self.classical_weight * p_classical[i]) + (self.quantum_weight * p_quantum[i])
            else:
                # Use 100% Classical (Out-of-Quantum-Subset)
                p_final[i] = p_classical[i]
                
        # Re-normalize just in case
        p_final = p_final / p_final.sum(axis=1, keepdims=True)
        return p_final

    def predict(self, X):
        """Returns the consensus class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_contributing_models(self):
        return {
            "classical": "XGBoost (Pre-trained)",
            "quantum": "Hybrid QNN (4-Qubit PQC)",
            "weights": {
                "classical": self.classical_weight,
                "quantum": self.quantum_weight
            }
        }
