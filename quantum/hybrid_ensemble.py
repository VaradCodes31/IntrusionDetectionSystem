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
        with open(classical_path, 'rb') as f:
            self.classical_model = pickle.load(f)
            
        # 2. Removed 15-class Label Encoder Load (Old structure)
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
        Returns: (n_samples, n_classes) array of probabilities.
        """
        # Ensure Dataframe
        if not isinstance(X, pd.DataFrame):
            # This handles cases where LIME/Counterfactuals pass numpy arrays
            # We must map them back to the original feature names.
            feature_names = self.classical_model.feature_names_in_
            X = pd.DataFrame(X, columns=feature_names)

        # 1. Classical Probabilities
        p_classical = self.classical_model.predict_proba(X)
        
        # 2. Quantum Probabilities
        X_q = self._preprocess_for_quantum(X)
        with torch.no_grad():
            logits = self.qnn(X_q)
            p_quantum = torch.softmax(logits, dim=1).numpy()
            
        # Both are strictly (N, 3), so direct addition is mathematically valid
        # 3. Decision Fusion
        p_final = (self.classical_weight * p_classical) + (self.quantum_weight * p_quantum)
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
