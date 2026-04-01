"""
quantum/data_encoder.py
──────────────────────────────────────────────────────────────────────────────
Purpose:
    Converts classical network feature vectors into angle-encoded quantum
    circuit inputs. This is the bridge between classical preprocessing and
    the quantum circuits in qnn_model.py and quantum_kernel_svm.py.

Theory (Angle Encoding):
    Each classical feature value xᵢ is mapped to a rotation angle θᵢ and
    encoded into a qubit via a parameterized Y-rotation gate:

        |ψᵢ⟩ = Ry(θᵢ)|0⟩

    For this to be valid, θᵢ must lie in [0, π]. We achieve this by applying
    a MinMaxScaler that maps each feature to the range [0, π] before encoding.

    The resulting quantum state for K features is a K-qubit product state:
        |ψ⟩ = ⊗ᵢ Ry(xᵢ_normalized)|0⟩

    Entanglement (correlations between features) is introduced in a later
    stage by the StronglyEntanglingLayers in the PQC, NOT here. This keeps
    encoding and processing cleanly separated, consistent with the data
    re-uploading model (Pérez-Salinas et al., 2020).
"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


def encode_features(
    X,
    features: list,
    scaler: MinMaxScaler = None,
    fit: bool = False,
) -> tuple:
    """
    Extracts the selected quantum features from X and normalizes them
    to the range [0, π] for angle encoding.

    Args:
        X:        DataFrame containing the network traffic features.
        features: List of feature names to extract (output of feature_selector).
        scaler:   A pre-fitted MinMaxScaler. Required if fit=False.
        fit:      If True, fit a new scaler on this data (training time).
                  If False, transform using the provided scaler (inference time).

    Returns:
        Tuple of (encoded_array: np.ndarray, scaler: MinMaxScaler).
        encoded_array shape: (n_samples, len(features))
    """
    # Extract only the selected quantum features
    available = [f for f in features if f in X.columns]
    if len(available) < len(features):
        missing = set(features) - set(available)
        print(f"[QML Encoder] Warning: {len(missing)} features missing, filling with 0: {missing}")
        for m in missing:
            X = X.copy()
            X[m] = 0.0
        available = features  # now all exist

    X_sub = X[available].values.astype(np.float64)

    # Replace inf/nan with 0 (common in network capture artifacts)
    X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)

    if fit:
        scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        X_encoded = scaler.fit_transform(X_sub)
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when fit=False.")
        X_encoded = scaler.transform(X_sub)
        # Clip to [0, π] in case test data exceeds training range
        X_encoded = np.clip(X_encoded, 0.0, np.pi)

    return X_encoded, scaler


def save_scaler(scaler: MinMaxScaler, path: str = "quantum/angle_scaler.pkl") -> None:
    """Persists the fitted MinMaxScaler for use during inference."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    joblib.dump(scaler, path)
    print(f"[QML Encoder] Angle scaler saved to '{path}'")


def load_scaler(path: str = "quantum/angle_scaler.pkl") -> MinMaxScaler:
    """Loads the persisted MinMaxScaler."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No scaler found at '{path}'. Run benchmark.py first to generate it."
        )
    return joblib.load(path)
