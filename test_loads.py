import os
import sys

# Ensure root directory is in path for local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Importing modules")
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import subprocess
import torch
import pickle

# Quantum imports
from quantum.hybrid_ensemble import HybridIDS
from quantum.qml_xai import calculate_quantum_ig, plot_quantum_attribution

def load_resources():
    print("Loading XGBoost")
    model = joblib.load("models/xgboost_model.pkl")
    
    print("Loading Label Encoder")
    with open("models/label_encoder.pkl", 'rb') as f:
        le = pickle.load(f)
    label_map = {i: label for i, label in enumerate(le.classes_)}
    
    print("Instantiating HybridIDS")
    hybrid_model = HybridIDS(
        classical_path="models/xgboost_model.pkl",
        quantum_path="quantum/qnn_weights.pt",
        scaler_path="quantum/angle_scaler.pkl",
        features_path="quantum/selected_features.json",
        label_encoder_path="models/label_encoder.pkl",
        classical_weight=0.8
    )
    
    return model, label_map, hybrid_model, le

if __name__ == "__main__":
    load_resources()
    print("Test successful")
