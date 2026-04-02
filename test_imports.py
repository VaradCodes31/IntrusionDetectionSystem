import os
import sys

# Ensure root directory is in path for local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Importing streamlit")
import streamlit as st
print("Importing pandas")
import pandas as pd
print("Importing joblib")
import joblib
print("Importing shap")
import shap
print("Importing plt")
import matplotlib.pyplot as plt
print("Importing numpy")
import numpy as np
print("Importing plotly")
import plotly.express as px
print("Importing torch")
import torch
print("Importing hybrid")
from quantum.hybrid_ensemble import HybridIDS
print("Done imports")

if __name__ == "__main__":
    print("Test successful")
