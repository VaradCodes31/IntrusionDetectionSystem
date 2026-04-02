"""
quantum/qml_xai.py
──────────────────────────────────────────────────────────────────────────────
Purpose:
    Implements Integrated Gradients (IG) for the 4-qubit Quantum Neural 
    Network (QNN). This provides the feature attribution for the quantum 
    branch of the NetSage-IDS consensus.

Mechanism:
    Integrated Gradients (Sundararajan et al., 2017) computes the integral 
    of the gradients along the straight-line path from a baseline x' (zeros)
    to the input x.

    Attributions = (x - x') * ∫ [∂f(x' + α(x - x')) / ∂x] dα

    Since the QNN simulator is implemented in PyTorch, we can compute 
    the gradients exactly via backpropagation, respecting the 
    underlying parameter-shift logic.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def calculate_quantum_ig(
    model, 
    input_tensor, 
    target_class_idx, 
    baseline=None, 
    steps=50
):
    """
    Computes Integrated Gradients for the HybridQNN model.
    
    Args:
        model:             The HybridQNN instance.
        input_tensor:      The scaled 4-feature tensor [0, pi]^4.
        target_class_idx:  Class index for which to explain the prediction.
        baseline:          Baseline tensor (defaults to all zeros).
        steps:             Number of approximation steps for the integral.
        
    Returns:
        1D numpy array of 4 attribution weights.
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
        
    # Scale input for path computation
    alphas = torch.linspace(0, 1, steps)
    delta = input_tensor - baseline
    
    # Path coordinates: baseline + alpha * delta
    path_inputs = baseline + alphas.view(-1, 1) * delta
    path_inputs.requires_grad = True
    
    # Forward pass on all points along the path
    logits = model(path_inputs)
    probs = torch.softmax(logits, dim=1)
    
    # Target the probability of the predicted class
    target_probs = probs[:, target_class_idx]
    
    # Compute gradients along the path
    model.zero_grad()
    target_probs.backward(torch.ones_like(target_probs))
    
    # Average the gradients
    avg_grads = torch.mean(path_inputs.grad, dim=0)
    
    # Compute the final attribution
    attributions = (delta * avg_grads).detach().numpy()
    return attributions

def plot_quantum_attribution(
    attributions, 
    feature_names, 
    class_name, 
    save_path=None
):
    """
    Visualizes Quantum Integrated Gradients in the CSOC aesthetic.
    """
    colors = ["#00f2ff" if val >= 0 else "#ff0055" for val in attributions]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0a0b10")
    ax.set_facecolor("#0a0b10")
    
    bars = ax.barh(range(len(feature_names)), attributions, color=colors, height=0.6)
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, color="#c0c0c0", fontsize=10)
    ax.axvline(0, color="#555555", linewidth=0.8)
    
    ax.set_xlabel("Integrated Gradient (Quantum Sub-circuit Influence)", color="#00f2ff", fontsize=11)
    ax.set_title(
        f"Quantum Explainability (Q-IG)  →  Class: {class_name}",
        color="#00f2ff", fontsize=13, fontweight="bold", pad=20
    )
    
    # Styling spines
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.tick_params(colors="#888888")
    
    # Legend
    pos_patch = mpatches.Patch(color="#00f2ff", label="Positive Quantum Bias")
    neg_patch = mpatches.Patch(color="#ff0055", label="Negative Quantum Bias")
    ax.legend(handles=[pos_patch, neg_patch], facecolor="#0a0b10", 
              labelcolor="white", fontsize=9, loc="lower right")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, transparent=False)
        
    return fig
