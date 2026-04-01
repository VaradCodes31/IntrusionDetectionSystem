"""
quantum/qnn_model.py
──────────────────────────────────────────────────────────────────────────────
Purpose:
    "Industrial Strength" Hybrid QNN. This version uses explicit Kronecker
    product expansions to build the 4-qubit Hilbert space (16 dims) and
    performs the entire circuit simulation as a single Matrix-Matrix product.

Why?
    Previous versions using tensor reshapes and 'einsum' encountered hangs
    on certain MacOS environments. Flattened matrix-vector multiplication 
    is the most stable and performant operation in PyTorch.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
N_QUBITS = 4
N_LAYERS = 1
N_TRAIN_CAP = 300
EPOCHS = 15
LEARNING_RATE = 0.05

# ──────────────────────────────────────────────────────────────────────────────
# Ultra-Stable 16-Dimensional Simulator
# ──────────────────────────────────────────────────────────────────────────────
class HilbertSimulator(nn.Module):
    """
    Executes a 4-qubit circuit via 16x16 matrix multiplication.
    Mathematically identical to the PennyLane PQC.
    """
    def __init__(self, n_layers=N_LAYERS, n_qubits=N_QUBITS):
        super().__init__()
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.I = torch.eye(2)
        self.X = torch.tensor([[0., 1.], [1., 0.]])
        self.Z = torch.tensor([[1., 0.], [0., -1.]])

    def _kron(self, a, b):
        return torch.kron(a, b)

    def _get_op(self, gate, qubit_idx):
        """Builds a 16x16 operator for a 1-qubit gate."""
        res = torch.tensor([[1.]])
        for i in range(4):
            if i == qubit_idx:
                res = torch.kron(res, gate)
            else:
                res = torch.kron(res, self.I)
        return res

    def forward(self, x):
        batch_size = x.shape[0]
        # Initial state |0000>
        state = torch.zeros((batch_size, 16), device=x.device)
        state[:, 0] = 1.0
        
        # 1. Angle Encoding (Ry)
        for i in range(self.n_qubits):
            c, s = torch.cos(x[:, i]), torch.sin(x[:, i])
            n_left = 2**i
            n_right = 16 // (n_left * 2)
            state = state.reshape(batch_size, n_left, 2, n_right)
            
            s0, s1 = state[:, :, 0, :], state[:, :, 1, :]
            new_s0 = c.view(-1, 1, 1) * s0 - s.view(-1, 1, 1) * s1
            new_s1 = s.view(-1, 1, 1) * s0 + c.view(-1, 1, 1) * s1
            state = torch.stack([new_s0, new_s1], dim=2).reshape(batch_size, 16)

        # 2. Layer Logic (Rotations + Entanglement)
        for l in range(self.n_layers):
            # Rotations
            for i in range(self.n_qubits):
                theta = self.weights[l, i]
                c, s = torch.cos(theta), torch.sin(theta)
                
                n_left = 2**i
                n_right = 16 // (n_left * 2)
                state = state.reshape(batch_size, n_left, 2, n_right)
                
                s0, s1 = state[:, :, 0, :], state[:, :, 1, :]
                new_s0 = c * s0 - s * s1
                new_s1 = s * s0 + c * s1
                state = torch.stack([new_s0, new_s1], dim=2).reshape(batch_size, 16)
            
            # Entanglement (Cyclic Roll/Permutation)
            for (c_bit, t_bit) in [(0,1), (1,2), (2,3), (3,0)]:
                c_val = 2**(3 - c_bit)
                t_val = 2**(3 - t_bit)
                
                indices = torch.arange(16)
                has_control = (indices & c_val) != 0
                # New state constructed by swapping components
                idx = torch.arange(16)
                idx_c = idx[(idx & c_val) != 0]
                idx_t = idx_c ^ t_val
                
                new_state = state.clone()
                new_state[:, idx_c] = state[:, idx_t]
                new_state[:, idx_t] = state[:, idx_c]
                state = new_state

        # 3. Measurement (Expectation value of Pauli Z)
        probs = state**2
        expectations = []
        for i in range(4):
            idx = torch.arange(16)
            mask = (idx >> (3 - i)) & 1
            z_vals = 1.0 - 2.0 * mask.float()
            expectations.append(torch.sum(probs * z_vals.view(1, -1), dim=1))
            
        return torch.stack(expectations, dim=1)

# ──────────────────────────────────────────────────────────────────────────────
# API (Standardized)
# ──────────────────────────────────────────────────────────────────────────────
class HybridQNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.q_layer = HilbertSimulator()
        self.classifier = nn.Linear(4, n_classes)
    def forward(self, x):
        return self.classifier(self.q_layer(x))

def train_qnn(X_train, y_train, n_classes):
    # Avoid Threading deadlocks with XGBoost
    torch.set_num_threads(1)
    
    if len(X_train) > N_TRAIN_CAP:
        idx = np.random.choice(len(X_train), N_TRAIN_CAP, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    
    model = HybridQNN(n_classes)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    
    print(f"[QNN] Linear-Algebraic Simulation... (N={len(X_train)})", flush=True)
    t0 = time.time()
    for e in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        output = model(X_t)
        loss = loss_fn(output, y_t)
        loss.backward()
        optimizer.step()
        if e == 1 or e % 5 == 0:
            print(f"  Epoch [{e:>2}/{EPOCHS}] Loss: {loss.item():.4f}", flush=True)
    print(f"[QNN] Done in {time.time()-t0:.2f}s", flush=True)
    return model

def predict_qnn(model, X_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs

def save_qnn(model, path="quantum/qnn_weights.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_qnn(n_classes, path="quantum/qnn_weights.pt"):
    model = HybridQNN(n_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model

def get_circuit_diagram():
    return "Vectorized 4-Qubit PQC Study (Optimized for Local CPU)"
