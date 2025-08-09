# Hybrid PennyLane Classifier (Notebook as script)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pennylane as qml

from utils.data_loader import load_eeg_or_simulate

# Load data
X, y = load_eeg_or_simulate(n_subjects=200, n_features=16, seed=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

# Torch datasets
class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = torch.utils.data.DataLoader(NumpyDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(NumpyDataset(X_val, y_val), batch_size=32, shuffle=False)

# Quantum circuit
n_qubits, n_layers = 4, 2
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev, interface='torch')
def quantum_net(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    weights = weights.reshape(n_layers, n_qubits, 3)
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[l,i,0], wires=i)
            qml.RY(weights[l,i,1], wires=i)
            qml.RZ(weights[l,i,2], wires=i)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.q_params = nn.Parameter(torch.randn(n_layers * n_qubits * 3) * 0.01)
    def forward(self, x):
        return torch.stack([quantum_net(x[i], self.q_params) for i in range(x.shape[0])])

class HybridModel(nn.Module):
    def __init__(self, n_input, n_qubits, n_layers, n_classes=2):
        super().__init__()
        self.feature_fc = nn.Sequential(
            nn.Linear(n_input, 32),
            nn.ReLU(),
            nn.Linear(32, n_qubits),
            nn.Tanh()
        )
        self.qlayer = QuantumLayer(n_qubits, n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )
    def forward(self, x):
        x = self.feature_fc(x) * np.pi
        q_out = self.qlayer(x)
        return self.classifier(q_out)

model = HybridModel(X_train.shape[1], n_qubits, n_layers)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # validation
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb).argmax(dim=1).numpy()
            preds.extend(pred.tolist())
            ys.extend(yb.numpy().tolist())
    acc = accuracy_score(ys, preds)
    print(f"Epoch {epoch} val acc {acc:.4f}")
