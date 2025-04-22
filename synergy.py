import cirq
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Generate real data (Gaussian distribution)
np.random.seed(42)
real_data = np.random.normal(0, 1, 10000)  # Mean=0, std=1
real_data = torch.tensor(real_data, dtype=torch.float32).reshape(-1, 1)

# 2. Quantum Generator (Cirq)
n_qubits = 4
qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
n_params = 2 * n_qubits  # Two parameters per qubit (RX, RZ)

def create_generator_circuit(params):
    circuit = cirq.Circuit()
    for i in range(n_qubits):
        # Parameterized RX and RZ gates
        circuit.append(cirq.rx(params[2*i]).on(qubits[i]))
        circuit.append(cirq.rz(params[2*i + 1]).on(qubits[i]))
        if i < n_qubits - 1:
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    circuit.append(cirq.measure(*qubits, key='m'))
    return circuit

def sample_generator(params, n_samples=1000):
    circuit = create_generator_circuit(params)
    simulator = cirq.Simulator()
    results = simulator.run(circuit, param_resolver={f'theta_{i}': params[i] for i in range(len(params))}, repetitions=n_samples)
    measurements = results.measurements['m']
    # Convert binary measurements to float in [-2, 2]
    values = np.sum(measurements * (2 ** np.arange(n_qubits)[::-1]), axis=1) / (2 ** (n_qubits - 1)) - 1
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 1)

# 3. Classical Discriminator (PyTorch)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

discriminator = Discriminator()
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
criterion = nn.BCELoss()

# 4. QGAN Training
def train_discriminator(real_samples, fake_samples):
    d_optimizer.zero_grad()
    real_labels = torch.ones(real_samples.size(0), 1)
    fake_labels = torch.zeros(fake_samples.size(0), 1)
    
    # Train on real data
    real_output = discriminator(real_samples)
    d_loss_real = criterion(real_output, real_labels)
    
    # Train on fake data
    fake_output = discriminator(fake_samples.detach())
    d_loss_fake = criterion(fake_output, fake_labels)
    
    d_loss = (d_loss_real + d_loss_fake) / 2
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()

def train_generator(params, n_samples=1000):
    fake_samples = sample_generator(params, n_samples)
    fake_output = discriminator(fake_samples)
    g_loss = criterion(fake_output, torch.ones(n_samples, 1))  # Generator wants fake to be classified as real
    return g_loss.item()

def optimize_generator(params, n_samples=1000):
    def objective(params):
        return train_generator(params, n_samples)
    result = minimize(objective, params, method='COBYLA', options={'maxiter': 10})
    return result.x, result.fun

# Training Loop
n_epochs = 100
batch_size = 1000
initial_params = np.random.randn(n_params) * 0.1
params = initial_params
g_losses, d_losses = [], []

for epoch in range(n_epochs):
    # Sample real and fake data
    idx = np.random.choice(len(real_data), batch_size, replace=False)
    real_samples = real_data[idx]
    fake_samples = sample_generator(params, batch_size)
    
    # Train discriminator
    d_loss = train_discriminator(real_samples, fake_samples)
    
    # Train generator
    params, g_loss = optimize_generator(params, batch_size)
    
    g_losses.append(g_loss)
    d_losses.append(d_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

# 5. Visualization
fake_samples = sample_generator(params, 10000).numpy()
plt.hist(real_data.numpy(), bins=50, alpha=0.5, label='Real (Gaussian)', density=True)
plt.hist(fake_samples, bins=50, alpha=0.5, label='Generated', density=True)
plt.title('QGAN: Real vs Generated Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.savefig('qgan_distribution.png')
plt.close()

# Plot losses
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.title('QGAN Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('qgan_losses.png')