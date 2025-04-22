## Project Overview

The QGAN combines a **quantum generator** (a 4-qubit variational circuit) with a **classical discriminator** (a PyTorch neural network) to generate synthetic data. The generator leverages quantum superposition and entanglement to encode complex probability distributions, while the discriminator evaluates whether samples are real or fake. Through a minimax game, the QGAN learns to produce samples indistinguishable from a Gaussian distribution, with losses converging near 0.693 (ln(2)).

### Key Features
- **Quantum Generator**: A 4-qubit circuit with 16 parameterized gates, optimized via COBYLA.
- **Classical Discriminator**: A feedforward neural network with label smoothing and gradient clipping for stable training.
- **Outputs**: Histograms and loss plots styled in a futuristic cyan (#00CED1) and orange (#FF4500) aesthetic, using Montserrat font.
- **Applications**: Synthetic patient data for Virtual Hospital, financial modeling, and more.

## Quantum Circuit Explanation

The quantum circuit is the heart of the QGAN’s generator, implemented in Google Cirq. Below is a detailed breakdown of its structure, mechanics, and quantum principles.

### Circuit Structure
- **Qubits**: 4 qubits, each represented as a `cirq.GridQubit`, form a 16-dimensional quantum state space (2⁴).
- **Gates**:
  - **Parameterized Rotations**: Each qubit undergoes a sequence of gates:
    - `RX(θ)`: Rotation around the X-axis.
    - `RZ(θ)`: Rotation around the Z-axis.
    - `RY(θ)`: Rotation around the Y-axis.
    - `RZ(θ)`: Second Z-rotation.
    - Total: 4 parameters per qubit × 4 qubits = 16 parameters.
  - **Entangling Gates**: CNOT gates between adjacent qubits (e.g., qubit 0 to 1, 1 to 2) create entanglement, correlating their states.
  - **Measurement**: All qubits are measured in the computational basis, yielding a 4-bit string (e.g., 1011).
- **Output Mapping**: The 4-bit string is converted to a real number in [-1, 1]:
  ```
  value = (sum(bits * 2^[3,2,1,0]) / 2^(n_qubits-1)) - 1
  ```
  Example: 1011 → 11 (decimal) → 11/8 - 1 = 0.375.

### Role in QGAN
- The circuit encodes a probability distribution over [-1, 1], adjusted by its 16 parameters.
- By running the circuit multiple times (`n_samples=50`), it generates a batch of samples approximating the learned distribution.
- The parameters are optimized to minimize the generator loss, making the samples mimic a Gaussian (mean=0, std=1).

### Quantum Principles
- **Superposition**: Each qubit exists in a combination of |0⟩ and |1⟩, allowing the circuit to represent 2⁴ states simultaneously. This enables efficient exploration of probability distributions.
- **Entanglement**: CNOT gates entangle qubits, creating correlations that enhance the circuit’s ability to model complex distributions with fewer parameters than classical generators.
- **Measurement Collapse**: Measuring the qubits collapses the quantum state to a classical 4-bit string, producing a sample. Repeated measurements build the distribution.

### Intuition
Think of the quantum circuit as a “quantum artist” painting a probability distribution:
- The 16 parameters are brushstrokes, shaping the distribution’s shape.
- Superposition lets the artist sketch all possible patterns at once.
- Entanglement adds depth, linking strokes for richer patterns.
- Measurement captures a single stroke, repeated to form the final painting—a Gaussian distribution.

### Why Quantum?
The quantum circuit’s high-dimensional state space and entanglement allow it to potentially outperform classical generators in modeling complex data with fewer resources. For 4 qubits, the 16-dimensional state space encodes distributions compactly, offering a glimpse into quantum advantages for generative AI.

## Installation

### Prerequisites
- **System**: Linux (Ubuntu recommended), macOS, or Windows.
- **Python**: 3.12 (3.11 as fallback if issues occur).
- **Hardware**: 16-32GB RAM minimum (32GB+ recommended to avoid memory issues).

### Developer System Config
- **System**: Ubuntu Linux
- **Python**: 3.13.
- **Hardware**: 16GB (But Crashes Took Place(cloud recommended)).

### Setup
1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv qgan_env
   source qgan_env/bin/activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install cirq==1.3.0 torch==2.4.0 numpy==1.26.4 matplotlib==3.9.2 scipy==1.14.1 psutil==6.0.0
   ```
3. **Verify Installation**:
   ```bash
   pip list
   ```
   Ensure versions match the specified ones.

## Usage

1. **Save the Script**:
   - Copy the `qgan_gaussian_final.py` script to your working directory.
   - Avoid names like `__gc__.py` (reserved for Python magic methods).

2. **Run the Script**:
   ```bash
   python3 qgan_gaussian_final.py
   ```

3. **Outputs**:
   - **Console**: Training progress with discriminator and generator losses (e.g., “Epoch 0, D Loss: 0.6910, G Loss: 0.6237”).
   - **Files**:
     - `qgan_distribution_epoch_X.png`: Histograms comparing real (Gaussian) vs. generated distributions, styled in cyan (#00CED1) and orange (#FF4500) with Montserrat font.
     - `qgan_losses_epoch_X.png`: Loss plots showing convergence near 0.693.
     - `qgan.log`: Logs with memory usage (e.g., “RSS=250.67MB”) and errors.

4. **Expected Results**:
   - Training runs for 50-100 epochs or stops early if losses plateau.
   - Histograms show the generated distribution converging to the Gaussian (mean=0, std=1).
   - Losses stabilize near 0.693, indicating a balanced GAN.

## Troubleshooting

### Segmentation Faults
**Issue**: The script may crash with “segmentation fault (core dumped)” due to memory overload in Cirq’s simulator or PyTorch, especially on systems with 4-8GB RAM.

**Solutions**:
1. **Check Logs**:
   - Open `qgan.log` and look for memory spikes (e.g., “RSS=4000MB”) or errors (e.g., “Generator sampling failed”).
   - If `faulthandler` provides a stack trace, note the C++ function causing the crash.
2. **Reduce Memory Usage**:
   - Edit `qgan_gaussian_final.py` and set `batch_size=25`, `n_samples=25`:
     ```python
     batch_size = 25
     n_samples = 25  # In sample_generator, train_generator, optimize_generator
     ```
3. **Use DensityMatrixSimulator**:
   - Uncomment the following in `sample_generator`:
     ```python
     simulator = cirq.DensityMatrixSimulator()
     ```
   - This is slower but more memory-efficient.
4. **Monitor RAM**:
   - Run `top` or `htop` during execution. If RAM exceeds 90%, try the above steps or use a higher-RAM system.
5. **Run on Google Colab**:
   - Upload the script to Colab (12GB RAM free tier):
     ```bash
     !pip install cirq==1.3.0 torch==2.4.0 numpy==1.26.4 matplotlib==3.9.2 scipy==1.14.1 psutil==6.0.0
     ```
     - Run and download outputs (`qgan_distribution_epoch_X.png`, `qgan_losses_epoch_X.png`).
6. **Python 3.11 Fallback**:
   - If Python 3.12 causes issues:
     ```bash
     sudo apt install python3.11 python3.11-venv
     python3.11 -m venv qgan_env
     source qgan_env/bin/activate
     pip install cirq==1.3.0 torch==2.4.0 numpy==1.26.4 matplotlib==3.9.2 scipy==1.14.1 psutil==6.0.0
     ```

### Training Instability
- **Discriminator Overpowering** (D loss << G loss):
  - Reduce learning rate:
    ```python
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    ```
- **Generator Struggling** (G loss > 0.8):
  - Increase circuit depth to 6 parameters per qubit (see script comments for code).

## Results

### Training Progress
- **Losses**: Start at ~0.6910 (D) and 0.6237 (G), stabilizing near 0.693 (ln(2)) after 50-100 epochs, indicating a balanced GAN.
- **Example Output**:
  ```
  Epoch 0, D Loss: 0.6910, G Loss: 0.6237
  Epoch 10, D Loss: 0.6900, G Loss: 0.6400
  Epoch 20, D Loss: 0.6850, G Loss: 0.6600
  ...
  ```

### Visualizations
- **Histograms** (`qgan_distribution_epoch_X.png`):
  - Show real (Gaussian) vs. generated distributions.
  - Convergence is visible when histograms overlap (typically by epoch 50-100).
  - Styled in cyan (#00CED1) for real data and orange (#FF4500) for generated, with Montserrat font.
- **Loss Plots** (`qgan_losses_epoch_X.png`):
  - Plot generator (orange) and discriminator (cyan) losses.
  - Show stabilization near 0.693, indicating successful training.

### Applications
- **Healthcare**: Generates synthetic patient data (e.g., blood reports) for Virtual Hospital, enabling privacy-preserving AI training.
- **Finance**: Synthesizes financial data for fraud detection or risk modeling.
- **Scalability**: Extensible to complex datasets like MNIST or QM9 for image or molecular data generation.

## Contributing
We welcome contributions to enhance the QGAN’s performance, scalability, or applications. Please:
- Fork the repository.
- Submit pull requests with clear descriptions.
- Report issues via GitHub Issues.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
Developed by Synaptic Loop, a quantum-AI startup at VIT AP. Reach out at [insert contact email] or visit our [GitHub repository](https://github.com/synaptic-loop/qgan) for updates.

---
*“Powered by quantum entanglement and AI, our QGAN doesn’t just generate data—it thinks in superposition, shaping the future of synthetic intelligence.”*