import numpy as np
import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Generate sequences
# ---------------------------
try:
    num_sequences = int(input("Please enter the desired number of sequences (between 20 to 30): "))
    length = int(input("Please enter the desired length of each sequence (between 1 and 10): "))
    if num_sequences > 30 or num_sequences < 20 or length < 1 or length > 10:
        raise ValueError
except ValueError:
    print("Input error: using default values: 20 sequences of length 5.")
    num_sequences = 20
    length = 5

sequences = [''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length)) for _ in range(num_sequences)]
print(f"\nGenerated {num_sequences} sequences of length {length}.")

# ---------------------------
# 2. DNA -> bits conversion function
# ---------------------------
def dna_to_bits(seq):
    mapping = {'A':'00', 'C':'01', 'G':'10', 'T':'11'}
    return ''.join(mapping[c] for c in seq)

# ---------------------------
# 3. Quantum Hamming Distance
# ---------------------------
def quantum_hamming(seq1, seq2, shots=1024):
    bits1 = dna_to_bits(seq1)
    bits2 = dna_to_bits(seq2)
    n = len(bits1)  # number of qubits

    # 1 ancilla + n qubits for x + n qubits for y
    qc = QuantumCircuit(2*n + 1, 1)

    # Prepare states |x> and |y> (X gate for bit=1)
    for i, b in enumerate(bits1):
        if b == '1':
            qc.x(i + 1)
    for i, b in enumerate(bits2):
        if b == '1':
            qc.x(i + 1 + n)

    # Swap Test
    qc.h(0)  # Hadamard on ancilla
    for i in range(n):
        qc.cswap(0, i + 1, i + 1 + n)
    qc.h(0)
    qc.measure(0,0)

    # Simulation
    sim = AerSimulator()
    t_qc = transpile(qc, sim)
    counts = sim.run(t_qc, shots=shots).result().get_counts()
    p0 = counts.get('0', 0) / shots

    fidelity = max(0, 2*p0 - 1)  # |<x|y>|^2
    return 1 - fidelity          # quantum Hamming distance

# ---------------------------
# 4. Compute distance matrices
# ---------------------------
D_quantum = np.zeros((num_sequences, num_sequences))
D_classical = np.zeros((num_sequences, num_sequences))

print(f"Computing distances for {num_sequences} sequences...")

for i in range(num_sequences):
    for j in range(i + 1, num_sequences):
        # Quantum
        q_dist = quantum_hamming(sequences[i], sequences[j])
        D_quantum[i][j] = D_quantum[j][i] = q_dist

        # Classical (Hamming normalized to 0-1)
        c_dist = sum(a != b for a, b in zip(sequences[i], sequences[j])) / length
        D_classical[i][j] = D_classical[j][i] = c_dist

print("Done!")

# ---------------------------
# 5. Heatmap visualization
# ---------------------------
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(D_classical, ax=ax[0], cmap="YlGnBu", annot=False)
ax[0].set_title("Classical Hamming Distance")
ax[0].set_xlabel("Sequence Index")
ax[0].set_ylabel("Sequence Index")

sns.heatmap(D_quantum, ax=ax[1], cmap="YlGnBu", annot=False)
ax[1].set_title("Quantum Hamming Distance (Swap Test)")
ax[1].set_xlabel("Sequence Index")
ax[1].set_ylabel("Sequence Index")

plt.tight_layout()
plt.show()

# ---------------------------
# 6. Error analysis and correlation
# ---------------------------
correlation = np.corrcoef(D_classical.flatten(), D_quantum.flatten())[0, 1]
mean_error = np.mean(np.abs(D_classical - D_quantum))

print(f"\n--- PROJECT ANALYSIS ---")
print(f"Correlation between methods: {correlation:.4f}")
print(f"Mean estimation error: {mean_error:.4f}")

# Sample comparisons
print("\nSample comparisons (Classical vs Quantum):")
for _ in range(3):
    i, j = random.randint(0, num_sequences-1), random.randint(0, num_sequences-1)
    print(f"Pair ({i}, {j}): Classical = {D_classical[i,j]:.2f}, Quantum = {D_quantum[i,j]:.2f}")

np.set_printoptions(precision=2, suppress=True)

print("\n=== NUMERICAL COMPARISON (5x5 fragment) ===")
print("\nCLASSICAL HAMMING (Normalized):")
print(D_classical[:5, :5])

print("\nQUANTUM HAMMING (Swap Test):")
print(D_quantum[:5, :5])

diff = D_quantum - D_classical
print(f"\nAverage difference between methods: {np.mean(diff):.4f}")
