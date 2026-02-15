import numpy as np
import random
from math import log2, ceil
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate sequences
try:
    num_sequences = int(input("Please enter the desired number of sequences (20–30): "))
    length = int(input("Please enter the desired length of each sequence (1–10): "))
    if num_sequences < 20 or num_sequences > 30 or length < 1 or length > 10:
        raise ValueError
except ValueError:
    print("Invalid input — using default values: 20 sequences of length 4.")
    num_sequences = 20
    length = 4

sequences = [
    ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length))
    for _ in range(num_sequences)
]

print(f"\nGenerated {num_sequences} DNA sequences of length {length}.")

# 2. DNA -> bits
def dna_to_bits(base):
    mapping = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
    return mapping[base]

# 3. Quantum Hamming Distance (Swap Test)

def quantum_hamming(seq1, seq2, shots=2048):
    assert len(seq1) == len(seq2)
    n = len(seq1)

    idx_qubits = ceil(log2(n))
    ancilla = 0
    idx_start = 1
    data_start = 1 + idx_qubits

    total_qubits = 1 + idx_qubits + 2
    qc = QuantumCircuit(total_qubits, 1)

    # Prepare uniform superposition over indices |i>
    for q in range(idx_qubits):
        qc.h(idx_start + q)

    # Encode sequence 1 → |x>
    for i, base in enumerate(seq1):
        ctrl_bits = format(i, f'0{idx_qubits}b')
        bits = dna_to_bits(base)

        for q, bit in enumerate(ctrl_bits):
            if bit == '0':
                qc.x(idx_start + q)

        for d, b in enumerate(bits):
            if b == '1':
                qc.mcx(
                    [idx_start + q for q in range(idx_qubits)],
                    data_start + d
                )

        for q, bit in enumerate(ctrl_bits):
            if bit == '0':
                qc.x(idx_start + q)

    qc.barrier()

    # Swap test
    qc.h(ancilla)

    # Encode sequence 2 → |y> (controlled by ancilla)
    for i, base in enumerate(seq2):
        ctrl_bits = format(i, f'0{idx_qubits}b')
        bits = dna_to_bits(base)

        for q, bit in enumerate(ctrl_bits):
            if bit == '0':
                qc.x(idx_start + q)

        for d, b in enumerate(bits):
            if b == '1':
                qc.mcx(
                    [ancilla] + [idx_start + q for q in range(idx_qubits)],
                    data_start + d
                )

        for q, bit in enumerate(ctrl_bits):
            if bit == '0':
                qc.x(idx_start + q)

    qc.h(ancilla)
    qc.measure(ancilla, 0)

    # Simulation
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    p0 = counts.get('0', 0) / shots
    overlap = max(0.0, 2 * p0 - 1)

    # Normalized Hamming distance estimate
    return 1 - overlap


# 4. Distance matrices

D_classical = np.zeros((num_sequences, num_sequences))
D_quantum = np.zeros((num_sequences, num_sequences))

print("Computing distance matrices...")

for i in range(num_sequences):
    for j in range(i + 1, num_sequences):
        # Classical normalized Hamming distance
        c_dist = sum(a != b for a, b in zip(sequences[i], sequences[j])) / length
        D_classical[i, j] = D_classical[j, i] = c_dist

        # Quantum distance (swap test)
        q_dist = quantum_hamming(sequences[i], sequences[j])
        D_quantum[i, j] = D_quantum[j, i] = q_dist

print("Done.")


# 5. Visualization

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(D_classical, ax=ax[0], cmap="YlGnBu")
ax[0].set_title("Classical Hamming Distance (Normalized)")

sns.heatmap(D_quantum, ax=ax[1], cmap="YlGnBu")
ax[1].set_title("Quantum Hamming Distance (Swap Test)")

plt.tight_layout()
plt.show()


# 6. Error analysis

correlation = np.corrcoef(D_classical.flatten(), D_quantum.flatten())[0, 1]
mean_error = np.mean(np.abs(D_classical - D_quantum))

print("\n--- PROJECT ANALYSIS ---")
print(f"Correlation (classical vs quantum): {correlation:.4f}")
print(f"Mean absolute error: {mean_error:.4f}")

print("\nSample comparisons:")
for _ in range(3):
    i, j = random.sample(range(num_sequences), 2)
    print(
        f"Pair ({i},{j}) → "
        f"Classical: {D_classical[i,j]:.2f}, "
        f"Quantum: {D_quantum[i,j]:.2f}"
    )

print("\n=== 5x5 MATRIX FRAGMENT ===")
print("Classical:")
print(np.round(D_classical[:5, :5], 2))
print("\nQuantum:")
print(np.round(D_quantum[:5, :5], 2))

