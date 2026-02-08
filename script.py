from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
import random

length = random.randint(1, 10)
num_sequences = random.randint(20, 30)

sequences = [''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length)) for _ in range(num_sequences)]

print(f"Generated {num_sequences} sequences, all of length {length}:")
for seq in sequences:
    print(seq)

def classical_hamming(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sum(a != b for a, b in zip(seq1, seq2))

def quantum_hamming(seq1, seq2, shots=1024):
    assert len(seq1) == len(seq2)
    n = len(seq1)

    qc = QuantumCircuit(n, n)

    for i in range(n):
        if seq1[i] != seq2[i]:
            qc.x(i)

    qc.measure(range(n), range(n))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    bitstring = max(counts, key=counts.get)
    return bitstring.count('1')

# macierze odległości
N = len(sequences)
D_quantum = np.zeros((N, N), dtype=int)
D_classical = np.zeros((N, N), dtype=int)

for i in range(N):
    for j in range(N):
        D_quantum[i][j] = quantum_hamming(sequences[i], sequences[j])
        D_classical[i][j] = classical_hamming(sequences[i], sequences[j])

print("Classical:\n", D_classical)
print("Quantum:\n", D_quantum)

diff_count = np.sum(D_classical != D_quantum)
print(f"Liczba różniących się elementów: {diff_count} na {N*N}")