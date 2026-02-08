import numpy as np
import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import seaborn as sns

try:
    num_sequences = int(input("Please enter the desired number of sequences (between 20 to 30): "))
    length = int(input("Please enter the desired length of each sequence (between 1 and 10): "))
    
    if num_sequences > 30 or num_sequences < 20 or length < 1 or length > 10:
        raise ValueError("Liczby muszą być dodatnie, a sekwencje co najmniej dwie.")
except ValueError as e:
    print(f"Błąd wejścia: {e}. Używam wartości domyślnych: 20 sekwencji o długości 5.")
    num_sequences = 20
    length = 5


sequences = [''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length)) for _ in range(num_sequences)]
print(f"\nWygenerowano {num_sequences} sekwencji o długości {length}.")

def get_quantum_gate_for_dna(char):
    """Mapowanie DNA na rotacje (Angle Encoding). 
    Każda zasada to inny punkt na okręgu jednostkowym."""
    mapping = {'A': 0, 'C': np.pi/2, 'G': np.pi, 'T': 3*np.pi/2}
    return mapping[char]

def quantum_swap_test(seq1, seq2, shots=1024):
    n = len(seq1)
    # Rejestr: 1 ancilla + n qubitów dla seq1 + n qubitów dla seq2
    qc = QuantumCircuit(2*n + 1, 1)
    
    # Kodowanie stanów (używamy bramki RY do ustawienia fazy/kąta)
    for i in range(n):
        qc.ry(get_quantum_gate_for_dna(seq1[i]), i + 1)
        qc.ry(get_quantum_gate_for_dna(seq2[i]), i + 1 + n)
    
    # Protokół Swap Test
    qc.h(0)
    for i in range(n):
        qc.cswap(0, i + 1, i + 1 + n)
    qc.h(0)
    
    qc.measure(0, 0)
    
    # Wykonanie
    sim = AerSimulator()
    t_qc = transpile(qc, sim)
    counts = sim.run(t_qc, shots=shots).result().get_counts()
    
    p0 = counts.get('0', 0) / shots
    # Wartość fidelity (podobieństwa)
    fidelity = max(0, 2*p0 - 1)
    return 1 - fidelity # Zwracamy dystans

# 2. Obliczanie macierzy (dla 25 sekwencji to 300 unikalnych par)
D_quantum = np.zeros((num_sequences, num_sequences))
D_classical = np.zeros((num_sequences, num_sequences))

print(f"Obliczam dystanse dla {num_sequences} sekwencji...")

for i in range(num_sequences):
    for j in range(i + 1, num_sequences):
        # Kwantowo
        q_dist = quantum_swap_test(sequences[i], sequences[j])
        D_quantum[i][j] = D_quantum[j][i] = q_dist
        
        # Klasycznie (Hamming znormalizowany do zakresu 0-1)
        c_dist = sum(a != b for a, b in zip(sequences[i], sequences[j])) / length
        D_classical[i][j] = D_classical[j][i] = c_dist

print("Gotowe!")
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Mapa dla dystansu klasycznego
sns.heatmap(D_classical, ax=ax[0], cmap="YlGnBu", annot=False)
ax[0].set_title("Klasyczna Odległość Hamminga")
ax[0].set_xlabel("Indeks sekwencji")
ax[0].set_ylabel("Indeks sekwencji")

# Mapa dla estymacji kwantowej
sns.heatmap(D_quantum, ax=ax[1], cmap="YlGnBu", annot=False)
ax[1].set_title("Kwantowa Estymacja Odległości (Swap Test)")
ax[1].set_xlabel("Indeks sekwencji")

plt.tight_layout()
plt.show()

# 2. Analiza błędu i korelacja
correlation = np.corrcoef(D_classical.flatten(), D_quantum.flatten())[0, 1]
mean_error = np.mean(np.abs(D_classical - D_quantum))

print(f"\n--- ANALIZA PROJEKTU ---")
print(f"Korelacja między metodami: {correlation:.4f}")
print(f"Średni błąd estymacji: {mean_error:.4f}")

# Wyświetlenie przykładowych porównań
print("\nPrzykładowe porównania (Klasyczne vs Kwantowe):")
for _ in range(3):
    i, j = random.randint(0, num_sequences-1), random.randint(0, num_sequences-1)
    print(f"Para ({i}, {j}): Klasycznie = {D_classical[i,j]:.2f}, Kwantowo = {D_quantum[i,j]:.2f}")

# Ustawienie precyzji wypisywania
np.set_printoptions(precision=2, suppress=True)

print("\n=== PORÓWNANIE NUMERYCZNE (Fragment 5x5) ===")
print("\nKLASYCZNY HAMMING (Znormalizowany):")
print(D_classical[:5, :5])

print("\nKWANTOWY DYSTANS (Swap Test):")
print(D_quantum[:5, :5])

# Obliczenie statystyk
diff = D_quantum - D_classical
print(f"\nŚrednia różnica między metodami: {np.mean(diff):.4f}")