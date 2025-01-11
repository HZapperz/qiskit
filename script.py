import math
import random
from math import gcd
import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1) Classical GCD-based factor extraction
# -------------------------------------------------
def get_factor_from_period(a, N, r):
    """
    Given:
      a: the base of exponentiation
      N: the integer we're trying to factor
      r: the 'period' found from quantum phase estimation
    returns (factor1, factor2) or (None, None) if it fails.
    """
    if r % 2 != 0:
        # If r is odd, we can't do a^(r/2) ± 1, so this attempt might fail.
        return None, None
    
    # Compute gcd(a^(r/2) ± 1, N)
    plus  = pow(a, r // 2, N) + 1
    minus = pow(a, r // 2, N) - 1

    factor1 = gcd(plus,  N)
    factor2 = gcd(minus, N)
    
    # We expect that at least one gcd might be non-trivial
    # If factor1 == 1 or N, check factor2, etc.
    candidates = []
    for f in [factor1, factor2]:
        if 1 < f < N:
            candidates.append(f)
    # If we found two distinct factors, we’re done
    if len(candidates) == 2:
        return candidates[0], candidates[1]
    # Or if we found just one factor
    if len(candidates) == 1:
        return candidates[0], N // candidates[0]
    
    return None, None


# -------------------------------------------------
# 2) The Inverse QFT subcircuit
# -------------------------------------------------
def inverse_qft(num_qubits):
    """
    Build an n-qubit inverse QFT circuit.
    """
    qc = QuantumCircuit(num_qubits, name='IQFT')
    # Optional: you could do the "swaps" here, 
    # but for factoring we often skip them or handle them outside.
    for j in range(num_qubits // 2):
        qc.swap(j, num_qubits - j - 1)
    # Do the controlled-phase gates
    for j in range(num_qubits):
        for m in range(j):
            qc.cp(-np.pi / (2 ** (j-m)), m, j)
        qc.h(j)
    return qc


# -------------------------------------------------
# 3) A small "add" circuit, as a building block for multiply-by-a mod N
#    (This is a naive, partial approach; real Shor often uses more advanced adders.)
# -------------------------------------------------
def add_mod_N(num_qubits, N, const):
    """
    Simplified modular adder using basic quantum gates
    """
    qc = QuantumCircuit(num_qubits + 1, name=f"add({const})mod{N}")
    
    # Simple phase addition
    for i in range(num_qubits):
        angle = 2 * np.pi * (const * 2**i) / N
        qc.p(angle, i)
    
    return qc.to_gate()


def multiply_by_a_mod_N(num_qubits, a, N):
    """
    Simplified modular multiplier
    """
    qc = QuantumCircuit(num_qubits + 1, name=f"mult({a})mod{N}")
    
    # Use simpler binary decomposition
    a_mod_N = a % N
    for i in range(num_qubits):
        if (a_mod_N & (1 << i)) != 0:
            adder = add_mod_N(num_qubits, N, 1 << i)
            qc.append(adder, range(num_qubits + 1))
    
    return qc.to_gate()


def controlled_modular_exponentiation(a, power, N, n_aux):
    """
    Purely unitary modular exponentiation
    """
    qc = QuantumCircuit(1 + n_aux + 1, name=f"C-Mult(a^{power})mod{N}")
    
    current_a = pow(a, power, N)
    if current_a > 1:  # Skip if multiplication by 1
        mult = multiply_by_a_mod_N(n_aux, current_a, N)
        controlled_mult = mult.control(1)
        qc.append(controlled_mult, range(1 + n_aux + 1))
    
    return qc.to_gate()  # Convert to gate immediately


# -------------------------------------------------
# 4) Building the full Shor circuit for factoring N=77
# -------------------------------------------------
def build_shor_circuit(a, N, n_count):
    """
    Build the complete Shor circuit
    """
    n_aux = math.ceil(math.log2(N))
    total_qubits = n_count + n_aux + 1
    
    qc = QuantumCircuit(total_qubits, n_count)
    
    # Initialize counting qubits in superposition
    for q in range(n_count):
        qc.h(q)
    
    # Initialize target register to |1>
    qc.x(n_count)
    
    # Apply controlled operations
    for q in range(n_count):
        power = 2**q
        c_exp = controlled_modular_exponentiation(a, power, N, n_aux)
        qc.append(c_exp, [q] + list(range(n_count, total_qubits)))
    
    # Apply inverse QFT
    iqft = inverse_qft(n_count)
    qc.append(iqft.to_instruction(), range(n_count))
    
    # Measure
    qc.measure(range(n_count), range(n_count))
    
    return qc


def find_period_from_phase(phase, n_count, tolerance=0.01):
    """
    Enhanced period finding using continued fractions and better filtering
    """
    candidates = []
    
    def continued_fraction(x, depth=20):  # Increased depth
        fractions = []
        a = int(x)
        fractions.append(a)
        x = x - a
        depth -= 1
        while depth > 0 and abs(x) > 1e-10:
            x = 1/x if x != 0 else 0
            a = int(x)
            fractions.append(a)
            x = x - a
            depth -= 1
        return fractions
    
    def convergents(fracs):
        n = [0, 1]
        d = [1, 0]
        for i in range(len(fracs)):
            n.append(fracs[i] * n[-1] + n[-2])
            d.append(fracs[i] * d[-1] + d[-2])
            if i > 0:  # Skip first convergent
                r = n[-1]
                # Only consider reasonable periods
                if 1 < r < 2**n_count and r < 100:  # Added upper bound
                    candidates.append(r)
        return candidates
    
    fracs = continued_fraction(phase)
    return convergents(fracs)


def run_shor_on_77(a=2, n_count=8, shots=2048):  # Reduced parameters
    """
    Simplified Shor's implementation
    """
    N = 77
    
    if math.gcd(a, N) != 1:
        print(f"Error: a={a} must be coprime with N={N}")
        return
    
    # Print expected period
    period = 1
    current = a % N
    sequence = [current]
    while current != 1:
        current = (current * a) % N
        sequence.append(current)
        period += 1
    print(f"\nExpected period for a={a}: {period}")
    print(f"Sequence: {sequence}")
    
    qc = build_shor_circuit(a, N, n_count)
    
    # Use basic simulator settings
    backend = Aer.get_backend('aer_simulator')
    
    transpiled_qc = transpile(qc, backend, optimization_level=1)  # Reduced optimization
    job = backend.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Show histogram
    plot_histogram(counts)
    plt.title(f"Shor's Algorithm (N=77, a={a})")
    plt.show()
    
    # Analyze measurements
    successful_factors = set()
    print("\nAnalyzing measurements:")
    
    for bitstring, count in counts.items():
        if count < shots * 0.01:
            continue
            
        measured_decimal = int(bitstring, 2)
        phase = measured_decimal / (2**n_count)
        print(f"\nMeasurement: {bitstring}")
        print(f"Phase: {phase:.6f}")
        
        # Try simple period candidates
        for r in range(1, min(100, 2**n_count)):
            if abs(r * phase - round(r * phase)) < 0.1:  # Look for near-integer relationships
                possible_r = r
                print(f"Possible period: {possible_r}")
                
                # Verify
                if pow(a, possible_r, N) == 1:
                    print(f"Valid period found: {possible_r}")
                    f1, f2 = get_factor_from_period(a, N, possible_r)
                    if f1 is not None and f2 is not None:
                        if 1 < f1 < N and 1 < f2 < N:
                            successful_factors.add(tuple(sorted([f1, f2])))

    if successful_factors:
        print("\nFound factors of 77:")
        for f1, f2 in successful_factors:
            print(f"77 = {f1} × {f2}")
    else:
        print("\nNo factors found. Try different parameters.")


# -------------------------------------------------
# USAGE EXAMPLE
# -------------------------------------------------
if __name__ == "__main__":
    # Try different values of 'a'
    for a in [2, 3, 5]:
        print(f"\n{'='*60}")
        print(f"Trying with a = {a}")
        print('='*60)
        run_shor_on_77(a=a)
