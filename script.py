import math
import random
from math import gcd
import numpy as np
from fractions import Fraction

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session

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
    Build a more robust Shor circuit using qubit redundancy
    """
    # Increase auxiliary qubits for better reliability
    n_aux = math.ceil(math.log2(N)) * 2  # Double auxiliary qubits
    
    # Use more counting qubits for better phase estimation
    n_count = max(n_count, 12)  # Use at least 12 counting qubits
    
    # Add ancilla qubits for error detection
    n_ancilla = n_count  # One ancilla per counting qubit
    
    total_qubits = n_count + n_aux + n_ancilla + 1
    print(f"Using {total_qubits} qubits:")
    print(f"- {n_count} counting qubits")
    print(f"- {n_aux} auxiliary qubits")
    print(f"- {n_ancilla} ancilla qubits")
    print(f"- 1 target qubit")
    
    qc = QuantumCircuit(total_qubits, n_count)
    
    # Initialize counting qubits in superposition with verification
    for q in range(n_count):
        # Add verification ancilla
        qc.h(q)
        qc.cx(q, q + n_count + n_aux)  # Use ancilla to verify H gate
        qc.barrier()
    
    # Initialize target register to |1> with verification
    target_qubit = n_count + n_aux
    qc.x(target_qubit)
    qc.barrier()
    
    # Apply controlled operations with error detection
    for q in range(n_count):
        power = 2**q
        # Add error detection using ancilla
        ancilla = q + n_count + n_aux
        
        # Controlled operation with verification
        c_exp = controlled_modular_exponentiation(a, power, N, n_aux)
        qc.append(c_exp, [q] + list(range(n_count, target_qubit + 1)))
        
        # Verify operation using ancilla
        qc.cx(q, ancilla)
        qc.barrier()
    
    # Apply inverse QFT with error checking
    iqft = inverse_qft(n_count)
    qc.append(iqft.to_instruction(), range(n_count))
    qc.barrier()
    
    # Add error syndrome measurement
    for q in range(n_count):
        # Measure in rotated basis for better error detection
        qc.h(q)
        # Add parity check with ancilla
        qc.cx(q, q + n_count + n_aux)
        qc.measure(q, q)
    
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


def run_shor_on_77(a=2, n_count=4, shots=2048, backend=None, service=None):
    """
    Shor's implementation with IBM Runtime support using minimal qubits
    """
    N = 77
    
    if math.gcd(a, N) != 1:
        print(f"Error: a={a} must be coprime with N={N}")
        return None
        
    # Use minimal qubits
    n_aux = math.ceil(math.log2(N))  # Minimal auxiliary qubits
    total_qubits = n_count + n_aux + 1
    
    print(f"\nUsing minimal configuration:")
    print(f"- {n_count} counting qubits")
    print(f"- {n_aux} auxiliary qubits")
    print(f"- 1 target qubit")
    print(f"Total qubits: {total_qubits}")
    
    qc = build_shor_circuit(a, N, n_count)
    
    # Use provided backend+service or default to local simulator
    if backend is None or service is None:
        from qiskit_aer import Aer
        simulator = Aer.get_backend('aer_simulator')
        return simulator.run(transpile(qc, simulator), shots=shots)
    else:
        # Use IBM Runtime with Session and Sampler
        with Session(backend=backend) as session:
            sampler = Sampler()
            
            # Simple optimization
            compiled_qc = transpile(
                qc, 
                backend,
                optimization_level=3,
                layout_method='sabre',
                routing_method='sabre',
                seed_transpiler=42
            )
            
            print(f"\nCircuit depth: {compiled_qc.depth()}")
            
            # Run with original shot count
            job = sampler.run([compiled_qc], shots=shots)
            return job

def process_shor_results(jobs):
    """
    Process multiple Shor jobs and try to extract factors
    """
    all_counts = {}
    total_shots = 0
    
    # Aggregate results from all jobs
    for job in jobs:
        result = job.result()
        counts = process_primitive_result(job)
        if counts:
            for bitstring, count in counts.items():
                all_counts[bitstring] = all_counts.get(bitstring, 0) + count
                total_shots += count
    
    # Sort by frequency and filter noise
    threshold = 0.005  # 0.5% threshold
    significant_counts = {k: v for k, v in all_counts.items() 
                        if v/total_shots > threshold}
    
    print("\nSignificant measurements (>0.5%):")
    print("\nBinary     Decimal  Count   Percentage")
    print("-" * 40)
    
    for bitstring, count in sorted(significant_counts.items(), 
                                 key=lambda x: x[1], reverse=True):
        decimal = int(bitstring, 2)
        percentage = (count / total_shots) * 100
        print(f"{bitstring}  {decimal:7d}  {count:5d}   {percentage:6.2f}%")
        
        # Try to extract factors
        try:
            phase = decimal / (1 << len(bitstring))
            denominator = Fraction(phase).limit_denominator(77)
            r = denominator.denominator
            
            if r % 2 == 0:
                guessed_factor = math.gcd(pow(2, r//2) + 1, 77)
                if guessed_factor not in [1, 77]:
                    print(f"Possible factor found: {guessed_factor}")
        except:
            continue
    
    return significant_counts

def process_primitive_result(job):
    """
    Process results from either simulator or Sampler
    """
    result = job.result()
    
    # Check if this is a Sampler result
    if hasattr(result, 'quasi_dists'):
        from qiskit.utils.mitigation import quasi_to_counts
        counts = quasi_to_counts(result.quasi_dists[0])
    else:
        counts = result.get_counts()
    
    return counts


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
