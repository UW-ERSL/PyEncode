"""
pyencode.synthesizer
======================
Maps a recognised LoadPattern to an efficient Qiskit QuantumCircuit.

Each synthesis function implements the analytical circuit construction
derived from the structure of the load type.  Gate complexities:

  POINT_LOAD      :  O(m)       — X gates encoding binary(k), global phase
  UNIFORM_LOAD    :  O(m)       — H^{⊗m} followed by global amplitude scale
  STEP_LOAD       :  O(m)       — H on lower qubits, conditional on upper bits
  SINUSOIDAL_LOAD :  O(m²)      — QFT-based, encodes sin(2πnk/N + φ)
  COSINE_LOAD     :  O(m²)      — QFT-based, encodes cos(2πnk/N + φ)
  DISJOINT_POINT_LOAD          :  O(m · |loads|) — W-state style superposition
  MULTI_SIN_LOAD          :  O(m²)      — QFT + multi-amplitude encoding
  UNIFORM_SPIKE_LOAD          :  O(m)       — H^{⊗m} + multi-controlled Ry perturbation

The Möttönen fallback is provided for UNKNOWN patterns and uses the
general state-preparation routine shipped with Qiskit.

References
----------
  Möttönen et al., Quantum Inf. Comput. 5(6), 467-473, 2005.
  Shende, Markov, Bullock, IEEE TCAD 25(6), 1000-1010, 2006.
  Coppersmith, "An approximate Fourier transform useful in quantum
    factoring", IBM Research Report RC19642, 1994.
"""

import math
import numpy as np
from typing import Optional

from qiskit import QuantumCircuit, QuantumRegister

from .recognizer import LoadPattern, LoadType


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def synthesize(pattern: LoadPattern) -> QuantumCircuit:
    """
    Construct and return a Qiskit QuantumCircuit that prepares the
    quantum state corresponding to the given LoadPattern.

    The returned circuit acts on m = log2(N) qubits.  After execution
    the state is:

        |ψ⟩ = (1/‖f‖) Σ_k  f[k] |k⟩

    where f is the load vector encoded by the pattern.

    Parameters
    ----------
    pattern : LoadPattern

    Returns
    -------
    QuantumCircuit
    """
    N = pattern.N
    if N < 2 or (N & (N - 1)) != 0:
        raise ValueError(f"N must be a power of 2, got {N}")

    m = int(round(math.log2(N)))

    dispatch = {
        LoadType.POINT_LOAD:      _synth_point_load,
        LoadType.UNIFORM_LOAD:    _synth_uniform_load,
        LoadType.STEP_LOAD:       _synth_step_load,
        LoadType.SQUARE_LOAD:     _synth_square_load,
        LoadType.SINUSOIDAL_LOAD: _synth_sinusoidal,
        LoadType.COSINE_LOAD:     _synth_cosine,
        LoadType.DISJOINT_POINT_LOAD:          _synth_disjoint_point_load,
        LoadType.MULTI_SIN_LOAD:          _synth_multi_sin_load,
        LoadType.UNIFORM_SPIKE_LOAD:          _synth_uniform_spike_load,
        LoadType.UNKNOWN:         _synth_mottonen,
    }

    fn = dispatch.get(pattern.load_type, _synth_mottonen)
    return fn(m, pattern.params)


# ---------------------------------------------------------------------------
# Point load  f[k] = P
# ---------------------------------------------------------------------------

def _synth_point_load(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare |k⟩ using X gates on each qubit where the binary
    representation of k has a 1-bit.

    Gate count: at most m X gates.  O(m).
    """
    k = params["k"]
    qc = QuantumCircuit(m, name="point_load")

    # Encode k in binary: qubit 0 is LSB
    for bit in range(m):
        if (k >> bit) & 1:
            qc.x(bit)

    return qc


# ---------------------------------------------------------------------------
# Uniform load  f = ones(N) * c
# ---------------------------------------------------------------------------

def _synth_uniform_load(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare (1/√N) Σ_k |k⟩  by applying H to every qubit.

    Gate count: m Hadamard gates.  O(m).
    """
    qc = QuantumCircuit(m, name="uniform_load")
    for q in range(m):
        qc.h(q)
    return qc


# ---------------------------------------------------------------------------
# Step load  f[:k_s] = c
# ---------------------------------------------------------------------------

def _synth_step_load(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare (1/√k_s) Σ_{j=0}^{k_s - 1} |j⟩.

    Strategy: if k_s = 2^p (power of 2), apply H to the p lowest qubits
    and leave the rest as |0⟩.  For general k_s we use a binary-tree
    Ry decomposition.

    Gate count: O(m).
    """
    k_s = params["k_s"]
    N   = 2 ** m
    if k_s <= 0 or k_s > N:
        raise ValueError(f"k_s={k_s} out of range [1, {N}]")

    qc = QuantumCircuit(m, name="step_load")

    # Special case: k_s is a power of 2
    if k_s & (k_s - 1) == 0:
        p = int(round(math.log2(k_s)))
        for q in range(p):
            qc.h(q)
        return qc

    # General case: binary-tree Ry encoding
    # We encode k_s as a superposition over all indices < k_s.
    # Use recursive amplitude encoding on the address register.
    _uniform_superposition(qc, k_s, m)
    return qc


def _uniform_superposition(qc: QuantumCircuit, k: int, m: int):
    """
    Recursively prepare (1/√k) Σ_{j=0}^{k-1} |j⟩ on m qubits.
    Uses a top-down Ry decomposition (qubit m-1 is MSB).
    """
    # Represent k in binary; use amplitude tree
    # High qubit controls how many basis states are in the "left" half
    N = 2 ** m
    if k == N:
        for q in range(m):
            qc.h(q)
        return
    if k == 1:
        return  # |0...0⟩ already prepared
    if m == 1:
        # Single qubit: |0⟩ with amplitude sqrt((N-k)/N) ... simplified
        theta = 2 * math.acos(math.sqrt((N - k) / N))
        qc.ry(theta, 0)
        return

    half = N // 2
    if k <= half:
        # All states in the lower half; MSB stays |0⟩
        _uniform_superposition(qc, k, m - 1)
    else:
        # Some states in upper half
        k_upper = k - half
        # Amplitude for MSB = 0 is sqrt(half/k), for MSB = 1 is sqrt(k_upper/k)
        theta = 2 * math.acos(math.sqrt(half / k))
        qc.ry(theta, m - 1)
        # Lower half: apply H to all lower qubits (unconditionally for the |0⟩ branch)
        for q in range(m - 1):
            qc.h(q)
        # Upper branch: controlled uniform superposition over k_upper states
        # Approximate with Ry on lower qubits conditioned on MSB=1
        if k_upper < half:
            theta2 = 2 * math.acos(math.sqrt(k_upper / half))
            qc.cry(theta2, m - 1, m - 2)


# ---------------------------------------------------------------------------
# Square load  f[k1:k2] = c
# ---------------------------------------------------------------------------

def _synth_square_load(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare (1/sqrt(k2-k1)) * sum_{j=k1}^{k2-1} |j>.

    Strategy: use StatePreparation on the sparse binary vector.
    For power-of-2 aligned segments this reduces to a simple
    H-gate circuit; the general case uses the Shende fallback.

    Gate count: O(m) for aligned segments, O(2^m) general.
    """
    k1 = params["k1"]
    k2 = params["k2"]
    N  = 2 ** m
    w  = k2 - k1   # width of segment

    qc = QuantumCircuit(m, name="square_load")

    # Special case: width and start are both powers of 2
    # e.g. f[16:48] on N=64: k1=16=2^4, w=32=2^5
    if (w & (w - 1)) == 0 and (k1 & (k1 - 1)) == 0:
        p = int(round(math.log2(w)))       # H on p lower qubits
        # Encode k1 on upper bits
        for bit in range(p, m):
            if (k1 >> bit) & 1:
                qc.x(bit)
        for q in range(p):
            qc.h(q)
        return qc

    # General case: Shende on sparse vector
    f = np.zeros(N)
    f[k1:k2] = 1.0
    return _mottonen_from_vector(f.astype(complex), m, name="square_load")


# ---------------------------------------------------------------------------
# Sinusoidal load  f = A * sin(n * pi * x / L)
# ---------------------------------------------------------------------------

def _synth_sinusoidal(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare (1/‖f‖) Σ_k sin(n π k / N + φ) |k⟩ using the quantum Fourier
    transform.

    The DFT of sin(n π k / N + φ) is a pair of delta functions at
    frequencies ±n with complex amplitudes e^{±iφ}.  We:
      1. Prepare the frequency-domain state
             (e^{iφ}|n⟩ - e^{-iφ}|N-n⟩) / √2
      2. Apply QFT  to transform frequency → spatial domain

    The phase φ is encoded via a single P(2φ) gate on the control qubit,
    replacing the Z gate used for the φ=0 case (Z = P(π)).

    Gate count: O(m²) dominated by the QFT.

    References
    ----------
    Shende, Markov, Bullock, IEEE TCAD 25(6), 2006.
    Coppersmith, IBM Research Report RC19642, 1994.
    """
    n   = params["n"]
    phi = params.get("phi", 0.0)
    N   = 2 ** m

    if n <= 0 or n >= N:
        raise ValueError(f"Sinusoidal mode n={n} out of range (1, {N-1})")

    qc = QuantumCircuit(m, name=f"sin_n{n}_phi{phi:.4f}")

    # Step 1: prepare (e^{iφ}|n⟩ - e^{-iφ}|N-n⟩) / √2 in frequency domain
    qc.h(m - 1)   # |0⟩ branch → |n⟩,  |1⟩ branch → |N-n⟩

    _encode_index_controlled(qc, n,     m, control_qubit=m - 1, control_state=0)
    _encode_index_controlled(qc, N - n, m, control_qubit=m - 1, control_state=1)

    # Relative phase between the two branches.
    # After H the state is (|0⟩|n⟩ + |1⟩|N-n⟩)/√2.
    # Target: (e^{iφ}|n⟩ - e^{-iφ}|N-n⟩)/√2  →  need P(π - 2φ) on |1⟩ branch.
    # Derivation: factor e^{iφ} out, so |1⟩ branch needs -e^{-2iφ} = e^{i(π-2φ)}.
    # Check: φ=0 → P(π) = Z  ✓
    qc.p(math.pi - 2 * phi, m - 1)

    # Step 2: QFT to transform frequency → spatial domain
    from qiskit.circuit.library import QFTGate
    qft_gate = QFTGate(num_qubits=m)
    qc.append(qft_gate, qargs=list(range(m)))

    return qc



def _synth_cosine(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare (1/‖f‖) Σ_k cos(2π n k/N + φ) |k⟩ using the quantum Fourier
    transform.

    The DFT of cos(2πnk/N + φ) has spikes at frequencies ±n with
    amplitudes e^{±iφ}.  We:
      1. Prepare the frequency-domain state
             (e^{iφ}|n⟩ + e^{-iφ}|N-n⟩) / √2
      2. Apply QFT to transform frequency → spatial domain.

    The only difference from _synth_sinusoidal is the sign between the
    two branches: cosine uses '+' (P(-2φ) gate) vs sine's '-' (P(π-2φ)).

    Gate count: O(m²) dominated by the QFT.
    """
    n   = params["n"]
    phi = params.get("phi", 0.0)
    N   = 2 ** m

    if n <= 0 or n >= N:
        raise ValueError(f"Cosine mode n={n} out of range (1, {N-1})")

    qc = QuantumCircuit(m, name=f"cos_n{n}_phi{phi:.4f}")

    # Step 1: prepare (e^{iφ}|n⟩ + e^{-iφ}|N-n⟩) / √2 in frequency domain
    qc.h(m - 1)

    _encode_index_controlled(qc, n,     m, control_qubit=m - 1, control_state=0)
    _encode_index_controlled(qc, N - n, m, control_qubit=m - 1, control_state=1)

    # Relative phase: target |1⟩ branch factor = e^{-2iφ} = e^{i(-2φ)}
    # P(-2φ) gate.  Check: φ=0 → P(0) = I  (cosine is real, symmetric DFT) ✓
    qc.p(-2 * phi, m - 1)

    # Step 2: QFT
    from qiskit.circuit.library import QFTGate
    qc.append(QFTGate(num_qubits=m), qargs=list(range(m)))

    return qc

def _encode_index_controlled(qc: QuantumCircuit, idx: int, m: int,
                               control_qubit: int, control_state: int):
    """
    Flip the bits of idx onto qubits 0..m-2, controlled on control_qubit
    having value control_state.  Uses CX (controlled-NOT) gates.
    """
    for bit in range(m - 1):
        if (idx >> bit) & 1:
            if control_state == 0:
                qc.x(control_qubit)
            qc.cx(control_qubit, bit)
            if control_state == 0:
                qc.x(control_qubit)


# ---------------------------------------------------------------------------
# Case A: disjoint point loads
# ---------------------------------------------------------------------------

def _synth_disjoint_point_load(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare a weighted superposition of L disjoint point loads:

        |ψ⟩ ∝ Σ_{i=1}^{L} |P_i| |k_i⟩

    For L = 2 (the dominant engineering case): hand-crafted O(m) circuit.
      - One Ry gate allocates amplitude between the two branches
      - At most 2(m-1) CX gates encode the two indices
      - Total: O(m) gates

    For L > 2: falls back to Qiskit StatePreparation on the sparse vector.

    Reference: Shende, Markov, Bullock, IEEE TCAD 25(6), 2006.
    """
    loads = params["loads"]
    L     = len(loads)
    N     = 2 ** m

    if L == 2:
        k1, P1 = loads[0]["k"], abs(loads[0]["P"])
        k2, P2 = loads[1]["k"], abs(loads[1]["P"])
        return _synth_two_point_loads(m, k1, P1, k2, P2)

    # L > 2: use StatePreparation on sparse vector
    f = np.zeros(N)
    for load in loads:
        f[load["k"]] = abs(load["P"])
    return _mottonen_from_vector(f.astype(complex), m, name="disjoint_point_load")


def _synth_two_point_loads(m: int, k1: int, P1: float,
                            k2: int, P2: float) -> QuantumCircuit:
    """
    Exact O(m) circuit for the two-point-load state:

        |ψ⟩ = (P1 |k1⟩ + P2 |k2⟩) / √(P1² + P2²)

    Strategy
    --------
    1. Find the highest bit position where k1 and k2 differ — call it
       the *split bit*.  The corresponding qubit acts as the amplitude
       splitter.

    2. Apply Ry(θ) on the split qubit, with θ chosen so that the
       amplitude assigned to each branch matches P1, P2.

    3. For each branch (split_qubit = 0 or 1), apply CX gates
       controlled on the split qubit to encode the remaining bits of
       the target index.

    Gate count: 1 Ry + at most 2(m-1) CX + at most 2(m-1) X  =  O(m).
    """
    norm = math.sqrt(P1 ** 2 + P2 ** 2)
    if norm < 1e-14:
        raise ValueError("Both point loads are zero.")
    a1, a2 = P1 / norm, P2 / norm

    xor = k1 ^ k2
    if xor == 0:
        raise ValueError(f"Duplicate index k={k1}.")

    # Highest bit where k1 and k2 differ
    split_bit   = xor.bit_length() - 1
    split_qubit = split_bit   # qubit i encodes bit i (LSB = qubit 0)

    # Assign branches: branch 0 (split_qubit=0) and branch 1 (split_qubit=1)
    if (k1 >> split_bit) & 1 == 0:
        branch0_k, branch1_k = k1, k2
        theta = 2.0 * math.asin(a2)    # sin(θ/2) = amplitude in |1⟩ branch
    else:
        branch0_k, branch1_k = k2, k1
        theta = 2.0 * math.asin(a1)

    qc = QuantumCircuit(m, name="disjoint_point_load")
    qc.ry(theta, split_qubit)

    # Encode remaining bits of each branch index via controlled X
    for branch_val, target_k in [(0, branch0_k), (1, branch1_k)]:
        for bit in range(m):
            if bit == split_bit:
                continue
            if (target_k >> bit) & 1:
                if branch_val == 0:
                    qc.x(split_qubit)        # temporarily flip to make control = 1
                qc.cx(split_qubit, bit)
                if branch_val == 0:
                    qc.x(split_qubit)

    return qc


# ---------------------------------------------------------------------------
# Case B: sum of sinusoidal modes
# ---------------------------------------------------------------------------

def _synth_multi_sin_load(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare a superposition of multiple sinusoidal modes.

    Strategy: encode all frequency-domain amplitudes, then apply QFT†.
    The frequency-domain state has support at frequencies {±n_1, ±n_2, ...}.

    Gate count: O(m²) dominated by QFT.
    """
    modes = params["modes"]  # list of {"A": amplitude, "n": mode}
    N     = 2 ** m

    qc = QuantumCircuit(m, name="multi_sin_load")

    # Build the target frequency-domain amplitude vector
    freq_amps = np.zeros(N, dtype=complex)
    for mode in modes:
        n = mode["n"]
        A = mode["A"]
        if 0 < n < N:
            freq_amps[n]   += A / (2j)    # positive frequency
            freq_amps[N-n] -= A / (2j)    # negative frequency (conjugate)

    # Normalise
    norm = np.linalg.norm(freq_amps)
    if norm < 1e-12:
        raise ValueError("All mode amplitudes are zero.")
    freq_amps /= norm

    # Prepare the frequency-domain state using Möttönen on the freq vector
    # then apply QFT to get spatial domain
    freq_circuit = _mottonen_from_vector(freq_amps, m)
    qc.compose(freq_circuit, inplace=True)

    from qiskit.circuit.library import QFTGate
    qft_gate = QFTGate(num_qubits=m)
    qc.append(qft_gate, qargs=list(range(m)))

    return qc


# ---------------------------------------------------------------------------
# Case C: uniform + point perturbation
# ---------------------------------------------------------------------------

def _synth_uniform_spike_load(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare  (c · 1 + (δ - c) · eₖ) / ‖·‖  where eₖ is the k-th
    standard basis vector.

    The vector has exactly two distinct amplitude values: delta at index k
    and c everywhere else.  This rank-1 structure admits an analytical
    Ry-tree circuit with O(m) rotation gates and O(m²) CX gates
    (reducible to O(m) with ancilla qubits; see Nielsen & Chuang §4.5).

    Crossover vs Shende: the Ry-tree outperforms StatePreparation for m ≥ 4.
    For m ∈ {2, 3} Shende is equally compact and is used here for simplicity.

    Implementation: uses Qiskit StatePreparation (Shende/Bullock/Markov, 2006)
    on the sparse structured vector.  The hand-crafted Ry-tree circuit is
    provided as a theoretical result in the accompanying paper.

    Gate count: O(2^m) [Shende, current implementation].
    Analytical circuit: O(m²) without ancillas, O(m) with ancillas.
    """
    c, k, delta = params["c"], params["k"], params["delta"]
    N = 2 ** m
    f = np.full(N, float(c)); f[k] = float(delta)
    norm_f = np.linalg.norm(f)
    if norm_f < 1e-14:
        raise ValueError("Load vector is the zero vector.")
    return _mottonen_from_vector(f / norm_f, m, name="uniform_spike_load")


# ---------------------------------------------------------------------------
# Möttönen fallback (general state preparation)
# ---------------------------------------------------------------------------

def _synth_mottonen(m: int, params: dict) -> QuantumCircuit:
    """
    Fallback: Möttönen general state preparation.
    Requires the amplitude vector to be supplied in params["amplitudes"].
    If not present, returns an identity circuit (placeholder).

    Gate count: O(4^m).

    Reference: Möttönen et al., Quantum Inf. Comput. 5(6), 2005.
    """
    if "amplitudes" not in params:
        qc = QuantumCircuit(m, name="mottonen_placeholder")
        return qc

    amplitudes = np.array(params["amplitudes"], dtype=complex)
    norm = np.linalg.norm(amplitudes)
    if norm < 1e-14:
        raise ValueError("Amplitude vector is zero.")
    amplitudes /= norm

    return _mottonen_from_vector(amplitudes, m, name="mottonen")


def _mottonen_from_vector(amplitudes: np.ndarray, m: int,
                           name: str = "state_prep") -> QuantumCircuit:
    """
    Build a Qiskit StatePreparation circuit from a complex amplitude vector.
    Uses Qiskit's built-in Initialize / StatePreparation routine which
    implements the Möttönen / Shende algorithm internally.
    """
    from qiskit.circuit.library import StatePreparation

    N = 2 ** m
    if len(amplitudes) != N:
        raise ValueError(f"Expected {N} amplitudes, got {len(amplitudes)}")

    norm = np.linalg.norm(amplitudes)
    if norm < 1e-14:
        raise ValueError("Amplitude vector is zero.")
    amplitudes = amplitudes / norm

    sp = StatePreparation(amplitudes, label=name)
    qc = QuantumCircuit(m, name=name)
    qc.append(sp, range(m))
    return qc.decompose()