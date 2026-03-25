"""
pyencode.synthesizer
======================
Maps a recognized LoadPattern to an efficient Qiskit QuantumCircuit.

Each synthesis function implements the analytical circuit construction
derived from the structure of the load type.  Gate complexities:

  POINT_LOAD      :  O(m)       — X gates encoding binary(k), global phase
  UNIFORM_LOAD    :  O(m)       — H^{⊗m} followed by global amplitude scale
  STEP_LOAD       :  O(m)       — H on lower qubits, conditional on upper bits
  SINUSOIDAL_LOAD :  O(m²)      — QFT-based, encodes sin(2πnk/N + φ)
  COSINE_LOAD     :  O(m²)      — QFT-based, encodes cos(2πnk/N + φ)

  MULTI_POINT_LOAD             :  O(m · L)      — binary-tree Ry, arbitrary weights
  MULTI_SIN_LOAD          :  O(m²)      — QFT + multi-amplitude encoding
  UNIFORM_SPIKE_LOAD          :  O(m)       — H^{⊗m} + multi-controlled Ry perturbation

The Qiskit StatePreparation fallback is provided for UNKNOWN patterns
and uses the general state-preparation routine shipped with Qiskit.

References
----------
  Shende, Markov, Bullock, IEEE TCAD 25(6), 1000-1010, 2006.
  Gleinig & Hoefler, DAC 2021.
  Coppersmith, "An approximate Fourier transform useful in quantum
    factoring", IBM Research Report RC19642, 1994.
"""

import math
import numpy as np
from typing import Optional

from qiskit import QuantumCircuit, QuantumRegister

from .recognizer import LoadPattern, VectorType


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
        VectorType.DISCRETE:        _synth_point_load,
        VectorType.UNIFORM:      _synth_uniform_load,
        VectorType.STEP:         _synth_step_load,
        VectorType.SQUARE:       _synth_square_load,
        VectorType.SINE:   _synth_sinusoidal,
        VectorType.COSINE:       _synth_cosine,
        VectorType.MULTI_DISCRETE:  _synth_disjoint_point_load,
        VectorType.MULTI_SINE:    _synth_multi_sin_load,
        VectorType.UNIFORM_SPIKE: _synth_uniform_spike_load,
        VectorType.UNKNOWN:           _synth_qiskit_fallback,
        # New unified types (paper API) — delegate to existing synthesizers
        VectorType.SPARSE:            _synth_sparse,
        VectorType.FOURIER:           _synth_fourier,
    }

    fn = dispatch.get(pattern.load_type, _synth_qiskit_fallback)
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
    Prepare (1/√M) Σ_{j=0}^{M-1} |j⟩  for any M in [1, 2^m].

    Algorithm
    ---------
    Implements Algorithm 1 of Shukla & Vedula, *Quantum Information
    Processing* 23:38, 2024.  DOI: 10.1007/s11128-024-04258-4.

    Write M in binary as M = 2^{l_0} + 2^{l_1} + ... + 2^{l_k}
    with 0 ≤ l_0 < l_1 < ... < l_k ≤ m−1 (positions of 1-bits, LSB first).

    Steps:
      1. X on qubits l_1, ..., l_k  (flip the upper boundary bits)
      2. H on qubits 0..l_0−1       (uniform over lowest block)
      3. R_Y(θ_0) on q_{l_1}        (split amplitude: lower / upper)
      4. CH(ctrl=q_{l_1}, active-low) on q_{l_0}..q_{l_1−1}
      5. Repeat with CRY + CH for each subsequent 1-bit

    Gate count: exactly Σ l_j gates = O(m log m) worst case, O(m) typical.
    No ancilla qubits. No multi-controlled gates. Each gate has at most
    one control qubit.

    Reference
    ---------
    Shukla, A. & Vedula, P. (2024). An efficient quantum algorithm for
    preparation of uniform quantum superposition states.
    Quantum Inf. Process. 23, 38.
    """
    M = params["k_s"]
    N = 2 ** m
    if M <= 0 or M > N:
        raise ValueError(f"k_s={M} out of range [1, {N}]")

    qc = QuantumCircuit(m, name="step_load")

    # Power-of-2 special case: pure H gates
    if M & (M - 1) == 0:
        p = int(round(math.log2(M)))
        for q in range(p):
            qc.h(q)
        return qc

    # Find l_j: positions of 1-bits in M, ascending (LSB first)
    ls = [i for i in range(m) if (M >> i) & 1]
    k  = len(ls) - 1   # index of highest 1-bit entry

    # Step 4: X on qubits l_1 ... l_k (all 1-bit positions except the lowest)
    for lj in ls[1:]:
        qc.x(lj)

    # Step 5: M_0 = 2^{l_0}
    M_prev = 2 ** ls[0]

    # Steps 6-7: if l_0 > 0, H on qubits 0 .. l_0-1
    for q in range(ls[0]):
        qc.h(q)

    # Step 8: R_Y(θ_0) on q_{l_1}, θ_0 = -2 arccos(sqrt(M_0 / M))
    theta0 = -2.0 * math.acos(math.sqrt(M_prev / M))
    qc.ry(theta0, ls[1])

    # Step 9: controlled-H on q_{l_0} .. q_{l_1 - 1}, conditioned on q_{l_1} = 0
    # (active-low: X before and after CH)
    for q in range(ls[0], ls[1]):
        qc.x(ls[1])
        qc.ch(ls[1], q)
        qc.x(ls[1])

    # Steps 10-13: loop over remaining 1-bits
    for idx in range(1, k):
        lm      = ls[idx]       # current bit position
        lm_next = ls[idx + 1]   # next bit position

        # M_{m} = M_{m-1} + 2^{l_m}  (already computed as M_prev + 2^{lm})
        M_cur = M_prev + 2 ** lm

        # Step 11: CRY(θ_m) on q_{l_{m+1}}, conditioned on q_{l_m} = 0
        theta_m = -2.0 * math.acos(math.sqrt(2 ** lm / (M - M_prev)))
        qc.x(lm)
        qc.cry(theta_m, lm, lm_next)
        qc.x(lm)

        # Step 12: controlled-H on q_{l_m} .. q_{l_{m+1} - 1},
        # conditioned on q_{l_{m+1}} = 0  (active-low)
        for q in range(lm, lm_next):
            qc.x(lm_next)
            qc.ch(lm_next, q)
            qc.x(lm_next)

        # Step 13: update M_prev
        M_prev = M_cur

    return qc








# ---------------------------------------------------------------------------
# Square load  f[k1:k2] = c
# ---------------------------------------------------------------------------

def _synth_square_load(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare (1/√w) Σ_{j=k1}^{k2-1} |j⟩  where w = k2 - k1.

    Strategy
    --------
    A square block [k1, k2) is the difference of two step prefixes:

        1[k1 ≤ j < k2] = 1[j < k2] - 1[j < k1]

    In amplitude encoding:

        √w |ψ_sq⟩ = √k2 |ψ_step(k2)⟩ - √k1 |ψ_step(k1)⟩

    We therefore build |ψ_sq⟩ by combining two STEP circuits via a
    single Ry rotation on an ancilla-free register:

      1. Ry(α) on qubit m-1 splits amplitude: cos(α/2)→lower, sin(α/2)→upper
         where α = 2 arccos(√(k1/w_total)), w_total = k1+k2... 

    Simpler direct construction
    ---------------------------
    We use the same top-down controlled-Ry recursion as STEP, but now
    the "active window" starts at k1 instead of 0.

    At each qubit level (MSB first), we track the window [lo, hi) of
    indices still to be covered.  Qubits outside the window stay |0⟩;
    qubits at the boundary get a (controlled) Ry to split correctly.

    Gate count: O(m) — at most one (C)Ry per qubit level.
    """
    k1 = params["k1"]
    k2 = params["k2"]
    N  = 2 ** m
    w  = k2 - k1

    qc = QuantumCircuit(m, name="square_load")

    # Special case: w-aligned power-of-2 block → X gates + H gates
    if (w & (w - 1)) == 0 and (k1 % w == 0):
        p = int(round(math.log2(w)))
        for bit in range(m):
            if (k1 >> bit) & 1:
                qc.x(bit)
        for q in range(p):
            qc.h(q)
        return qc

    # General case: top-down window-tracking Ry decomposition
    _square_ry_decompose(qc, k1, k2, m - 1, ctrl_qubits=[], ctrl_vals=[])
    return qc


def _apply_controlled_ry(qc: QuantumCircuit,
                          theta: float,
                          ctrl_qubits: list,
                          ctrl_vals: list,
                          target: int) -> None:
    """
    Apply Ry(theta) on `target`, controlled on ctrl_qubits == ctrl_vals.
    Active-low controls are handled by flanking X gates.
    Uses a single CRy for one control, mcry for multiple.
    """
    if not ctrl_qubits:
        qc.ry(theta, target)
        return
    flip = [q for q, v in zip(ctrl_qubits, ctrl_vals) if v == 0]
    for q in flip:
        qc.x(q)
    if len(ctrl_qubits) == 1:
        qc.cry(theta, ctrl_qubits[0], target)
    else:
        qc.mcry(theta, ctrl_qubits, target)
    for q in flip:
        qc.x(q)


def _square_ry_decompose(qc: QuantumCircuit,
                          lo: int, hi: int,
                          qubit: int,
                          ctrl_qubits: list,
                          ctrl_vals: list) -> None:
    """
    Recursively prepare the uniform superposition over indices [lo, hi)
    on qubits 0..qubit, conditioned on ctrl_qubits == ctrl_vals.

    At each level we split the address space into lower and upper halves.
    The window [lo, hi) may fall entirely in one half, or straddle both.
    Gate count: O(m) (C)Ry gates at the circuit level.
    """
    slots = hi - lo
    if slots <= 0 or qubit < 0:
        return

    half = 1 << qubit

    block_size = 1 << (qubit + 1)
    boundary   = (lo // block_size) * block_size + half

    lower_lo = lo;          lower_hi = min(hi, boundary)
    upper_lo = max(lo, boundary);   upper_hi = hi

    lower_slots = max(0, lower_hi - lower_lo)
    upper_slots = max(0, upper_hi - upper_lo)

    if upper_slots == 0:
        if qubit > 0:
            _square_ry_decompose(qc, lower_lo, lower_hi, qubit - 1,
                                 ctrl_qubits, ctrl_vals)
        return

    if lower_slots == 0:
        _apply_controlled_ry(qc, math.pi, ctrl_qubits, ctrl_vals, qubit)
        if qubit > 0:
            _square_ry_decompose(qc, upper_lo, upper_hi, qubit - 1,
                                 ctrl_qubits + [qubit], ctrl_vals + [1])
        return

    theta = 2.0 * math.acos(math.sqrt(lower_slots / slots))
    _apply_controlled_ry(qc, theta, ctrl_qubits, ctrl_vals, qubit)

    if qubit > 0:
        _square_ry_decompose(qc, lower_lo, lower_hi, qubit - 1,
                             ctrl_qubits + [qubit], ctrl_vals + [0])
        _square_ry_decompose(qc, upper_lo, upper_hi, qubit - 1,
                             ctrl_qubits + [qubit], ctrl_vals + [1])


# ---------------------------------------------------------------------------
# Sinusoidal load  f = A * sin(n * pi * x / L)
# ---------------------------------------------------------------------------

def _synth_sinusoidal(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare (1/‖f‖) Σ_k sin(2π n k / N + φ) |k⟩ using the quantum Fourier
    transform.

    The DFT of sin(2π n k / N + φ) is a pair of delta functions at
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
# Case A: sparse point loads — Gleinig-Hoefler algorithm
# ---------------------------------------------------------------------------

def _synth_disjoint_point_load(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare a weighted superposition of L ≥ 1 point loads:

        |ψ⟩ = (1/‖a‖) Σ_i a_i |k_i⟩,   a_i = |P_i|

    Algorithm: Gleinig & Hoefler, DAC 2021.

    The circuit is built by inverting the "disentangling" direction:
    Algorithm 2 repeatedly calls Algorithm 1, which reduces an L-sparse
    state to an (L-1)-sparse state using O(n) CNOT gates per call.
    Total gate count: O(L·n).

    This is correct for ALL index patterns — including clustered indices
    (e.g. k=1,4,7 in N=2^20) that defeated the earlier Ry-tree approach.

    References
    ----------
    Gleinig & Hoefler, "An Efficient Algorithm for Sparse Quantum State
      Preparation", DAC 2021.  O(|S|·n) CNOT, O(|S| log|S| + n) 1-qubit.
    """
    loads = params["loads"]
    L     = len(loads)

    # Normalise amplitudes (use absolute values; signs absorbed into phase)
    amps = np.array([abs(load["P"]) for load in loads], dtype=float)
    norm = np.linalg.norm(amps)
    if norm < 1e-14:
        raise ValueError("All point-load magnitudes are zero.")
    amps = amps / norm

    indices = [int(load["k"]) for load in loads]

    if L == 1:
        qc = QuantumCircuit(m, name="sparse_load")
        for bit in range(m):
            if (indices[0] >> bit) & 1:
                qc.x(bit)
        return qc

    # Build circuit via Gleinig Algorithm 2
    qc = QuantumCircuit(m, name="sparse_load")
    _gleinig_encode(qc, dict(zip(
        [_int_to_bits(k, m) for k in indices],
        amps.tolist()
    )), m)
    return qc


# ---------------------------------------------------------------------------
# Gleinig-Hoefler implementation
# ---------------------------------------------------------------------------

def _int_to_bits(k: int, n: int) -> tuple:
    """Convert integer k to an n-bit tuple (LSB first)."""
    return tuple((k >> i) & 1 for i in range(n))


def _bits_to_int(bits: tuple) -> int:
    """Convert n-bit tuple (LSB first) back to integer."""
    return sum(b << i for i, b in enumerate(bits))


def _gleinig_encode(qc: QuantumCircuit,
                    state: dict,
                    n: int) -> None:
    """
    Algorithm 2 (Gleinig & Hoefler, DAC 2021).

    Takes a sparse quantum state represented as
        state: dict mapping n-bit tuple -> amplitude (real, normalized)
    and appends gates to qc that prepare that state from |0^n>.

    Works by repeatedly calling _gleinig_reduce (Algorithm 1) which
    merges two basis states into one, until a single basis state remains,
    then adds X gates to map that state to |0^n>.  The circuit is then
    inverted so it maps |0^n> -> target state.
    """
    gates_forward = []   # list of (gate_type, args) to be inverted

    current_state = dict(state)

    while len(current_state) > 1:
        new_gates = _gleinig_reduce(current_state, n)
        gates_forward.extend(new_gates)
        # Apply each gate classically to track state transformation
        for gate in new_gates:
            _apply_gate_to_state(current_state, gate, n)

    # Now current_state has one basis state: add X gates to map it to |0^n>
    assert len(current_state) == 1
    remaining_bits = list(current_state.keys())[0]
    for i, b in enumerate(remaining_bits):
        if b:
            gates_forward.append(('x', i))

    # Invert: reverse gate list and invert each gate
    for gate in reversed(gates_forward):
        gtype = gate[0]
        if gtype == 'x':
            qc.x(gate[1])
        elif gtype == 'cx':
            qc.cx(gate[1], gate[2])
        elif gtype == 'cry':
            # Inverse of CRy(theta) is CRy(-theta)
            # ctrl == -1 means unconditional Ry
            if gate[2] == -1:
                qc.ry(-gate[1], gate[3])
            else:
                qc.cry(-gate[1], gate[2], gate[3])
        elif gtype == 'mcry':
            # Inverse of MCRy(theta) is MCRy(-theta)
            _mcry(qc, -gate[1], gate[2], gate[3], gate[4])


def _gleinig_reduce(state: dict, n: int) -> list:
    """
    Algorithm 1 (Gleinig & Hoefler, DAC 2021).

    Given a sparse state dict (bits_tuple -> amplitude), find two basis
    states x1, x2 that can be merged using O(n) CNOT gates and one
    multi-controlled Ry.  Returns a list of gates as (type, *args) tuples.

    After applying the returned gates to `state`, the state will have
    one fewer basis state.
    """
    S = list(state.keys())
    gates = []

    # --- Find x1, x2 and the distinguishing control bits ---
    # Use the two-WHILE-loop procedure from Algorithm 1.
    # dif_qubits/dif_vals identify the unique path to x1 in the tree.

    T = list(S)
    dif_qubits = []
    dif_vals   = []

    # First WHILE: narrow T to a single element x1
    while len(T) > 1:
        # Find qubit b that splits T as unevenly as possible (neither set empty)
        best_b = None
        best_imbalance = -1
        for b in range(n):
            t0 = [x for x in T if x[b] == 0]
            t1 = [x for x in T if x[b] == 1]
            if t0 and t1:
                imbalance = abs(len(t0) - len(t1))
                if imbalance > best_imbalance:
                    best_imbalance = imbalance
                    best_b = b
                    best_T0, best_T1 = t0, t1

        b = best_b
        dif_qubits.append(b)
        if len(best_T0) < len(best_T1):
            T = best_T0
            dif_vals.append(0)
        else:
            T = best_T1
            dif_vals.append(1)

    x1 = T[0]

    # Pop the last entry to get 'dif' qubit
    dif = dif_qubits.pop()
    dif_vals.pop()

    # Second WHILE: find x2 — the unique other element sharing the path prefix
    T2 = [x for x in S if x != x1
          and all(x[dif_qubits[i]] == dif_vals[i] for i in range(len(dif_qubits)))]

    while len(T2) > 1:
        best_b = None
        best_imbalance = -1
        for b in range(n):
            t0 = [x for x in T2 if x[b] == 0]
            t1 = [x for x in T2 if x[b] == 1]
            if t0 and t1:
                imbalance = abs(len(t0) - len(t1))
                if imbalance > best_imbalance:
                    best_imbalance = imbalance
                    best_b = b
                    best_T0, best_T1 = t0, t1

        b = best_b
        dif_qubits.append(b)
        if len(best_T0) < len(best_T1):
            T2 = best_T0
            dif_vals.append(0)
        else:
            T2 = best_T1
            dif_vals.append(1)

    x2 = T2[0]

    # --- Ensure x1[dif] == 1 (swap labels if needed) ---
    if x1[dif] != 1:
        x1, x2 = x2, x1

    # --- CNOT gates: make x1 and x2 differ only on qubit `dif` ---
    # For each bit b != dif where x1[b] != x2[b]:
    #   add CNOT(dif -> b), which flips x2[b] (since x2[dif]=0, CNOT targets b)
    for b in range(n):
        if b != dif and x1[b] != x2[b]:
            gates.append(('cx', dif, b))

    # --- NOT gates on dif_qubits to set control state to all-1 ---
    flip_bits = [dif_qubits[i] for i in range(len(dif_qubits))
                 if x2[dif_qubits[i]] != 1]
    for b in flip_bits:
        gates.append(('x', b))

    # --- Multi-controlled Ry to merge x1 and x2 ---
    # G gate: maps  cx1|1> + cx2|0>  ->  e^{i*lambda}|0>
    # which is CRy(2*arccos(cx2 / norm)) controlled on dif_qubits=1
    # (after the NOT gates above, control is all-1 for both x1 and x2)
    cx1 = state[x1]
    cx2 = state[x2]
    norm_pair = math.sqrt(cx1**2 + cx2**2)
    # We need Ry(theta) to map  cx1|1> + cx2|0>  ->  norm|0>
    # Standard Ry: |0>->cos(θ/2)|0>+sin(θ/2)|1>, |1>->-sin(θ/2)|0>+cos(θ/2)|1>
    # Requiring the |1> coefficient to vanish: cx1*cos(θ/2) + cx2*sin(θ/2) = 0
    # => theta = -2*arctan(cx1/cx2)  [negative angle]
    theta = -2.0 * math.atan2(cx1, cx2)

    if not dif_qubits:
        gates.append(('cry', theta, -1, dif))  # unconditional Ry (no controls)
    elif len(dif_qubits) == 1:
        gates.append(('cry', theta, dif_qubits[0], dif))
    else:
        gates.append(('mcry', theta, list(dif_qubits), list(dif_vals), dif))

    # --- Unflip NOT gates ---
    for b in flip_bits:
        gates.append(('x', b))

    return gates


def _apply_gate_to_state(state: dict, gate: tuple, n: int) -> None:
    """
    Apply a gate (classically) to the sparse state dict in-place.
    Used to track how the state transforms as gates are added.
    """
    gtype = gate[0]
    keys = list(state.keys())

    if gtype == 'x':
        b = gate[1]
        new_state = {}
        for bits, amp in state.items():
            new_bits = list(bits)
            new_bits[b] ^= 1
            new_state[tuple(new_bits)] = amp
        state.clear()
        state.update(new_state)

    elif gtype == 'cx':
        ctrl, tgt = gate[1], gate[2]
        new_state = {}
        for bits, amp in state.items():
            new_bits = list(bits)
            if new_bits[ctrl] == 1:
                new_bits[tgt] ^= 1
            new_state[tuple(new_bits)] = amp
        state.clear()
        state.update(new_state)

    elif gtype == 'cry':
        theta, ctrl, tgt = gate[1], gate[2], gate[3]
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        new_state = dict(state)
        processed = set()
        for bits in list(state.keys()):
            if bits in processed:
                continue
            # Only act on basis states where control is satisfied
            if ctrl != -1 and bits[ctrl] != 1:
                continue
            # Build the partner (same bits except tgt flipped)
            partner = list(bits); partner[tgt] ^= 1; partner = tuple(partner)
            processed.add(bits)
            processed.add(partner)
            # Identify which is the |0> and which is the |1> component on tgt
            if bits[tgt] == 0:
                b0, b1 = bits, partner
            else:
                b0, b1 = partner, bits
            a0 = state.get(b0, 0.0)   # amplitude of tgt=0 component
            a1 = state.get(b1, 0.0)   # amplitude of tgt=1 component
            # Ry(theta): |0>->c|0>+s|1>,  |1>->-s|0>+c|1>
            new_a0 = c * a0 - s * a1
            new_a1 = s * a0 + c * a1
            if abs(new_a0) > 1e-14:
                new_state[b0] = new_a0
            else:
                new_state.pop(b0, None)
            if abs(new_a1) > 1e-14:
                new_state[b1] = new_a1
            else:
                new_state.pop(b1, None)
        state.clear()
        state.update(new_state)

    elif gtype == 'mcry':
        theta, ctrl_list, ctrl_vals, tgt = gate[1], gate[2], gate[3], gate[4]
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        new_state = dict(state)
        processed = set()
        for bits in list(state.keys()):
            if bits in processed:
                continue
            # All controls must be 1 (X-flanking is applied separately as 'x' gates)
            if not all(bits[q] == 1 for q in ctrl_list):
                continue
            partner = list(bits); partner[tgt] ^= 1; partner = tuple(partner)
            processed.add(bits)
            processed.add(partner)
            if bits[tgt] == 0:
                b0, b1 = bits, partner
            else:
                b0, b1 = partner, bits
            a0 = state.get(b0, 0.0)
            a1 = state.get(b1, 0.0)
            new_a0 = c * a0 - s * a1
            new_a1 = s * a0 + c * a1
            if abs(new_a0) > 1e-14:
                new_state[b0] = new_a0
            else:
                new_state.pop(b0, None)
            if abs(new_a1) > 1e-14:
                new_state[b1] = new_a1
            else:
                new_state.pop(b1, None)
        state.clear()
        state.update(new_state)


def _mcry(qc: QuantumCircuit, theta: float,
          ctrl_qubits: list, ctrl_vals: list, target: int) -> None:
    """Multi-controlled Ry with active-low controls via X flanking."""
    flip = [q for q, v in zip(ctrl_qubits, ctrl_vals) if v == 0]
    for q in flip:
        qc.x(q)
    if len(ctrl_qubits) == 1:
        qc.cry(theta, ctrl_qubits[0], target)
    else:
        qc.mcry(theta, ctrl_qubits, target)
    for q in flip:
        qc.x(q)


def _controlled_x(qc, ctrl_qubits, ctrl_vals, target):
    """Multi-controlled X (Toffoli) with active-low controls via X flanking."""
    flip = [q for q, v in zip(ctrl_qubits, ctrl_vals) if v == 0]
    for q in flip:
        qc.x(q)
    if len(ctrl_qubits) == 0:
        qc.x(target)
    elif len(ctrl_qubits) == 1:
        qc.cx(ctrl_qubits[0], target)
    else:
        qc.mcx(ctrl_qubits, target)
    for q in flip:
        qc.x(q)


# ---------------------------------------------------------------------------
# Case B: sum of sinusoidal modes
# ---------------------------------------------------------------------------

def _synth_multi_sin_load(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare a superposition of multiple sinusoidal modes.

    Strategy: the DFT of sum_t A_t sin(2pi n_t k/N) has 2T nonzero
    entries at frequencies {n_t, N-n_t} with equal magnitudes |A_t|/2
    and phases -pi/2 (pos freq) and +pi/2 (neg freq).

    1. Use the Ry-tree point load synthesizer to prepare the
       magnitude distribution over the 2T frequency entries.
    2. Apply Z on the MSB to encode the relative phase between
       positive (MSB=0) and negative (MSB=1) frequencies.
    3. Apply QFT to transform frequency -> spatial domain.

    Gate count: O(T*m) for Ry tree + O(m^2) for QFT = O(m^2).

    Requires all mode frequencies n_t < N/2 (standard assumption).
    """
    modes = params["modes"]  # list of {"A": amplitude, "n": mode}
    N     = 2 ** m

    qc = QuantumCircuit(m, name="multi_sin_load")

    # Build frequency-domain point loads (magnitudes only)
    freq_loads = []
    for mode in modes:
        n = mode["n"]
        A = abs(mode["A"])
        if 0 < n < N:
            freq_loads.append({"k": n,     "P": A})
            freq_loads.append({"k": N - n, "P": A})

    if not freq_loads:
        raise ValueError("All mode amplitudes are zero.")

    # Step 1: Ry-tree to prepare magnitude distribution
    freq_circuit = _synth_disjoint_point_load(m, {"loads": freq_loads})
    qc.compose(freq_circuit, inplace=True)

    # Step 2: Phase correction — Z on MSB gives relative phase pi
    # between positive-freq entries (MSB=0, n_t < N/2) and
    # negative-freq entries (MSB=1, N-n_t >= N/2).
    # This encodes the sin antisymmetry: (|n> - |N-n>)/sqrt(2).
    qc.z(m - 1)

    # Step 3: QFT to transform frequency -> spatial domain
    from qiskit.circuit.library import QFTGate
    qc.append(QFTGate(num_qubits=m), qargs=list(range(m)))

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

    Crossover vs Qiskit StatePreparation: the Ry-tree outperforms it for m ≥ 4.
    For m ∈ {2, 3} Qiskit StatePreparation is equally compact.

    Implementation: uses Qiskit StatePreparation (Shende/Bullock/Markov, 2006)
    on the sparse structured vector.

    Gate count: O(2^m) [current implementation].
    Analytical circuit: O(m²) without ancillas, O(m) with ancillas.
    """
    c, k, delta = params["c"], params["k"], params["delta"]
    N = 2 ** m
    f = np.full(N, float(c)); f[k] = float(delta)
    norm_f = np.linalg.norm(f)
    if norm_f < 1e-14:
        raise ValueError("Load vector is the zero vector.")
    return _state_preparation_from_vector(f / norm_f, m, name="uniform_spike_load")


# ---------------------------------------------------------------------------
# Qiskit StatePreparation fallback (general state preparation)
# ---------------------------------------------------------------------------

def _synth_qiskit_fallback(m: int, params: dict) -> QuantumCircuit:
    """
    Fallback: Qiskit StatePreparation for UNKNOWN patterns.
    Requires the amplitude vector to be supplied in params["amplitudes"].
    If not present, returns an identity circuit (placeholder).

    Gate count: O(2^m) — exponential in the number of qubits.

    Reference: Shende, Markov, Bullock, IEEE TCAD 25(6), 2006.
    """
    if "amplitudes" not in params:
        qc = QuantumCircuit(m, name="mottonen_placeholder")
        return qc

    amplitudes = np.array(params["amplitudes"], dtype=complex)
    norm = np.linalg.norm(amplitudes)
    if norm < 1e-14:
        raise ValueError("Amplitude vector is zero.")
    amplitudes /= norm

    return _state_preparation_from_vector(amplitudes, m, name="mottonen")


def _state_preparation_from_vector(amplitudes: np.ndarray, m: int,
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

# ---------------------------------------------------------------------------
# New unified synthesizers — thin wrappers for the paper API
# ---------------------------------------------------------------------------

def _synth_sparse(m: int, params: dict) -> QuantumCircuit:
    """
    SPARSE: delegate to the Gleinig-Hoefler disjoint-point-load synthesizer.

    For s=1 this reduces to _synth_point_load (X gates only).
    For s>1 the full Gleinig-Hoefler O(s*m) construction is used.
    """
    loads = params["loads"]
    if len(loads) == 1:
        return _synth_point_load(m, loads[0])
    return _synth_disjoint_point_load(m, params)


def _synth_fourier(m: int, params: dict) -> QuantumCircuit:
    """
    FOURIER: delegate to the multi-sine QFT synthesizer.

    For T=1 mode with phi=0 this is identical to _synth_sinusoidal.
    For T=1 with phi=pi/2 it matches _synth_cosine.
    For T>1 modes it uses _synth_multi_sin_load.
    All paths share the same O(m^2) inverse-QFT pipeline.
    """
    modes = params["modes"]
    if len(modes) == 1:
        mode = modes[0]
        phi = mode.get("phi", 0.0)
        # Delegate to single-mode synthesizers
        p = {"n": mode["n"], "A": mode["A"], "phi": phi}
        return _synth_sinusoidal(m, p)
    return _synth_multi_sin_load(m, params)
