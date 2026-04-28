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
import cmath
import numpy as np
from typing import Optional

from qiskit import QuantumCircuit, QuantumRegister

from .recognizer import LoadPattern, PatternKind


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
        PatternKind.STEP:         _synth_step_load,
        PatternKind.SQUARE:       _synth_square_load,
        PatternKind.UNKNOWN:           _synth_qiskit_fallback,
        # New unified types (paper API) — delegate to existing synthesizers
        PatternKind.WALSH:             _synth_walsh,
        PatternKind.SPARSE:            _synth_sparse,
        PatternKind.FOURIER:           _synth_fourier,
        PatternKind.GEOMETRIC:         _synth_geometric,
        PatternKind.HAMMING:          _synth_hamming,
        PatternKind.STAIRCASE:         _synth_staircase,
        PatternKind.DICKE:             _synth_dicke,
        PatternKind.POLYNOMIAL:        _synth_polynomial,
    }

    fn = dispatch.get(pattern.kind, _synth_qiskit_fallback)
    qc = fn(m, pattern.params)

    # STEP / SQUARE: the dedicated synthesizers ignore the leading
    # constant `c` because it only affects normalization; for complex
    # `c` we must still record arg(c) as a global phase so that the
    # circuit composes correctly under SUM / PARTITION / TENSOR.  For
    # real positive c (the original behaviour) arg(c) = 0 and no phase
    # is added.
    if pattern.kind in (PatternKind.STEP, PatternKind.SQUARE):
        c = pattern.params.get("c", 1.0)
        arg_c = cmath.phase(complex(c))
        if abs(arg_c) > 1e-14:
            qc.global_phase += arg_c

    return qc


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
# Step load  f[:k_e] = c
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
    M = params["k_e"]
    N = 2 ** m
    if M <= 0 or M > N:
        raise ValueError(f"k_e={M} out of range [1, {N}]")

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
# Square load  f[k_s:k_e] = c
# ---------------------------------------------------------------------------

def _synth_square_load(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare (1/√w) Σ_{j=k_s}^{k_e-1} |j⟩  where w = k_e - k_s.

    Construction
    ------------
    Uses the identity  [k_s, k_e)  =  shift of  [0, w)  by constant k_s:

        |ψ_sq⟩ = ADD(k_s) · STEP(w)

    where STEP(w) prepares the uniform superposition over [0, w) and
    ADD(k_s) is the Draper QFT-based adder for the classical constant k_s.

    Steps
    -----
    1. STEP(w) on m qubits: O(m) gates.
    2. ADD(k_s): QFT + O(m) phase gates + QFT†.
       The QFT dominates at O(m²), making the total circuit O(m²).

    Special cases
    -------------
    k_s = 0 : reduces to plain STEP(k_e) with no adder — O(m) total.
    Aligned power-of-2 block : X gates + H gates — O(m) total.

    Gate count: O(m) for STEP + O(m²) for adder = O(m²) in general.
    For k_s = 0: O(m).  For power-of-2-aligned blocks: O(m).

    Note on the paper's O(m) claim
    --------------------------------
    The paper describes an ancilla-based LCU decomposition as an O(m)
    alternative.  That construction produces a (m+1)-qubit circuit where
    the ancilla qubit is entangled with the data register and cannot be
    deterministically uncomputed; in practice the STEP+adder approach
    here is simpler, correct, and ancilla-free.
    """
    k_s = params["k_s"]
    k_e = params["k_e"]
    N  = 2 ** m
    w  = k_e - k_s

    if w <= 0:
        raise ValueError(f"SQUARE requires k_s < k_e, got k_s={k_s} k_e={k_e}")

    # ── Special case: k_s == 0 → plain STEP, no adder ─────────────────────
    if k_s == 0:
        return _synth_step_load(m, {"k_e": k_e})

    # ── Special case: aligned power-of-2 block → X + H, no adder ─────────
    if (w & (w - 1)) == 0 and (k_s % w == 0):
        p = int(round(math.log2(w)))
        qc = QuantumCircuit(m, name="square_load")
        for bit in range(m):
            if (k_s >> bit) & 1:
                qc.x(bit)
        for q in range(p):
            qc.h(q)
        return qc

    # ── General case: STEP(w) + Draper QFT constant adder(k_s) ────────────
    step = _synth_step_load(m, {"k_e": w})
    adder = _draper_add_const(m, k_s)
    qc = step.compose(adder)
    qc.name = "square_load"
    return qc


def _draper_add_const(m: int, k: int) -> QuantumCircuit:
    """
    Add classical constant k mod 2^m to an m-qubit register in place.

    Uses the Draper QFT-based adder: QFT, O(m) single-qubit phase gates,
    then QFT†.  No ancilla, no CX gates in the phase stage.

    Gate count: O(m²) dominated by the QFT.

    Reference: Draper, 'Addition on a Quantum Computer', arXiv:quant-ph/0008033.
    """
    from qiskit.circuit.library import QFTGate

    k = k % (2 ** m)
    qc = QuantumCircuit(m, name=f"add_{k}")
    if k == 0:
        return qc

    qc.append(QFTGate(m), range(m))
    for j in range(m):
        angle = 2.0 * math.pi * k / (2 ** (m - j))
        qc.p(angle, j)
    qc.append(QFTGate(m).inverse(), range(m))
    return qc


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


def _synth_disjoint_point_load_signed(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare a weighted superposition of L ≥ 2 point loads with SIGNED
    or COMPLEX amplitudes:

        |ψ⟩ = (1/‖a‖) Σ_i a_i |k_i⟩,   a_i ∈ ℂ

    Same Gleinig-Hoefler construction as _synth_disjoint_point_load, but
    carries signs/phases through the pairwise reduction directly.  When
    every a_i is real, the merge step uses the existing signed
    θ = -2·atan2(c_x1, c_x2) parameterisation (no extra gates).  When
    any a_i has a non-zero imaginary part, the merge inserts a
    controlled-phase gate to strip the relative phase between the two
    amplitudes, then performs a real magnitude merge.

    This eliminates the need for a post-hoc multi-controlled-Z
    phase-flip pass — critical for POLYNOMIAL, where most Walsh
    coefficients are negative and the phase-flip pass would otherwise
    dominate the transpiled gate count.
    """
    loads = params["loads"]
    L     = len(loads)
    assert L >= 2, "Use _synth_point_load for L=1"

    # Normalise amplitudes keeping signs/phases intact.  Use complex
    # storage so the dtype is preserved through the Gleinig tree even
    # for real inputs (the merge step then routes via the real path).
    raw = np.array([complex(load["P"]) for load in loads], dtype=complex)
    norm = np.linalg.norm(raw)
    if norm < 1e-14:
        raise ValueError("All point-load magnitudes are zero.")
    amps = raw / norm

    indices = [int(load["k"]) for load in loads]

    qc = QuantumCircuit(m, name="sparse_load_signed")
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
    Algorithm 2 (Gleinig & Hoefler, DAC 2021), generalised to complex
    amplitudes.

    Takes a sparse quantum state represented as
        state: dict mapping n-bit tuple -> amplitude (real or complex,
                                                       normalised)
    and appends gates to qc that prepare that state from |0^n⟩.

    Works by repeatedly calling _gleinig_reduce (Algorithm 1) which
    merges two basis states into one, until a single basis state remains,
    then adds X gates to map that state to |0^n⟩.  The circuit is then
    inverted so it maps |0^n⟩ → target state.

    For complex amplitudes the surviving 1-sparse amplitude after L-1
    merges may have a non-trivial phase; this is absorbed into
    ``qc.global_phase`` so that the prepared state matches the target
    exactly (not merely up to a global phase).
    """
    gates_forward = []   # list of (gate_type, args) to be inverted

    # Cast to complex storage so the merge step can detect non-real
    # amplitudes uniformly.  For purely real inputs the per-merge
    # routing falls through to the existing signed-atan2 path.
    current_state = {bits: complex(amp) for bits, amp in state.items()}

    while len(current_state) > 1:
        new_gates = _gleinig_reduce(current_state, n)
        gates_forward.extend(new_gates)
        # Apply each gate classically to track state transformation
        for gate in new_gates:
            _apply_gate_to_state(current_state, gate, n)

    # Now current_state has one basis state: add X gates to map it to |0^n>
    assert len(current_state) == 1
    remaining_bits, remaining_amp = next(iter(current_state.items()))
    for i, b in enumerate(remaining_bits):
        if b:
            gates_forward.append(('x', i))

    # Track global phase so the prepared state equals the target exactly.
    # The forward gate sequence maps |ψ⟩ → remaining_amp · |0^n⟩.
    # The inverse therefore prepares (1/remaining_amp) · |ψ⟩ from |0^n⟩;
    # we add arg(remaining_amp) to the circuit's global phase to cancel
    # the residual factor.  For real positive remaining_amp this is a
    # no-op (preserving existing real-only behaviour); for negative real
    # it adds π exactly, matching the explicit handling already used by
    # _synth_sparse for the L=1 case.
    phase_correction = cmath.phase(remaining_amp)
    if abs(phase_correction) > 1e-14:
        qc.global_phase += phase_correction

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
        elif gtype == 'p':
            # ('p', delta, target) — uncontrolled phase shift on |1⟩.
            # Inverse: P(-delta).
            qc.p(-gate[1], gate[2])
        elif gtype == 'cp':
            # ('cp', delta, ctrl, target) — controlled phase shift.
            # Inverse: CP(-delta).
            qc.cp(-gate[1], gate[2], gate[3])
        elif gtype == 'mcp':
            # ('mcp', delta, ctrl_list, ctrl_vals, target) — multi-
            # controlled phase shift.  Inverse: MCP(-delta).  The
            # ctrl_vals are honoured by the X-flanking already emitted
            # alongside this gate (same pattern as 'mcry'); mcp itself
            # treats every control as active-high.
            qc.mcp(-gate[1], gate[2], gate[4])


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

    # --- Multi-controlled merge to combine x1 and x2 ---------------------
    # G gate: maps  cx1|1⟩ + cx2|0⟩  →  ‖(cx1, cx2)‖ · e^{i·arg(cx2)} |0⟩
    # on the |dif⟩ subspace where dif_qubits are all-1 (achieved by the X
    # flanking emitted above).
    #
    # Real-only path  (no extra gates, preserves backward-compat counts):
    #   Use the standard signed parameterisation
    #       θ = -2 · atan2(cx1, cx2)
    #   so that one CRy(θ) on the dif qubit absorbs both the magnitude
    #   merge and any sign flip via atan2's sign handling.
    #
    # Complex path  (one extra controlled-phase per merge):
    #   First strip the relative phase δ = arg(cx1) − arg(cx2) from the
    #   |1⟩-branch via a (multi-)controlled phase shift CP(-δ) on dif.
    #   After the strip, both branches share phase arg(cx2) and the
    #   magnitude merge reduces to the real case
    #       θ = -2 · atan2(|cx1|, |cx2|).
    #
    # Decomposition references:
    #   Möttönen, Vartiainen, Bergholm & Salomaa, "Transformation of
    #   quantum states using uniformly controlled rotations", QIC 2005
    #   (Sec. III) — original Ry/Rz factorisation for complex amplitudes.
    #   Plesch & Brukner, Phys. Rev. A 83, 032302 (2011) — equivalent
    #   phase-strip + real-merge view for sparse state preparation.
    cx1 = state[x1]
    cx2 = state[x2]
    cx1_imag = abs(cx1.imag) if isinstance(cx1, complex) else 0.0
    cx2_imag = abs(cx2.imag) if isinstance(cx2, complex) else 0.0
    needs_phase_strip = cx1_imag > 1e-14 or cx2_imag > 1e-14

    if needs_phase_strip:
        delta = cmath.phase(complex(cx1)) - cmath.phase(complex(cx2))
        # Normalise δ to (-π, π]
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        if abs(delta) > 1e-14:
            if not dif_qubits:
                gates.append(('p', -delta, dif))
            elif len(dif_qubits) == 1:
                gates.append(('cp', -delta, dif_qubits[0], dif))
            else:
                gates.append(('mcp', -delta, list(dif_qubits),
                              list(dif_vals), dif))
        # After the strip, the magnitude merge is exactly the real case.
        abs_cx1 = abs(cx1)
        abs_cx2 = abs(cx2)
        theta = -2.0 * math.atan2(abs_cx1, abs_cx2)
    else:
        # Real-valued amplitudes (possibly stored as Python complex with
        # zero imaginary): use the original signed-atan2 path so gate
        # counts on existing real-only test inputs are unchanged.
        cx1r = float(cx1.real) if isinstance(cx1, complex) else float(cx1)
        cx2r = float(cx2.real) if isinstance(cx2, complex) else float(cx2)
        theta = -2.0 * math.atan2(cx1r, cx2r)

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

    elif gtype == 'p':
        # ('p', delta, target): uncontrolled phase shift on |1⟩ of target.
        delta, tgt = gate[1], gate[2]
        factor = cmath.exp(1j * delta)
        new_state = {}
        for bits, amp in state.items():
            if bits[tgt] == 1:
                new_state[bits] = amp * factor
            else:
                new_state[bits] = amp
        state.clear()
        state.update(new_state)

    elif gtype == 'cp':
        # ('cp', delta, ctrl, target): controlled phase shift.
        delta, ctrl, tgt = gate[1], gate[2], gate[3]
        factor = cmath.exp(1j * delta)
        new_state = {}
        for bits, amp in state.items():
            if bits[ctrl] == 1 and bits[tgt] == 1:
                new_state[bits] = amp * factor
            else:
                new_state[bits] = amp
        state.clear()
        state.update(new_state)

    elif gtype == 'mcp':
        # ('mcp', delta, ctrl_list, ctrl_vals, target): multi-controlled
        # phase shift.  X-flanking on active-low controls is emitted
        # separately as 'x' gates, so here we treat every control as
        # active-high (must be |1⟩) — same convention as 'mcry'.
        delta, ctrl_list, ctrl_vals, tgt = gate[1], gate[2], gate[3], gate[4]
        factor = cmath.exp(1j * delta)
        new_state = {}
        for bits, amp in state.items():
            if all(bits[q] == 1 for q in ctrl_list) and bits[tgt] == 1:
                new_state[bits] = amp * factor
            else:
                new_state[bits] = amp
        state.clear()
        state.update(new_state)


def _mcry(qc: QuantumCircuit, theta: float,
          ctrl_qubits: list, ctrl_vals: list, target: int) -> None:
    """Multi-controlled Ry — assumes control qubits are already in the
    all-1 state when the gate fires.

    The Gleinig-Hoefler path explicitly adds 'x' gates to the gate list
    to flank the MCRY for active-low controls (see _gleinig_reduce), and
    _apply_gate_to_state's 'mcry' branch correspondingly ignores
    ctrl_vals (assumes all controls must be 1).  Adding another layer of
    X flanking here would double-flip the low-value controls and cause
    the MCRY to fire on the wrong basis states.

    Parameter ctrl_vals is accepted for interface compatibility with the
    callers' gate-list encoding, but is intentionally not acted on.
    """
    if len(ctrl_qubits) == 1:
        qc.cry(theta, ctrl_qubits[0], target)
    else:
        qc.mcry(theta, ctrl_qubits, target)


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

    For target signal  f_k = sum_t  A_t  sin(2 pi n_t k / N + phi_t),
    the +2 pi convention DFT (the convention applied by Qiskit's QFTGate)
    has nonzero frequency-domain coefficients

        C_{n_t}     = (A_t / 2) * exp(  i * (phi_t - pi/2))
        C_{N - n_t} = (A_t / 2) * exp(  i * (pi/2 - phi_t))

    derived from the identity sin(x) = (1/(2i)) * (e^{ix} - e^{-ix}).
    Each mode therefore contributes 2 nonzero entries with mode-dependent
    phases; for the all-phi=0 case both phases collapse to {-pi/2, +pi/2}
    and the imbalance is captured by a single Z gate on the MSB.  For
    nonzero phi_t these two phases vary per mode, so the per-mode phase
    must be loaded into the frequency-domain register directly.

    Strategy:
      1. Build the complex frequency-domain anchor list with the per-mode
         phases above.
      2. Use the signed Gleinig-Hoefler point loader, which accepts
         complex amplitudes natively, to prepare the magnitude-and-phase
         distribution.
      3. Apply QFT to transform frequency -> spatial domain.

    Gate count: O(T*m) for the signed loader + O(m^2) for QFT = O(m^2).

    Requires all mode frequencies n_t < N/2 (standard assumption).

    Backward compatibility: when every phi_t = 0 the complex amplitudes
    above reduce to (-i*A_t/2) at p = n_t and (+i*A_t/2) at p = N - n_t
    -- the same magnitudes as the previous implementation but with a
    relative phase of pi between the two halves, exactly the Z-on-MSB
    pattern the old code emitted by hand.  The signed loader subsumes
    that special case, so the previous _synth_disjoint_point_load + Z
    pipeline is no longer needed.
    """
    modes = params["modes"]  # list of {"A": amplitude, "n": mode, "phi": phase}
    N     = 2 ** m

    qc = QuantumCircuit(m, name="multi_sin_load")

    # Step 1: build complex frequency-domain anchors with per-mode phases.
    freq_loads = []
    for mode in modes:
        n   = mode["n"]
        A   = abs(mode["A"])
        phi = mode.get("phi", 0.0)
        if not (0 < n < N):
            continue
        # See module docstring above for the derivation.
        c_pos = (A / 2.0) * cmath.exp(1j * (phi - math.pi / 2.0))
        c_neg = (A / 2.0) * cmath.exp(1j * (math.pi / 2.0 - phi))
        freq_loads.append({"k": n,     "P": c_pos})
        freq_loads.append({"k": N - n, "P": c_neg})

    if not freq_loads:
        raise ValueError("All mode amplitudes are zero.")

    # Step 2: complex-aware Gleinig-Hoefler load.  The signed loader
    # handles purely-real (sign-only) and fully-complex amplitudes
    # uniformly; for the all-phi=0 case this reproduces the same
    # frequency-domain state as the previous Ry-tree + Z construction.
    freq_circuit = _synth_disjoint_point_load_signed(m, {"loads": freq_loads})
    qc.compose(freq_circuit, inplace=True)

    # Step 3: QFT to transform frequency -> spatial domain.
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

    For s=1 this reduces to _synth_point_load (X gates only); a non-real
    amplitude is encoded as a global phase ``arg(P)`` (which subsumes the
    π for the negative-real case) so that the sign/phase is preserved
    when the circuit is used as a controlled sub-block (e.g. inside SUM,
    where a global phase becomes a relative phase).

    For s>1 the full Gleinig-Hoefler O(s*m) construction is used.  When
    any amplitude is negative or complex we route through the signed
    loader, whose pairwise merge handles arbitrary real amplitudes via
    θ = -2·atan2(c_x1, c_x2) and arbitrary complex amplitudes via the
    additional phase-strip CP/MCP gate emitted by ``_gleinig_reduce``.
    """
    loads = params["loads"]
    # Real-but-signed (any P with negative real part) and any complex
    # (non-zero imaginary) both need the signed/phase-aware path.
    has_negative = any(complex(ld["P"]).real < 0.0
                       and abs(complex(ld["P"]).imag) < 1e-14
                       for ld in loads)
    has_complex  = any(abs(complex(ld["P"]).imag) > 1e-14 for ld in loads)
    needs_signed = has_negative or has_complex

    if len(loads) == 1:
        qc = _synth_point_load(m, loads[0])
        # Encode the amplitude's phase as a global phase on the
        # single-basis-state circuit.  arg(P) is 0 for real positive,
        # π for real negative, arbitrary for complex.
        P = complex(loads[0]["P"])
        phi = cmath.phase(P)
        if abs(phi) > 1e-14:
            qc.global_phase += phi
        return qc

    if needs_signed:
        return _synth_disjoint_point_load_signed(m, params)
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


def _synth_walsh(m: int, params: dict) -> QuantumCircuit:
    """
    Generalized WALSH: U_walsh on qubit k + H^{otimes m}.

    Prepares a two-level piecewise-constant state with period P = 2^(k+1):
      amplitude proportional to c0 where bit k of i is 0
      amplitude proportional to c1 where bit k of i is 1

    Real (c0, c1) case
    ------------------
    The single-qubit preparation reduces to R_y(theta) with
        theta = 2 * atan2(c0 - c1, c0 + c1).
    When c1 = -c0: theta = pi and R_y(pi) = X (standard Walsh).
    Circuit: R_y(theta) on qubit k, H on all m qubits.
    Gate count: m + 1.

    Complex (c0, c1) case
    ---------------------
    Let N_c = sqrt(|c0|^2 + |c1|^2),
        alpha = (c0 + c1) / (sqrt(2) * N_c),
        beta  = (c0 - c1) / (sqrt(2) * N_c),
    so |alpha|^2 + |beta|^2 = 1 and after H^{otimes m} the
    bit-k = b component carries amplitude c_b / Z.  Decompose
    alpha = e^{i*phi_a} cos(theta/2),  beta = e^{i*phi_b} sin(theta/2):
      * R_y(theta) on qubit k  -- cos/sin amplitudes,
      * P(phi_b - phi_a) on qubit k  -- relative phase,
      * qc.global_phase += phi_a  -- absolute phase on |0>_k branch.
    Reduces to the real construction when c0, c1 are real.

    Complexity: O(m).
    """
    k  = params["k"]
    c0 = params.get("c0", 1.0)
    c1 = params.get("c1", -c0)

    if k < 0 or k >= m:
        raise ValueError(f"WALSH qubit index k={k} out of range [0, {m}).")

    is_complex = isinstance(c0, complex) or isinstance(c1, complex)
    qc = QuantumCircuit(m, name="walsh")

    if not is_complex:
        # ------- backward-compatible real path -------
        denom = c0 + c1
        if abs(denom) < 1e-14:
            theta = math.pi
        else:
            theta = 2 * math.atan2(c0 - c1, c0 + c1)
        if abs(theta - math.pi) < 1e-12:
            qc.x(k)
        else:
            qc.ry(theta, k)
    else:
        # ------- complex path -------
        c0c, c1c = complex(c0), complex(c1)
        N_c = math.sqrt(abs(c0c)**2 + abs(c1c)**2)
        if N_c < 1e-14:
            raise ValueError("WALSH amplitudes (c0, c1) cannot both be zero.")
        alpha = (c0c + c1c) / (math.sqrt(2) * N_c)
        beta  = (c0c - c1c) / (math.sqrt(2) * N_c)
        abs_a, abs_b = abs(alpha), abs(beta)
        theta = 2 * math.atan2(abs_b, abs_a)
        # Special cases: pure |0> or pure |1> -- avoid unnecessary R_y / P.
        if abs_b < 1e-14:
            phi_a = cmath.phase(alpha)
            qc.global_phase += phi_a
        elif abs_a < 1e-14:
            qc.x(k)
            phi_b = cmath.phase(beta)
            qc.global_phase += phi_b
        else:
            phi_a = cmath.phase(alpha)
            phi_b = cmath.phase(beta)
            qc.ry(theta, k)
            if abs(phi_b - phi_a) > 1e-14:
                qc.p(phi_b - phi_a, k)
            if abs(phi_a) > 1e-14:
                qc.global_phase += phi_a

    for q in range(m):
        qc.h(q)
    return qc


# ---------------------------------------------------------------------------
# GEOMETRIC  f_i = c * r^i  (product state, m independent Ry)
# ---------------------------------------------------------------------------

def _synth_geometric(m: int, params: dict) -> QuantumCircuit:
    """
    GEOMETRIC: prepare |psi> proportional to
        sum_{i=k_s}^{k_e-1}  r^(i - k_s) |i>            on m qubits,
    zero outside [k_s, k_e).  Three internal regimes are selected
    automatically (let w = k_e - k_s, N = 2^m):

      (a) k_s == 0 and k_e == N  (full register)
          Plain product state.  m R_y gates, 0 CX, depth 1.
          Theta_j = 2*arctan(|r|^(2^j)) on qubit j; for complex r,
          a P(arg(r)*2^j) phase gate is appended on qubit j.

      (b) [k_s, k_e) is a single dyadic block
          i.e. w is a power of two AND k_s % w == 0.
          log2(w) R_y/P pairs on the low qubits plus X gates on the
          upper qubits that encode k_s // w.  Still depth 1, 0 CX.

      (c) Otherwise  (general window)
          Dyadic decomposition of [k_s, k_e) into L <= m aligned blocks
          {(a_k, j_k)} with block k covering [a_k, a_k + 2^j_k).  Each
          block already matches regime (b), so the full state is the
          weighted superposition (with disjoint supports)

              |psi> = sum_k (w_k / Z) |block_k>,

              w_k = r^(a_k - k_s) * sqrt( (|r|^(2*2^j_k) - 1)
                                                 / (|r|^2 - 1) ),

          Assembly: Gleinig-Hoefler on the L anchor points {|a_k>} to
          load the weights, followed by a multi-controlled R_y (and,
          for complex r, MCP) on each free bit of each block to spread
          the anchor amplitude across the block.  Total: O(L*m^2) basis
          gates, no ancilla, post-selection probability 1 (the blocks
          are disjoint).

    Complex / signed r
    ------------------
    For real positive r the original product-state circuit is recovered
    exactly (arg(r) = 0 suppresses every P / MCP gate).  For real
    negative r the only emitted phase is on qubit 0 (other 2^j*pi mod 2pi
    vanish).  For complex r every qubit j picks up a phase arg(r)*2^j.

    Parameters
    ----------
    m : int
        Number of qubits (N = 2^m).
    params : dict
        Required:  'r' (real or complex, |r| != 0, r != 1).
        Optional:  'k_s' (int, default 0, 0 <= k_s < N).
                   'k_e' (int, default N, k_s < k_e <= N).
                   'c'   (real or complex, default 1.0) — its magnitude
                         only affects normalisation; its phase is
                         recorded as qc.global_phase.

    References
    ----------
    Xie et al. 2025 (product-state form of r^i).
    Gleinig & Hoefler, DAC 2021 (sparse anchor-loading step).
    Bentley & Saxe, J. Algorithms 1(4), 1980 (dyadic interval decomposition).
    """
    r = params["r"]
    k_s = params.get("k_s", 0)
    c   = params.get("c", 1.0)
    N = 1 << m
    k_e = params.get("k_e", N)
    if k_e is None:
        k_e = N

    abs_r = abs(r)
    arg_r = cmath.phase(complex(r))
    arg_c = cmath.phase(complex(c))
    is_complex_r = isinstance(r, complex) and abs(r.imag) > 1e-14

    qc = QuantumCircuit(m, name="geometric")

    def _per_qubit(j_idx: int, q_idx: int) -> None:
        """Per-qubit (|0> + r^(2^j_idx) |1>) preparation on qubit q_idx."""
        if is_complex_r:
            qc.ry(2.0 * math.atan(abs_r ** (1 << j_idx)), q_idx)
            phase_j = (arg_r * (1 << j_idx)) % (2 * math.pi)
            if phase_j > math.pi:
                phase_j -= 2 * math.pi
            if abs(phase_j) > 1e-14:
                qc.p(phase_j, q_idx)
        else:
            # Real (signed) path.  For r < 0 this still works because
            # r^(2^j) is real; atan handles negatives correctly.
            qc.ry(2.0 * math.atan(r ** (1 << j_idx)), q_idx)

    # --- Regime (a): plain full-register product state -------------------
    if k_s == 0 and k_e == N:
        for j in range(m):
            _per_qubit(j, j)
        if abs(arg_c) > 1e-14:
            qc.global_phase += arg_c
        return qc

    # --- Regime (b): single dyadic block covering [k_s, k_e) -----------
    w = k_e - k_s
    if (w & (w - 1)) == 0 and k_s % w == 0:
        m_low = w.bit_length() - 1
        for j in range(m_low):
            _per_qubit(j, j)
        upper_val = k_s // w
        for j in range(m - m_low):
            if (upper_val >> j) & 1:
                qc.x(m_low + j)
        if abs(arg_c) > 1e-14:
            qc.global_phase += arg_c
        return qc

    # --- Regime (c): dyadic decomposition of [k_s, k_e) ----------------
    blocks = _dyadic_decomposition(k_s, k_e)
    _dyadic_geometric_assemble(qc, m, r, k_s, blocks)
    if abs(arg_c) > 1e-14:
        qc.global_phase += arg_c
    return qc


# ---------------------------------------------------------------------------
# Helpers for GEOMETRIC regime (c)
# ---------------------------------------------------------------------------

def _dyadic_decomposition(s: int, e: int) -> list:
    """
    Decompose the half-open interval [s, e) into maximal power-of-two-aligned
    dyadic blocks.

    Each returned block is a pair (a_k, j_k) representing the interval
    [a_k, a_k + 2^j_k),  with  a_k % 2^j_k == 0.  The blocks are disjoint,
    cover [s, e) exactly, and their count L satisfies L <= 2*ceil(log2(e)).

    Classical cost: O(L) integer ops.

    Algorithm: walk left-to-right, greedily picking the largest aligned
    block starting at the current position that fits inside [s, e).
    At position `cur`, the largest admissible j is
        j = min( trailing_zeros(cur),  floor(log2(e - cur)) )
    with the first term dropped when cur == 0.

    Reference: Bentley & Saxe, J. Algorithms 1(4):301-358, 1980
    (decomposable searching).  Standard primitive in segment trees /
    Fenwick trees (Fenwick, SPE 24(3), 1994).

    Parameters
    ----------
    s : int
        Inclusive left endpoint (0 <= s < e).
    e : int
        Exclusive right endpoint.  Any positive integer (need not be a
        power of 2; the algorithm handles arbitrary upper bounds).

    Returns
    -------
    list[tuple[int, int]]
        Sequence of (a_k, j_k) in strictly increasing a_k order.
    """
    blocks = []
    cur = s
    while cur < e:
        room = e - cur                       # largest j with 2^j <= room
        mx = room.bit_length() - 1
        if cur == 0:
            j = mx
        else:
            tz = (cur & -cur).bit_length() - 1   # trailing zeros of cur
            j = tz if tz < mx else mx
        blocks.append((cur, j))
        cur += 1 << j
    return blocks


def _dyadic_geometric_assemble(qc: QuantumCircuit,
                               m: int,
                               r,
                               k_s: int,
                               blocks: list) -> None:
    """
    Build the GEOMETRIC circuit for regime (c) into `qc` (acts on all m
    qubits).  `blocks` is the output of _dyadic_decomposition(k_s, N).

    Two-step assembly:

      Step 1.  Gleinig-Hoefler sparse-state preparation on the L anchor
               points {|a_k>}_{k=0..L-1} with (possibly complex) weights
               w_k/Z.  After this step, the register is in
                   |psi_anchor> = sum_k (w_k/Z) |a_k>.

      Step 2.  For each block with j_k >= 1 and each free bit j in
               {0, 1, ..., j_k - 1}, apply R_y(2*arctan(|r|^(2^j))) on
               qubit j controlled on the upper (m - j_k) qubits matching
               the bit pattern  a_k >> j_k.  When r is complex, append a
               P(arg(r)*2^j) on the same qubit/control pattern.  Because
               the lower j_k bits of a_k are zero (alignment), every
               qubit being rotated starts in |0> within the controlled
               subspace, so the standard product-state formula applies.

    The weights are
        w_k = r^(a_k - k_s) * sqrt( (|r|^(2*2^j_k) - 1) / (|r|^2 - 1) )
    so that |w_k|^2 = sum_{i in block_k} |r|^(2*(i - k_s)) and arg(w_k)
    equals (a_k - k_s)*arg(r); the magnitude part is the standard real
    geometric-sum formula evaluated at |r|, and the phase part is
    absorbed into the (signed/complex) Gleinig load.  When r is real
    positive both magnitudes and phases reduce to the original formula.

    For real positive r the arg(r) terms vanish, no P/MCP gates are
    emitted, and the original real circuit is recovered exactly.
    """
    abs_r = abs(r)
    arg_r = cmath.phase(complex(r))
    abs_r2 = abs_r * abs_r

    # Block weights.  Magnitude uses |r|; phase uses (a_k-k_s)*arg(r).
    weights = []
    for (a_k, j_k) in blocks:
        size = 1 << j_k
        if abs(abs_r2 - 1.0) < 1e-14:
            block_norm2 = float(size)
        else:
            block_norm2 = (abs_r2 ** size - 1.0) / (abs_r2 - 1.0)
        magn = (abs_r ** (a_k - k_s)) * math.sqrt(block_norm2)
        if abs(arg_r) > 1e-14:
            w_k = magn * cmath.exp(1j * arg_r * (a_k - k_s))
        else:
            # real-r branch -- preserve original sign behaviour
            w_k = (r ** (a_k - k_s)) * math.sqrt(block_norm2)
        weights.append(w_k)
    Z = math.sqrt(sum(abs(w) * abs(w) for w in weights))

    # Step 1: load anchor superposition via Gleinig-Hoefler on L points.
    anchor_state = {}
    for (a_k, _j_k), w_k in zip(blocks, weights):
        bits = tuple((a_k >> q) & 1 for q in range(m))     # LSB-first
        anchor_state[bits] = w_k / Z
    if len(anchor_state) == 1:
        # Degenerate (L=1) — single aligned block handled by regime (b),
        # but guard anyway: just X-load the bit pattern (and global-phase
        # the residual amplitude phase).
        ((only_bits, only_amp),) = anchor_state.items()
        for q, b in enumerate(only_bits):
            if b:
                qc.x(q)
        if abs(cmath.phase(complex(only_amp))) > 1e-14:
            qc.global_phase += cmath.phase(complex(only_amp))
    else:
        _gleinig_encode(qc, anchor_state, m)

    # Step 2: spread each anchor across its block via MC-R_y on free bits.
    # Blocks with j_k == 0 (singletons) contribute nothing here.
    for (a_k, j_k) in blocks:
        if j_k == 0:
            continue
        ctrl_qubits = list(range(j_k, m))          # upper qubits
        ctrl_pattern = [(a_k >> q) & 1 for q in ctrl_qubits]
        for j in range(j_k):                       # free (lower) bits
            if abs(arg_r) > 1e-14:
                # complex path
                theta_j = 2.0 * math.atan(abs_r ** (1 << j))
                _mcry_on_pattern(qc, theta_j, ctrl_qubits, ctrl_pattern, target=j)
                phase_j = (arg_r * (1 << j)) % (2 * math.pi)
                if phase_j > math.pi:
                    phase_j -= 2 * math.pi
                if abs(phase_j) > 1e-14:
                    _mcp_on_pattern(qc, phase_j, ctrl_qubits, ctrl_pattern, target=j)
            else:
                # real path (preserves original behaviour incl. r < 0)
                theta_j = 2.0 * math.atan(r ** (1 << j))
                _mcry_on_pattern(qc, theta_j, ctrl_qubits, ctrl_pattern, target=j)


def _mcry_on_pattern(qc: QuantumCircuit,
                     theta: float,
                     ctrl_qubits: list,
                     ctrl_pattern: list,
                     target: int) -> None:
    """
    Apply R_y(theta) on `target` controlled on `ctrl_qubits` being in the
    arbitrary-bit-pattern `ctrl_pattern` (same length as ctrl_qubits, each
    entry 0 or 1).  Standard X-flip sandwich around Qiskit's mcry.

    When ctrl_qubits is empty, reduces to an uncontrolled R_y.
    When len(ctrl_qubits) == 1, uses the native 1-qubit controlled R_y.
    """
    if not ctrl_qubits:
        qc.ry(theta, target)
        return
    # Flip controls that should be 0 so that the "all-ones" mcry fires on
    # the requested pattern.
    for q, bit in zip(ctrl_qubits, ctrl_pattern):
        if bit == 0:
            qc.x(q)
    if len(ctrl_qubits) == 1:
        qc.cry(theta, ctrl_qubits[0], target)
    else:
        qc.mcry(theta, ctrl_qubits, target, None, mode="noancilla")
    for q, bit in zip(ctrl_qubits, ctrl_pattern):
        if bit == 0:
            qc.x(q)


def _mcp_on_pattern(qc: QuantumCircuit,
                    phase: float,
                    ctrl_qubits: list,
                    ctrl_pattern: list,
                    target: int) -> None:
    """
    Apply a phase shift exp(i·phase) on the |1⟩ branch of `target`,
    controlled on `ctrl_qubits` being in the arbitrary bit-pattern
    `ctrl_pattern` (same length as ctrl_qubits, each entry 0 or 1).

    Sister of _mcry_on_pattern: same X-flip sandwich, but emits a
    (multi-)controlled-phase gate instead of a (multi-)controlled-Ry.
    Used by the GEOMETRIC / PARTITION block-spread when the geometric
    ratio r is non-real-positive (so the per-qubit ratio r^(2^j) needs
    a phase correction beyond the magnitude-only Ry).
    """
    if not ctrl_qubits:
        qc.p(phase, target)
        return
    for q, bit in zip(ctrl_qubits, ctrl_pattern):
        if bit == 0:
            qc.x(q)
    if len(ctrl_qubits) == 1:
        qc.cp(phase, ctrl_qubits[0], target)
    else:
        qc.mcp(phase, ctrl_qubits, target)
    for q, bit in zip(ctrl_qubits, ctrl_pattern):
        if bit == 0:
            qc.x(q)

# ---------------------------------------------------------------------------
# HAMMING  f_i ∝ r^{wt(i)}  (product state, m identical R_y gates)
# ---------------------------------------------------------------------------

def _synth_hamming(m: int, params: dict) -> QuantumCircuit:
    """
    HAMMING: amplitudes depend only on Hamming weight of the index.

    The target state is
        f_i = c * r^{wt(i)}  for i = 0, ..., N-1,
    where wt(i) is the Hamming weight (number of 1-bits, or `popcount`)
    of the binary representation of i.  This factorises over the bits
    of i as
        f_i = c * prod_j r^(b_j)  where  b_j = (i >> j) & 1.

    Hence the normalised state is a product state:
        |psi> = bigotimes_{j=0}^{m-1}  (|0> + r|1>) / sqrt(1 + |r|^2).

    For real positive r, each qubit is prepared by the same single-qubit
    rotation
        R_y(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    with theta = 2 * arctan(r).

    For complex r, each qubit is prepared by R_y(2*arctan(|r|)) followed
    by P(arg(r)).  When r is real positive, arg(r) = 0 and no phase
    gate is emitted, recovering the original circuit exactly.

    Gate count: m R_y gates plus (when r is non-real-positive) m P gates.
    Depth: 1 (all rotations / phases commute on disjoint qubits).
    Complexity: O(m).

    Parameters
    ----------
    m : int
        Number of qubits (N = 2^m).
    params : dict
        Must contain 'r' (real or complex, |r| != 0).  Optional 'c'
        (default 1.0); its magnitude only affects normalisation, its
        phase is recorded as qc.global_phase.
    """
    r = params["r"]
    c = params.get("c", 1.0)
    abs_r = abs(r)
    arg_r = cmath.phase(complex(r))
    arg_c = cmath.phase(complex(c))

    theta = 2.0 * math.atan(abs_r)

    qc = QuantumCircuit(m, name="hamming")
    for j in range(m):
        qc.ry(theta, j)
    if abs(arg_r) > 1e-14:
        for j in range(m):
            qc.p(arg_r, j)
    if abs(arg_c) > 1e-14:
        qc.global_phase += arg_c

    return qc


# ---------------------------------------------------------------------------
# STAIRCASE  f_{2^k-1} = r^k  (cascaded CR_y with per-step angles)
# ---------------------------------------------------------------------------

def _synth_staircase(m: int, params: dict) -> QuantumCircuit:
    """
    STAIRCASE: sparse geometric staircase on unary indices.

    Target amplitudes (unnormalised):
        f_{2^k - 1} = r^k   for k = 0, 1, ..., m
    with all other amplitudes zero.

    Construction
    ------------
    The state is built level-by-level in a cascade of m rotations:

        step 0:   R_y(theta_0)  on q0
        step k:   CR_y(theta_k) controlled by q_{k-1}, target q_k,
                                                          for k = 1..m-1

    At the k_s of step k the "active" branch carries the probability mass
    of all levels >= k.  The angle theta_k is chosen so that cos(theta_k/2)
    is f_k's share of that mass and sin(theta_k/2) is the remaining tail.

    Let  T_k = sqrt( sum_{j=k}^{m} r^{2j} )  be the tail norm from level k.
    Then  cos(theta_k/2) = r^k / T_k  and  sin(theta_k/2) = T_{k+1} / T_k,
    so  theta_k = 2 * atan2(T_{k+1}, r^k).  In particular
    theta_0 = 2 * atan2(T_1, 1).

    Gate count: m (1 R_y + (m-1) CR_y).  Complexity: O(m).

    Complex / signed amplitudes
    ---------------------------
    For arbitrary complex r = |r| * e^{i*arg(r)}, the magnitude cascade is
    built using |r| (so all atan2 arguments are real-positive and the
    rotation angles are well-defined), and a per-qubit phase gate
    P(arg(r)) is then applied on every qubit.  Because the support is
    on indices i_k = 2^k - 1 with Hamming weight wt(i_k) = k, the layer
    P(arg(r))^{otimes m} multiplies amplitude k by e^{i*k*arg(r)},
    converting |r|^k into r^k as required.  When r is real-positive,
    arg(r) = 0 and no extra gates are emitted (backward compatible).
    A complex prefactor c contributes its phase via qc.global_phase.

    Parameters
    ----------
    m : int
        Number of qubits (N = 2^m).
    params : dict
        Must contain 'r' (real or complex, |r| != 0).  Optional 'c' only
        affects normalisation; its phase is recorded as a global phase.
    """
    r = params["r"]
    c = params.get("c", 1.0)
    abs_r = abs(r)
    arg_r = cmath.phase(complex(r))
    arg_c = cmath.phase(complex(c))

    # Tail norms T_k = sqrt( sum_{j=k}^{m} |r|^{2j} ) for k = 0, ..., m+1
    # (T_{m+1} = 0).
    T = [0.0] * (m + 2)
    for k in range(m + 1):
        T[k] = math.sqrt(sum(abs_r ** (2 * j) for j in range(k, m + 1)))
    # T[m+1] is 0 -- theta_m would be 0 (no rotation), so we stop at k=m-1.

    qc = QuantumCircuit(m, name="staircase")

    # Step 0: R_y on qubit 0.  cos(theta_0/2) = 1/T_0, sin = T_1/T_0.
    theta_0 = 2.0 * math.atan2(T[1], 1.0)
    qc.ry(theta_0, 0)

    # Steps 1..m-1: CR_y from q_{k-1} to q_k.
    for k in range(1, m):
        # cos(theta_k/2) = |r|^k / T_k,  sin = T_{k+1}/T_k
        theta_k = 2.0 * math.atan2(T[k + 1], abs_r ** k)
        qc.cry(theta_k, k - 1, k)

    # Per-qubit phase layer P(arg(r)) on every qubit:
    # converts |r|^k into r^k on the support index 2^k - 1.
    # Skipped entirely when r is real-positive (arg(r) = 0).
    if abs(arg_r) > 1e-14:
        for j in range(m):
            qc.p(arg_r, j)

    # Global phase from complex prefactor c.
    if abs(arg_c) > 1e-14:
        qc.global_phase += arg_c

    return qc


# ---------------------------------------------------------------------------
# DICKE  |D^m_k> — uniform superposition over weight-k basis states
#                 (Bärtschi-Eidenbenz split-cyclic-shift construction)
# ---------------------------------------------------------------------------

def _dicke_scs_block(qc: QuantumCircuit, qubits, n: int, kk: int) -> None:
    """Apply the SCS_{n, kk} block on (kk + 1) qubits.

    ``qubits`` is a length-(kk+1) list of absolute qubit indices; local
    indices 0..kk map to qubits[0..kk].  The block is built from
      (i)      on local (kk-1, kk),  angle 2 arccos(sqrt(1/n))
      (ii)_l   on local (kk-l, kk-l+1, kk),  l = 2..kk,
               angle 2 arccos(sqrt(l/n))

    Each (i)-gate is CX-CRy-CX; each (ii)_l is CX-CCRy-CX.

    References
    ----------
    Bärtschi & Eidenbenz, FCT 2019, Definition 3 / Lemma 6.
    """
    from qiskit.circuit.library import RYGate

    assert len(qubits) == kk + 1

    # Gate (i) on (kk-1, kk)
    a = qubits[kk - 1]
    b = qubits[kk]
    theta = 2.0 * math.acos(math.sqrt(1.0 / n))
    qc.cx(a, b)
    qc.cry(theta, b, a)
    qc.cx(a, b)

    # Gates (ii)_l for l = 2..kk on (kk-l, kk-l+1, kk)
    for l in range(2, kk + 1):
        a = qubits[kk - l]
        b = qubits[kk - l + 1]
        c = qubits[kk]
        theta = 2.0 * math.acos(math.sqrt(l / n))
        qc.cx(a, c)
        # annotated=False matches the previous (non-annotated) default and
        # avoids Qiskit 2.3's DeprecationWarning about the change in default.
        ccry = RYGate(theta).control(num_ctrl_qubits=2, ctrl_state="11",
                                     annotated=False)
        qc.append(ccry, [c, b, a])
        qc.cx(a, c)


def _synth_dicke(m: int, params: dict) -> QuantumCircuit:
    """
    DICKE: uniform superposition over all m-qubit basis states with Hamming
    weight exactly k.

    Target state
    ------------
        |D^m_k> = C(m, k)^(-1/2) * sum_{|S|=k} |e_S>,
    i.e. f_i = c * 1[wt(i) == k], zero elsewhere.

    Construction (Bärtschi-Eidenbenz, FCT 2019)
    -------------------------------------------
    1. Prepare |0^(m-k') 1^k'> with k' X gates on the top k' qubits,
       where k' = min(k, m-k).
    2. Apply U_{m,k'}, the product of SCS^l blocks for l = m down to 2:
         - for l > k':  SCS^l_{k'}       (first block)
         - for l <= k': SCS^l_{l-1}      (second block)
       Each SCS block decomposes into 1 two-qubit (i)-gate plus (k'-1)
       three-qubit (ii)-gates with analytic angles 2 arccos(sqrt(l/n)).
    3. If k > m/2, apply X^{otimes m} to exploit the Dicke symmetry
         |D^m_k> = X^{otimes m} |D^m_{m-k}>.
       This halves the cascade cost whenever k > m/2 while remaining
       exact.

    Properties
    ----------
      Gate count     : O(k' * (m - k'))  CX, k' = min(k, m-k)
      Depth          : O(m)
      Ancilla        : none
      Success prob.  : 1 (deterministic)

    Special cases
    -------------
      k = 0  ->  |0...0>    (identity circuit)
      k = m  ->  |1...1>    (m X gates)

    Parameters
    ----------
    m : int
        Number of qubits (N = 2^m).
    params : dict
        Must contain 'k' (int, 0 <= k <= m).  Optional 'c' (default 1.0)
        only affects normalisation.

    References
    ----------
    A. Bärtschi & S. Eidenbenz, "Deterministic Preparation of Dicke
    States", FCT 2019, LNCS 11651, pp. 126-139.
    """
    k = params["k"]

    qc = QuantumCircuit(m, name="dicke")
    c = params.get("c", 1.0)
    arg_c = cmath.phase(complex(c))

    if k == 0:
        if abs(arg_c) > 1e-14:
            qc.global_phase += arg_c
        return qc
    if k == m:
        for i in range(m):
            qc.x(i)
        if abs(arg_c) > 1e-14:
            qc.global_phase += arg_c
        return qc

    # Symmetry: |D^m_k> = X^{otimes m} |D^m_{m-k}>. Prepare the lighter
    # state first when k > m/2 to halve the cascade cost.
    use_complement = (k > m - k)
    k_eff = m - k if use_complement else k

    # Initial: X on the top k_eff qubits -> |0^(m-k_eff) 1^k_eff>
    for i in range(m - k_eff, m):
        qc.x(i)

    # Apply U_{m, k_eff} as a single combined loop:
    # for l = m, m-1, ..., 2 apply SCS^l_{min(k_eff, l-1)} on qubits
    # [l - min(k_eff, l-1) - 1, l).
    for l in range(m, 1, -1):
        kk = min(k_eff, l - 1)
        qubits = list(range(l - kk - 1, l))
        _dicke_scs_block(qc, qubits, n=l, kk=kk)

    # Bit-flip every qubit to map |D^m_{m-k}> -> |D^m_k>.
    if use_complement:
        for i in range(m):
            qc.x(i)

    if abs(arg_c) > 1e-14:
        qc.global_phase += arg_c

    return qc


# ---------------------------------------------------------------------------
# POLYNOMIAL  f(i) = sum_{j=0}^{d} c_j * x(i)^j  (Walsh-sparse loading)
# ---------------------------------------------------------------------------

def _fwht_inplace(a: np.ndarray) -> None:
    """Iterative unnormalised Walsh-Hadamard transform, in place.

    After this call, a[k] = sum_i (-1)^{popcount(k AND i)} * a_input[i].
    """
    h = 1
    N = len(a)
    while h < N:
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                x, y = a[j], a[j + h]
                a[j]     = x + y
                a[j + h] = x - y
        h *= 2


def _synth_polynomial(m: int, params: dict) -> QuantumCircuit:
    """
    POLYNOMIAL: Walsh-sparse loading plus a Hadamard layer.

    Construction
    ------------
    Let  f(i) = sum_{j=0}^{d} coeffs[j] * x(i)^j  be the target polynomial,
    where x(i) = i / (N - 1) if normalize_domain=True, else x(i) = i.

    Classical preprocessing:
      1. Evaluate f on the grid (O(N), numerical).
      2. Compute the Walsh-Hadamard transform, x = W f / sqrt(N).  Because
         f has polynomial degree d, only Walsh coefficients x_k at indices
         with Hamming weight |k|_b <= d are nonzero (Welch 2014; Gonzalez-
         Conde 2024).  Total nonzero: s = sum_{k=0}^{d} C(m, k).

    Quantum circuit:
      3. Prepare the sparse state |psi_x> = sum_k x_k |k>  using the
         Gleinig-Hoefler disjoint-point-load synthesiser (O(s * m) gates).
         The loader handles real signed and complex Walsh coefficients
         natively via complex atan2 / phase-strip rotations.
      4. Apply a single H layer to all m qubits.  Since H^{otimes m}
         applied to |k> gives (1/sqrt(N)) sum_i (-1)^{popcount(k AND i)} |i>,
         this precisely inverts the Walsh transform and yields
         sum_i f_i |i>  (normalised).

    Gate count: O(s * m) = O(m^{d+1}).  Exact (no approximation).
    Complex-coefficient backward compatibility: when every coefficient is
    real (imag == 0), the Walsh vector is real and the Gleinig-Hoefler
    loader takes its original real-amplitude path; no extra phase-strip
    gates are emitted.

    Parameters
    ----------
    m : int
        Number of qubits (N = 2^m).
    params : dict
        Must contain 'coeffs' (list of real or complex numbers).
        Optional 'normalize_domain' (bool, default True).
    """
    coeffs           = params["coeffs"]
    normalize_domain = params.get("normalize_domain", True)
    d = len(coeffs) - 1
    N = 2 ** m

    # Detect whether any coefficient is genuinely complex (imag != 0).
    is_complex = any(
        isinstance(c, complex) and abs(c.imag) > 1e-14 for c in coeffs
    )

    # 1. Evaluate the polynomial on the grid (use complex dtype only when
    #    necessary, so the real path is bit-for-bit unchanged).
    if normalize_domain and N > 1:
        x = np.arange(N, dtype=float) / (N - 1)
    else:
        x = np.arange(N, dtype=float)
    if is_complex:
        f = np.polyval(list(reversed(coeffs)), x).astype(complex)
    else:
        # Cast to float in case the user passed Python complex(z) with imag=0.
        real_coeffs = [float(c.real if isinstance(c, complex) else c)
                       for c in coeffs]
        f = np.polyval(list(reversed(real_coeffs)), x)

    # 2. Walsh-Hadamard transform (in-place for memory efficiency).
    walsh = f.copy()
    _fwht_inplace(walsh)
    walsh /= np.sqrt(N)

    # 3. Extract nonzero Walsh coefficients at Hamming weight <= d.
    tol = 1e-12 * max(1.0, float(np.linalg.norm(walsh)))
    sparse_loads = []
    for k in range(N):
        if bin(k).count("1") > d:
            continue
        if abs(walsh[k]) > tol:
            if is_complex:
                # Preserve full complex amplitude; the Gleinig-Hoefler
                # loader emits a phase-strip CP/MCP gate per merge.
                sparse_loads.append({"k": k, "P": complex(walsh[k])})
            else:
                # Preserve original SIGNED real path -- gate count and
                # transpiled depth are unchanged when all coeffs are real.
                sparse_loads.append({"k": k, "P": float(walsh[k].real)})

    if not sparse_loads:
        raise ValueError(
            "POLYNOMIAL: all Walsh coefficients are zero within tolerance."
        )

    # 4. Synthesise the (possibly complex) sparse Walsh state via
    #    Gleinig-Hoefler.  The loader normalises |P_i| internally and
    #    absorbs phases into the rotation / phase-strip angles.
    if len(sparse_loads) == 1:
        # Single-basis-state load: X gates plus (for complex amplitude)
        # a global phase carrying the argument of the lone coefficient.
        load = sparse_loads[0]
        qc = QuantumCircuit(m, name="polynomial_load")
        k = int(load["k"])
        for bit in range(m):
            if (k >> bit) & 1:
                qc.x(bit)
        amp = load["P"]
        if isinstance(amp, complex) and abs(cmath.phase(amp)) > 1e-14:
            qc.global_phase += cmath.phase(amp)
    else:
        qc = _synth_disjoint_point_load_signed(m, {"loads": sparse_loads})

    poly_qc = QuantumCircuit(m, name="polynomial")
    poly_qc.compose(qc, inplace=True)

    # 5. Hadamard layer transforms the Walsh register into the target.
    for q in range(m):
        poly_qc.h(q)
    return poly_qc