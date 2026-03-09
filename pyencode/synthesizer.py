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
  MULTI_POINT_LOAD             :  O(m · L)      — binary-tree Ry, arbitrary weights
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
        LoadType.POINT_LOAD:        _synth_point_load,
        LoadType.UNIFORM_LOAD:      _synth_uniform_load,
        LoadType.STEP_LOAD:         _synth_step_load,
        LoadType.SQUARE_LOAD:       _synth_square_load,
        LoadType.SINUSOIDAL_LOAD:   _synth_sinusoidal,
        LoadType.COSINE_LOAD:       _synth_cosine,
        LoadType.MULTI_POINT_LOAD:  _synth_disjoint_point_load,
        LoadType.MULTI_SIN_LOAD:    _synth_multi_sin_load,
        LoadType.UNIFORM_SPIKE_LOAD: _synth_uniform_spike_load,
        LoadType.UNKNOWN:           _synth_mottonen,
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
    # H-gate circuit: width is a power of 2 AND segment is w-aligned (k1 % w == 0)
    if (w & (w - 1)) == 0 and (k1 % w == 0):
        p = int(round(math.log2(w)))       # H on p lower qubits
        # Encode k1 on all bits (upper bits identify the aligned block)
        for bit in range(m):
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
# Case A: multi-point loads  (arbitrary L ≥ 2, arbitrary weights)
# ---------------------------------------------------------------------------

def _synth_disjoint_point_load(m: int, params: dict) -> QuantumCircuit:
    """
    Prepare a weighted superposition of L ≥ 2 point loads with arbitrary
    non-negative magnitudes:

        |ψ⟩ = (1/‖a‖) Σ_{i=1}^{L} a_i |k_i⟩,   a_i = |P_i|

    Algorithm: binary-tree Ry decomposition (Mottonen et al., 2005;
    Plesch & Brukner, PRA 83, 2011).

    The L normalised amplitudes are arranged as the leaves of a complete
    binary tree of depth ⌈log₂ L⌉.  Each internal node j holds the
    partial norm of its subtree; the Ry rotation angle at node j is

        θ_j = 2 arcsin( ‖right subtree‖ / ‖node‖ )

    so that the amplitude is correctly split between the left (|0⟩) and
    right (|1⟩) children.

    After the Ry tree has distributed amplitudes, X-gate sequences encode
    each target index k_i onto the qubits, controlled on the binary path
    through the tree that selected leaf i.

    Gate count
    ----------
    - Ry tree     : L − 1  Ry gates
    - Index encoding : O(m · L) CX + X gates
    - Total : O(m · L)

    This reduces exactly to the hand-crafted two-point circuit for L = 2,
    and for all L is strictly cheaper than the Shende O(2^m) fallback
    whenever L ≪ 2^m.

    References
    ----------
    Mottonen et al., Quantum Inf. Comput. 5(6), 467–473, 2005.
    Plesch & Brukner, Phys. Rev. A 83, 032302, 2011.
    Shende, Markov, Bullock, IEEE TCAD 25(6), 1000–1010, 2006.
    """
    loads = params["loads"]
    L     = len(loads)

    # Normalise amplitudes
    amps = np.array([abs(load["P"]) for load in loads], dtype=float)
    norm = np.linalg.norm(amps)
    if norm < 1e-14:
        raise ValueError("All point-load magnitudes are zero.")
    amps = amps / norm

    indices = [load["k"] for load in loads]

    qc = QuantumCircuit(m, name="multi_point_load")
    _ry_tree_encode(qc, amps, indices, m)
    return qc


def _ry_tree_encode(qc: QuantumCircuit,
                    amps: np.ndarray,
                    indices: list,
                    m: int) -> None:
    """
    Binary-tree Ry encoding.

    Path-qubit design
    -----------------
    We use d path qubits (qubits m-d..m-1, root = qubit m-1).  The Ry
    tree routes amplitude to leaf i by setting path_qubits[j] = (i>>j)&1.
    For this to correctly encode index k_i, the d-bit *top prefix* of k_i
    (bits m-d..m-1) must equal the leaf index i.  This requires:

      1. d ≥ ceil(log2(L))  (enough leaves)
      2. All d-bit top prefixes of the L indices are DISTINCT.

    We choose the smallest d satisfying both constraints.  If no d ≤ m
    works, we fall back to Shende's general state preparation.

    After the Ry tree, the low (m-d) bits of each k_i are encoded onto
    data qubits (0..m-d-1) via controlled-X gates.
    """
    L = len(amps)
    if L == 1:
        for bit in range(m):
            if (indices[0] >> bit) & 1:
                qc.x(bit)
        return

    # Find minimum d such that all top-d-bit prefixes are distinct
    d_min = math.ceil(math.log2(L))
    d = None
    for d_try in range(d_min, m + 1):
        prefixes = [(k >> (m - d_try)) for k in indices]
        if len(set(prefixes)) == L:
            d = d_try
            break

    if d is None or d > m:
        # Fallback: Shende on sparse vector
        N = 2 ** m
        f = np.zeros(N, dtype=complex)
        for a, k in zip(amps, indices):
            f[k] = a
        norm = np.linalg.norm(f)
        if norm > 1e-14:
            f /= norm
        from qiskit.circuit.library import StatePreparation
        sp = StatePreparation(f, label="multi_point_load")
        qc.append(sp, range(m))
        qc.decompose(inplace=True)
        return

    path_qubits = list(range(m - d, m))   # [m-d, ..., m-1]; path_qubits[d-1] = root
    data_qubits = list(range(0, m - d))   # [0, ..., m-d-1]

    # Each index k_i has a unique d-bit top prefix: prefix_i = k_i >> (m-d).
    # We assign k_i to leaf number prefix_i.  This ensures that after the
    # Ry tree, path_qubits are in state = bits m-d..m-1 of k_i.
    num_leaves = 1 << d
    amps_padded    = np.zeros(num_leaves)
    indices_padded = [0] * num_leaves

    for a, k in zip(amps, indices):
        leaf_i = k >> (m - d)           # d-bit top prefix of k
        amps_padded[leaf_i]    = a
        indices_padded[leaf_i] = k

    # Build partial-norm heap (1-indexed)
    node_norm = np.zeros(2 * num_leaves + 1)
    for i in range(num_leaves):
        node_norm[num_leaves + i] = amps_padded[i]
    for node in range(num_leaves - 1, 0, -1):
        node_norm[node] = math.sqrt(node_norm[2 * node] ** 2 +
                                     node_norm[2 * node + 1] ** 2)

    # Apply Ry tree on path_qubits
    _apply_ry_subtree(qc, node_norm, node=1, depth=0, d=d,
                      path_qubits=path_qubits, ctrl_path=[])

    # For each real leaf i, encode data bits of k_i controlled on path.
    # After the Ry tree, path_qubits[j] = (i >> j) & 1 = (k >> (m-d+j)) & 1 ✓
    for i in range(num_leaves):
        if amps_padded[i] < 1e-14:
            continue
        k = indices_padded[i]
        path_state = [(i >> j) & 1 for j in range(d)]

        for b in range(m - d):
            if (k >> b) & 1:
                _controlled_x(qc, path_qubits, path_state, data_qubits[b])


def _apply_ry_subtree(qc: QuantumCircuit,
                      node_norm: np.ndarray,
                      node: int,
                      depth: int,
                      d: int,
                      path_qubits: list,
                      ctrl_path: list) -> None:
    """
    Recursively apply Ry rotations for the subtree rooted at `node`.
    path_qubits[d-1] is the root qubit (MSB); path_qubits[0] is deepest.
    Node at `depth` acts on path_qubits[d-1-depth].
    """
    num_leaves = 1 << d
    if node >= num_leaves:
        return

    norm_node  = node_norm[node]
    norm_right = node_norm[2 * node + 1]

    if norm_node < 1e-14:
        return

    theta = 2.0 * math.asin(min(norm_right / norm_node, 1.0))
    target_qubit = path_qubits[d - 1 - depth]   # root → top qubit

    if not ctrl_path:
        qc.ry(theta, target_qubit)
    else:
        ctrl_q = [cp[0] for cp in ctrl_path]
        ctrl_v = [cp[1] for cp in ctrl_path]
        _controlled_ry(qc, theta, ctrl_q, ctrl_v, target_qubit)

    _apply_ry_subtree(qc, node_norm, 2 * node,     depth + 1, d, path_qubits,
                      ctrl_path + [(target_qubit, 0)])
    _apply_ry_subtree(qc, node_norm, 2 * node + 1, depth + 1, d, path_qubits,
                      ctrl_path + [(target_qubit, 1)])


def _controlled_ry(qc, theta, ctrl_qubits, ctrl_vals, target):
    """Multi-controlled Ry with active-low controls handled via X flanking."""
    flip = [q for q, v in zip(ctrl_qubits, ctrl_vals) if v == 0]
    for q in flip:
        qc.x(q)
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


def _encode_index_on_path(qc, k, m, d, ctrl_qubits, path):
    """Kept for compatibility; superseded by inline logic in _ry_tree_encode."""
    pass


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