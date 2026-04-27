"""
pyencode.mps
============
Standalone matrix-product-state (MPS) preparation, complementary to the
analytical pattern dictionary in ``pyencode``.  Unlike the closed-form
patterns (SPARSE, GEOMETRIC, FOURIER, ...) which build circuits directly
from typed parameters, MPS takes a real- or complex-valued amplitude
vector and produces an exact circuit at any bond dimension chi.  It is the
escape hatch for amplitude vectors whose structure does not match a closed
form but is nonetheless low-entanglement.

The construction is the right-canonical Schon sequential cascade
(Schon et al., PRL 95, 110503, 2005; Ran, PRA 101, 032310, 2020):

   1. Right-to-left SVD sweep on v (reshaped to (2^m, 1)) yields right-canonical
      MPS tensors A^[j] with bonds truncated to chi.
   2. The leftmost tensor's renormalisation absorbs the truncation deficit
      so the resulting circuit has p_succ = 1 deterministically (the bond
      register ends in |0> by construction; no post-selection).
   3. Each MPS tensor is completed to a (2*chi)x(2*chi) unitary U^[j] via
      a deterministic SVD-null-space completion that preserves the data
      columns verbatim (no per-column sign correction needed).
   4. The cascade U^[0] U^[1] ... U^[m-1] is appended to a Qiskit circuit
      with n_bond = ceil(log2(chi)) bond ancillas + m physical qubits.

Exactly one public entry point::

    from pyencode.mps import encode_mps
    circuit, info = encode_mps(v, bond_dim=4)

Return value matches ``pyencode.encode``:  a Qiskit ``QuantumCircuit``
plus an ``EncodingInfo`` dataclass with raw + transpiled gate counts and
depth measured under the same shared-config transpilation settings used
by every other PyEncode pattern (basis = {cx, u}, optimization_level = 3).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from .types import EncodingInfo
from .config import BASIS_GATES, OPTIMIZATION_LEVEL


# =============================================================
# Right-canonical MPS via right-to-left SVD sweep
# =============================================================

def _vector_to_right_canonical_mps(v: np.ndarray, m: int, chi_max: int):
    """Right-canonical MPS tensors via the standard SVD sweep.

    Returns ``(tensors, trunc_err_sq)`` where ``tensors[j]`` has shape
    ``(chi_l, 2, chi_r)`` (left-bond, physical, right-bond) and
    ``trunc_err_sq`` is the cumulative sum of squared discarded singular
    values.  ``1 - F_circuit <= trunc_err_sq`` for unit-norm input.
    """
    psi = v.reshape(-1, 1)
    tensors = [None] * m
    chi_right = 1
    trunc_err = 0.0
    for j in range(m - 1, 0, -1):
        rows = psi.shape[0]
        psi = psi.reshape(rows // 2, 2, chi_right).reshape(rows // 2,
                                                           2 * chi_right)
        U, s, Vt = np.linalg.svd(psi, full_matrices=False)
        chi_new = min(chi_max, len(s))
        trunc_err += float(np.sum(s[chi_new:] ** 2))
        U, s, Vt = U[:, :chi_new], s[:chi_new], Vt[:chi_new, :]
        tensors[j] = Vt.reshape(chi_new, 2, chi_right)
        psi = U * s[None, :]
        chi_right = chi_new
    tensors[0] = psi.reshape(1, 2, chi_right)
    return tensors, trunc_err


def _pad_tensors(tensors, chi_max):
    out = []
    for A in tensors:
        chi_l, _, chi_r = A.shape
        Apad = np.zeros((chi_max, 2, chi_max), dtype=A.dtype)
        Apad[:chi_l, :, :chi_r] = A
        out.append(Apad)
    return out


def _tensor_to_unitary(A_padded: np.ndarray) -> np.ndarray:
    """Complete an MPS tensor to a (2*chi) x (2*chi) unitary.

    The chi data columns of the Schon block (i_in = 0, a_in = 0..chi-1)
    are placed verbatim into Q at their original positions.  The
    remaining (d - n_nonzero) columns are filled deterministically from
    a basis of range(V)^perp computed via ``np.linalg.svd(V, full_matrices=True)``.
    Preserving data columns verbatim avoids the per-column sign-flip that
    QR's positive-diagonal convention would otherwise introduce, which
    would corrupt the Schon cascade's superposition over a_in.
    """
    chi = A_padded.shape[0]
    d = 2 * chi
    V = np.empty((d, chi), dtype=A_padded.dtype)
    V[:chi, :] = A_padded[:, 0, :].T
    V[chi:, :] = A_padded[:, 1, :].T
    nonzero = np.linalg.norm(V, axis=0) > 1e-12
    n_nz = int(nonzero.sum())
    U_svd, _, _ = np.linalg.svd(V, full_matrices=True)
    perp = U_svd[:, n_nz:]
    Q = np.empty((d, d), dtype=V.dtype)
    p = 0
    for k in range(chi):
        if nonzero[k]:
            Q[:, k] = V[:, k]
        else:
            Q[:, k] = perp[:, p]
            p += 1
    Q[:, chi:] = perp[:, p:p + chi]
    return Q


def _build_cascade_unitaries(tensors_padded) -> list[np.ndarray]:
    """Build per-site unitaries.  The leftmost tensor is renormalised to
    absorb any truncation norm deficit, ensuring p_succ = 1 deterministically."""
    out = []
    for j, A in enumerate(tensors_padded):
        if j == 0:
            n = np.linalg.norm(A)
            if n > 1e-15:
                A = A / n
        U = _tensor_to_unitary(A)
        d = U.shape[0]
        err = np.linalg.norm(U.conj().T @ U - np.eye(d))
        if err > 1e-8:
            raise RuntimeError(
                f"MPS site {j}: completed matrix is not unitary "
                f"(||U^dag U - I||_F = {err:.3e}). "
                "This indicates a numerical issue with the input vector."
            )
        out.append(U)
    return out


# =============================================================
# Public entry point
# =============================================================

def encode_mps(
    v: np.ndarray,
    bond_dim: int,
    *,
    validate: bool = False,
    tol: float = 1e-6,
    transpile_for_counts: bool = True,
) -> Tuple[QuantumCircuit, EncodingInfo]:
    """
    Prepare an arbitrary amplitude vector via a bounded-bond MPS circuit.

    Builds the right-canonical MPS of ``v`` truncated to bond dimension
    ``bond_dim``, then assembles the Schon sequential cascade as a Qiskit
    circuit on ``n_bond + m`` qubits, where::

        n_bond = max(1, ceil(log2(bond_dim)))
        m      = ceil(log2(len(v)))

    Bond ancillas (qubits 0..n_bond-1) start and end in |0> deterministically;
    physical qubits (n_bond..n_bond+m-1) carry the prepared state.  No
    post-selection is required: ``p_succ = 1`` exactly.

    Parameters
    ----------
    v : array_like
        Amplitude vector to prepare.  Real or complex.  If ``len(v)`` is
        not a power of 2 it is zero-padded to ``2**ceil(log2(len(v)))``.
        The vector is renormalised to unit L2 norm internally.
    bond_dim : int
        Maximum MPS bond dimension chi >= 1.  At chi >= rank(v) the MPS
        is exact; at smaller chi the tail singular values are truncated
        and ``info.params['truncation_error_sq']`` reports the
        cumulative truncation error (an upper bound on 1 - F_circuit).
    validate : bool, optional
        If True, simulate the circuit and verify the prepared physical
        state matches v to ``tol``.  Requires O(2^m) memory.
    tol : float, optional
        Tolerance for validation.  Default 1e-6.
    transpile_for_counts : bool, optional
        If True (default), transpile to {cx, u} at optimization_level=3
        to populate ``info.gate_count_2q`` and ``info.circuit_depth``.

    Returns
    -------
    circuit : qiskit.QuantumCircuit
        Acts on ``n_bond + m`` qubits.  The leftmost ``n_bond`` qubits are
        bond ancillas (returned to |0>).  Qubits ``n_bond + i`` for
        ``i = 0..m-1`` carry the prepared state with little-endian
        convention: qubit ``n_bond`` is the LSB of the index into v.
    info : pyencode.EncodingInfo
        Same dataclass returned by ``pyencode.encode``.  Additional MPS
        diagnostics are stored in ``info.params``::

            bond_dim              : the supplied chi
            n_bond                : ceil(log2(chi)) bond ancillas
            truncation_error_sq   : cumulative discarded singular-value
                                    weight, an upper bound on 1 - F
            n_padded              : 2**m, the padded register length

    Examples
    --------
    >>> import numpy as np
    >>> from pyencode.mps import encode_mps
    >>> v = np.random.default_rng(0).standard_normal(64)
    >>> v /= np.linalg.norm(v)
    >>> qc, info = encode_mps(v, bond_dim=4, validate=True)
    >>> info.gate_count_2q is not None
    True
    """
    # ---------- input handling ----------
    v = np.asarray(v).ravel()
    if v.size < 2:
        raise ValueError(f"len(v)={v.size}; need at least 2 amplitudes.")
    if bond_dim < 1:
        raise ValueError(f"bond_dim must be >= 1, got {bond_dim}.")

    n_orig = v.size
    m = max(1, int(math.ceil(math.log2(n_orig))))
    N = 1 << m
    if n_orig < N:
        v_padded = np.zeros(N, dtype=v.dtype)
        v_padded[:n_orig] = v
        v = v_padded
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        raise ValueError("Input vector is the zero vector.")
    v = v / norm

    chi_max = int(bond_dim)
    n_bond = max(0, int(math.ceil(math.log2(chi_max)))) if chi_max > 1 else 0

    # ---------- MPS construction ----------
    tensors, trunc_err_sq = _vector_to_right_canonical_mps(v, m, chi_max)
    tensors_padded = _pad_tensors(tensors, chi_max)
    unitaries = _build_cascade_unitaries(tensors_padded)

    # ---------- circuit assembly ----------
    qc = QuantumCircuit(n_bond + m, name=f"MPS(chi={chi_max})")
    bond = list(range(n_bond))
    for j, U in enumerate(unitaries):
        qc.append(UnitaryGate(U, label=f"U_{j}"),
                  bond + [n_bond + (m - 1 - j)])

    # ---------- validation ----------
    validated = False
    f_vec = None
    if validate:
        sv = Statevector(qc).data.reshape(2 ** m, 2 ** n_bond)
        sv_phys = sv[:, 0]
        # bond should be in |0> deterministically
        p_succ = float(np.linalg.norm(sv_phys) ** 2)
        if abs(p_succ - 1.0) > tol:
            raise RuntimeError(
                f"MPS p_succ = {p_succ:.6e} (deviates from 1 by "
                f"{abs(p_succ - 1.0):.3e}); bond register did not end in |0>."
            )
        sv_phys = sv_phys / np.linalg.norm(sv_phys)
        # Resolve global phase before comparing to v
        phase = np.vdot(v, sv_phys)
        if abs(phase) > 0:
            sv_phys = sv_phys * (np.conj(phase) / abs(phase))
        err = np.linalg.norm(sv_phys - v)
        if err > tol:
            raise RuntimeError(
                f"MPS validation failed: ||prepared - v||_2 = {err:.3e} "
                f"exceeds tol = {tol:.3e}.  The truncation error squared "
                f"was {trunc_err_sq:.3e}; consider increasing bond_dim."
            )
        validated = True
        f_vec = v

    # ---------- gate-count metrics ----------
    total_gates = sum(qc.count_ops().values())
    gate_count_1q = gate_count_2q = circuit_depth = None
    if transpile_for_counts:
        try:
            print("Transpiling MPS circuit for gate counts...")
            tr = transpile(qc, basis_gates=BASIS_GATES,
                           optimization_level=OPTIMIZATION_LEVEL)
            ops = tr.count_ops()
            gate_count_1q = ops.get("u", 0)
            gate_count_2q = ops.get("cx", 0)
            circuit_depth = tr.depth()
            print(f"Transpiled gate counts: {gate_count_1q} 1q gates, {gate_count_2q} 2q gates; depth {circuit_depth}.")    
        except Exception:
            print(" Transpilation failed; gate counts and depth will be None.")
            pass

    # ---------- info packaging ----------
    info = EncodingInfo(
        pattern_name="MPS",
        N=N,
        m=m,
        gate_count=total_gates,
        complexity=f"O(m*chi^2) with chi={chi_max}",
        validated=validated,
        params={
            "bond_dim": chi_max,
            "n_bond": n_bond,
            "truncation_error_sq": trunc_err_sq,
            "n_padded": N,
        },
        circuit_code="",   # closed-form pattern emit_code does not apply
        success_probability=1.0,
        vector=f_vec,
        gate_count_1q=gate_count_1q,
        gate_count_2q=gate_count_2q,
        circuit_depth=circuit_depth,
    )
    return qc, info


# =============================================================
# Advanced entry: pre-built tensors
# =============================================================

def encode_mps_from_tensors(
    tensors: Sequence[np.ndarray],
    *,
    validate_unit_norm: bool = True,
    transpile_for_counts: bool = True,
) -> Tuple[QuantumCircuit, EncodingInfo]:
    """
    Build the MPS-PREP circuit from pre-computed right-canonical tensors.

    Use this when you already have site tensors from an external source
    (e.g. DMRG output, custom truncation strategy) and want to skip
    PyEncode's SVD sweep.  ``tensors[j]`` must have shape
    ``(chi_l_j, 2, chi_r_j)`` with ``chi_l_0 = 1``, ``chi_r_{m-1} = 1``,
    and adjacent bonds matching: ``chi_r_j == chi_l_{j+1}``.

    The tensors are expected to be right-canonical: for every ``j``,
    ``sum_{i,b} A^{[j]}_{a,i,b}^* A^{[j]}_{a',i,b} = delta_{a,a'}``.
    The function does not enforce this; pass ``validate_unit_norm=True``
    to add a runtime check that the leftmost tensor has unit Frobenius
    norm (otherwise p_succ != 1 and the prepared state is renormalised).
    """
    tensors = [np.asarray(A) for A in tensors]
    m = len(tensors)
    if m < 1:
        raise ValueError("Need at least one tensor.")
    if tensors[0].shape[0] != 1:
        raise ValueError(
            f"Leftmost tensor must have left-bond 1, got "
            f"{tensors[0].shape[0]}."
        )
    if tensors[-1].shape[2] != 1:
        raise ValueError(
            f"Rightmost tensor must have right-bond 1, got "
            f"{tensors[-1].shape[2]}."
        )
    chi_max = max(max(A.shape[0], A.shape[2]) for A in tensors)
    for j in range(m - 1):
        if tensors[j].shape[2] != tensors[j + 1].shape[0]:
            raise ValueError(
                f"Bond mismatch between site {j} and {j+1}: "
                f"{tensors[j].shape[2]} vs {tensors[j+1].shape[0]}."
            )
    if validate_unit_norm:
        nrm = np.linalg.norm(tensors[0])
        if abs(nrm - 1.0) > 1e-8:
            raise ValueError(
                f"Leftmost tensor not unit-normed (||A^[0]||_F = {nrm:.6e}); "
                "right-canonical MPS with rightmost bond = 1 requires this."
            )

    tensors_padded = _pad_tensors(tensors, chi_max)
    unitaries = _build_cascade_unitaries(tensors_padded)

    n_bond = max(0, int(math.ceil(math.log2(chi_max)))) if chi_max > 1 else 0
    qc = QuantumCircuit(n_bond + m, name=f"MPS(chi={chi_max})")
    bond = list(range(n_bond))
    for j, U in enumerate(unitaries):
        qc.append(UnitaryGate(U, label=f"U_{j}"),
                  bond + [n_bond + (m - 1 - j)])

    total_gates = sum(qc.count_ops().values())
    gate_count_1q = gate_count_2q = circuit_depth = None
    if transpile_for_counts:
        try:
            tr = transpile(qc, basis_gates=BASIS_GATES,
                           optimization_level=OPTIMIZATION_LEVEL)
            ops = tr.count_ops()
            gate_count_1q = ops.get("u", 0)
            gate_count_2q = ops.get("cx", 0)
            circuit_depth = tr.depth()
        except Exception:
            pass

    info = EncodingInfo(
        pattern_name="MPS",
        N=1 << m,
        m=m,
        gate_count=total_gates,
        complexity=f"O(m*chi^2) with chi={chi_max}",
        validated=False,
        params={
            "bond_dim": chi_max,
            "n_bond": n_bond,
            "truncation_error_sq": 0.0,
            "n_padded": 1 << m,
        },
        circuit_code="",
        success_probability=1.0,
        vector=None,
        gate_count_1q=gate_count_1q,
        gate_count_2q=gate_count_2q,
        circuit_depth=circuit_depth,
    )
    return qc, info


__all__ = ["encode_mps", "encode_mps_from_tensors", "mps_cascade_unitaries"]


def mps_cascade_unitaries(v: np.ndarray, bond_dim: int):
    """Return the per-site unitaries of the right-canonical Schon cascade.

    Advanced entry point for users who want to assemble the cascade by hand
    -- for example, to build a controlled-MPS cascade for an LCU dispatch,
    or to apply a custom gate decomposition to each site unitary.  For
    the standard, uncontrolled MPS-PREP circuit, prefer ``encode_mps``.

    Returns
    -------
    unitaries : list[np.ndarray]
        ``unitaries[j]`` is a ``(2*chi) x (2*chi)`` unitary acting on
        ``n_bond + 1`` qubits, with the convention that bond qubits are
        the leftmost ``n_bond`` arguments and the physical qubit is the
        rightmost one.  The leftmost site (j=0) is the renormalised
        leftmost MPS tensor; subsequent sites are right-canonical.
    info : dict
        ``{'n_bond', 'bond_dim', 'm', 'truncation_error_sq', 'n_padded'}``.
    """
    v = np.asarray(v).ravel()
    if v.size < 2:
        raise ValueError(f"len(v)={v.size}; need at least 2 amplitudes.")
    if bond_dim < 1:
        raise ValueError(f"bond_dim must be >= 1, got {bond_dim}.")

    n_orig = v.size
    m = max(1, int(math.ceil(math.log2(n_orig))))
    N = 1 << m
    if n_orig < N:
        v_padded = np.zeros(N, dtype=v.dtype)
        v_padded[:n_orig] = v
        v = v_padded
    nrm = np.linalg.norm(v)
    if nrm < 1e-15:
        raise ValueError("Input vector is the zero vector.")
    v = v / nrm

    chi_max = int(bond_dim)
    n_bond = max(0, int(math.ceil(math.log2(chi_max)))) if chi_max > 1 else 0
    tensors, trunc_err_sq = _vector_to_right_canonical_mps(v, m, chi_max)
    tensors_padded = _pad_tensors(tensors, chi_max)
    unitaries = _build_cascade_unitaries(tensors_padded)
    return unitaries, {
        "n_bond": n_bond,
        "bond_dim": chi_max,
        "m": m,
        "truncation_error_sq": trunc_err_sq,
        "n_padded": N,
    }
