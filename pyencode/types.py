"""
pyencode.types
==============
Typed constructor classes and EncodingInfo dataclass.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .recognizer import VectorType

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _VectorObj:
    """Base class for typed vector constructors."""
    vector_type: VectorType
    params: dict

    def __repr__(self):
        p = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.vector_type.name}({p})"


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

class STEP(_VectorObj):
    """STEP(k_s, c) — prefix uniform superposition [0, k_s). O(m) gates.

    Implements the Shukla-Vedula (2024) interval circuit.
    STEP(k_s=N) produces the full uniform superposition H^m|0>.
    """
    def __init__(self, k_s, c=1.0):
        self.vector_type = VectorType.STEP
        self.params = {"k_s": int(k_s), "c": float(c)}


class SQUARE(_VectorObj):
    """SQUARE(k1, k2, c) — interval uniform superposition [k1, k2). O(m) gates.

    Extends STEP to arbitrary intervals. Novel contribution of PyEncode.
    SQUARE(k1=0, k2=k_s) is identical to STEP(k_s).
    """
    def __init__(self, k1, k2, c=1.0):
        self.vector_type = VectorType.SQUARE
        self.params = {"k1": int(k1), "k2": int(k2), "c": float(c)}


class WALSH(_VectorObj):
    """WALSH(k, c_pos, c_neg) — generalized Walsh function. O(m) gates.

    Prepares a two-level piecewise-constant state with period P = 2^(k+1):
      amplitude proportional to c_pos where bit k of i is 0,
      amplitude proportional to c_neg where bit k of i is 1.

    Circuit: R_y(theta) on qubit k, then H on all m qubits. Total: m+1 gates.
    When c_neg = -c_pos (default), R_y(pi) = X and this reduces to the
    standard Walsh function (signed uniform superposition).

    Parameters
    ----------
    k : int
        Qubit index (0 = LSB). Period P = 2^(k+1). Must satisfy 0 <= k < m.
    c_pos : float, optional
        Amplitude on the b_k=0 half. Default 1.0.
    c_neg : float, optional
        Amplitude on the b_k=1 half. Default -c_pos (standard Walsh).

    Examples
    --------
    >>> circuit, info = encode(WALSH(k=1), N=8)             # standard: +1/-1
    >>> circuit, info = encode(WALSH(k=1, c_pos=2.0), N=8)  # standard: +2/-2
    >>> circuit, info = encode(WALSH(k=2, c_pos=1.0, c_neg=4.0), N=8)  # Ry variant
    """
    def __init__(self, k, c_pos=1.0, c_neg=None):
        self.vector_type = VectorType.WALSH
        c_pos = float(c_pos)
        c_neg = float(c_neg) if c_neg is not None else -c_pos
        self.params = {"k": int(k), "c_pos": c_pos, "c_neg": c_neg}

class SPARSE(_VectorObj):
    """SPARSE([(x1, a1), (x2, a2), ...]) — s point masses at arbitrary indices.

    Implements the Gleinig-Hoefler algorithm. Gate complexity O(s * m).
    The s=1 case reduces to a single computational-basis state (X gates only).

    Example
    -------
    >>> circuit, info = encode(SPARSE([(19, 1.0)]), N=64)
    >>> circuit, info = encode(SPARSE([(1, 3.0), (6, 4.0)]), N=8)
    """
    def __init__(self, entries):
        self.vector_type = VectorType.SPARSE
        loads = []
        for item in entries:
            try:
                x, a = item
            except (TypeError, ValueError):
                raise TypeError(
                    f"SPARSE expects a list of (index, amplitude) tuples, "
                    f"got {item!r}."
                )
            loads.append({"k": int(x), "P": float(a)})
        if len(loads) == 0:
            raise ValueError("SPARSE requires at least one entry.")
        self.params = {"loads": loads}


class FOURIER(_VectorObj):
    """FOURIER(modes=[(n, A, phi), ...]) — T sinusoidal modes via inverse QFT.

    Gate complexity O(m^2) for any T.
    Single-mode T=1 subsumes SINE (phi=0) and COSINE (phi=pi/2).

    Example
    -------
    >>> circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=16)
    >>> circuit, info = encode(FOURIER(modes=[(1, 1.0, 0), (3, 0.5, 0)]), N=16)
    """
    def __init__(self, modes):
        self.vector_type = VectorType.FOURIER
        mode_list = []
        for item in modes:
            try:
                if len(item) == 3:
                    n, A, phi = item
                elif len(item) == 2:
                    n, A = item
                    phi = 0.0
                else:
                    raise ValueError
            except (TypeError, ValueError):
                raise TypeError(
                    f"FOURIER expects (n, A) or (n, A, phi) tuples, "
                    f"got {item!r}."
                )
            mode_list.append({"n": int(n), "A": float(A), "phi": float(phi)})
        if len(mode_list) == 0:
            raise ValueError("FOURIER requires at least one mode.")
        self.params = {"modes": mode_list}


class GEOMETRIC(_VectorObj):
    """GEOMETRIC(ratio, c) — exponential decay / geometric sequence. O(m) gates.

    Prepares a state proportional to f_i = c * ratio^i for i = 0, ..., N-1.

    The vector is multiplicatively separable over the bits of i:
      f_i = c * ratio^(sum_j b_j * 2^j) = c * prod_j ratio^(b_j * 2^j)
    so the quantum state is a product state prepared by m independent
    R_y rotations — one per qubit, zero entangling gates.

    Parameters
    ----------
    ratio : float
        Base of the geometric sequence. Must satisfy 0 < ratio and ratio != 1.
        Typical values: 0 < ratio < 1 for decay, ratio > 1 for growth.
    c : float, optional
        Leading amplitude (default 1.0). Only affects normalization.

    Examples
    --------
    >>> circuit, info = encode(GEOMETRIC(ratio=0.95), N=64)
    >>> circuit, info = encode(GEOMETRIC(ratio=0.5), N=16)
    """
    def __init__(self, ratio, c=1.0):
        self.vector_type = VectorType.GEOMETRIC
        ratio = float(ratio)
        if ratio <= 0:
            raise ValueError(f"GEOMETRIC ratio must be positive, got {ratio}.")
        if abs(ratio - 1.0) < 1e-14:
            raise ValueError(
                "GEOMETRIC ratio=1.0 is a uniform vector; use STEP(k_s=N) instead."
            )
        self.params = {"ratio": ratio, "c": float(c)}


class LCU(_VectorObj):
    """LCU([(w1, VectorObj1), (w2, VectorObj2), ...]) — linear combination via ancilla.

    Prepares a weighted superposition of structured component states:
      |psi> ∝ sum_j w_j |f^(j)>

    using ceil(log2(r)) ancilla qubits and controlled component circuits.

    When all components have disjoint support (e.g. multiple SQUARE intervals),
    success probability is exactly 1.0 and the ancilla uncomputes cleanly.
    For overlapping components, success probability p < 1.0 and post-selection
    or amplitude amplification is required.

    Parameters
    ----------
    components : list of (weight, VectorObj) tuples
        Unnormalized weights and typed constructors.
        All weights must be positive.

    Examples
    --------
    >>> # Piecewise-constant: two disjoint intervals (p=1)
    >>> circuit, info = encode(
    ...     LCU([(1.0, SQUARE(k1=0,  k2=8,  c=1.0)),
    ...          (4.0, SQUARE(k1=8,  k2=16, c=1.0))]), N=16)
    >>> # info.success_probability -> 1.0

    >>> # Mixed patterns: overlapping (p<1, warning issued)
    >>> circuit, info = encode(
    ...     LCU([(1.0, STEP(k_s=8, c=1.0)),
    ...          (1.0, FOURIER(modes=[(1, 1.0, 0)]))]), N=16)
    >>> # UserWarning: overlapping support, p < 1.0
    """
    def __init__(self, components):
        self.vector_type = VectorType.LCU
        if not components:
            raise ValueError("LCU requires at least one component.")
        weights = []
        objs = []
        for item in components:
            try:
                w, obj = item
            except (TypeError, ValueError):
                raise TypeError(
                    f"LCU expects (weight, VectorObj) tuples, got {item!r}."
                )
            if not isinstance(obj, _VectorObj):
                raise TypeError(
                    f"LCU component must be a VectorObj, got {type(obj).__name__}."
                )
            w = float(w)
            if w <= 0:
                raise ValueError(
                    f"LCU weights must be positive, got {w}."
                )
            weights.append(w)
            objs.append(obj)
        self.params = {"weights": weights, "components": objs}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class EncodingInfo:
    """
    Metadata returned by encode().

    Attributes
    ----------
    vector_type : str
        Name of the recognized vector pattern.
    N : int
        Number of vector components (must be a power of 2).
    m : int
        Number of qubits (m = log2(N)).
    gate_count : int
        Total number of gates in the returned circuit.
    complexity : str
        Asymptotic gate complexity (e.g. "O(m)", "O(m^2)").
    validated : bool
        True if the circuit was validated via statevector simulation.
    params : dict
        Supplied vector parameters.
    circuit_code : str
        Standalone Qiskit source that builds the same circuit.
    success_probability : float
        Post-selection probability (1.0 for single-pattern constructors;
        p in (0,1] for LCU).
    vector : np.ndarray or None
        The classically constructed amplitude vector f, populated only
        when validate=True. Requires O(2^m) memory. None otherwise.
    gate_count_1q : int or None
        Number of single-qubit gates after transpilation to {cx, u}.
        None if transpilation was not performed.
    gate_count_2q : int or None
        Number of two-qubit (CX) gates after transpilation to {cx, u}.
        None if transpilation was not performed.
    circuit_depth : int or None
        Circuit depth after transpilation to {cx, u}.
        None if transpilation was not performed.
    """
    vector_type: str
    N: int
    m: int
    gate_count: int
    complexity: str
    validated: bool
    params: dict
    circuit_code: str = ""
    success_probability: float = 1.0
    vector: Optional[np.ndarray] = None
    gate_count_1q: Optional[int] = None
    gate_count_2q: Optional[int] = None
    circuit_depth: Optional[int] = None

    def __str__(self) -> str:
        lines = [
            f"PyEncode  v{__version__}",
            f"  Vector type : {self.vector_type}",
            f"  N           : {self.N}  (m = {self.m} qubits)",
            f"  Gate count  : {self.gate_count}",
            f"  Complexity  : {self.complexity}",
            f"  Validated   : {'yes' if self.validated else 'no'}",
        ]
        if self.gate_count_1q is not None:
            lines.append(f"  Gates 1q/2q : {self.gate_count_1q} / {self.gate_count_2q}")
        if self.circuit_depth is not None:
            lines.append(f"  Depth       : {self.circuit_depth}")
        if self.success_probability < 1.0:
            lines.append(f"  Success prob: {self.success_probability:.4f}  "
                         f"(post-selection required)")
        if self.vector is not None:
            lines.append(f"  Vector      : numpy array, shape ({self.N},)")
        if self.params:
            lines.append(f"  Parameters  : {self.params}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Complexity table
# ---------------------------------------------------------------------------

_COMPLEXITY = {
    VectorType.STEP:    "O(m)",
    VectorType.SQUARE:  "O(m²)",   # general; O(m) for k1=0 or aligned blocks
    VectorType.WALSH:   "O(m)",
    VectorType.SPARSE:  "O(s\u00b7m)",
    VectorType.FOURIER: "O(m\u00b2)",
    VectorType.GEOMETRIC: "O(m)",
    VectorType.UNKNOWN: "O(2^m)",
}


# ---------------------------------------------------------------------------
# Parameter schemas (for encode() validation)
# ---------------------------------------------------------------------------

_PARAM_SCHEMAS = {
    VectorType.STEP: {
        "required": {"k_s"},
        "optional": {"c"},
        "description": "k_s=<prefix end index>",
    },
    VectorType.SQUARE: {
        "required": {"k1", "k2"},
        "optional": {"c"},
        "description": "k1=<start>, k2=<end>",
    },
    VectorType.WALSH: {
        "required": {"k"},
        "optional": {"c_pos", "c_neg"},
        "description": "k=<qubit index>, c_pos=<pos amplitude>, c_neg=<neg amplitude>",
    },
    VectorType.SPARSE: {
        "required": {"loads"},
        "optional": set(),
        "description": 'loads=[{"k": idx, "P": amp}, ...]',
    },
    VectorType.FOURIER: {
        "required": {"modes"},
        "optional": set(),
        "description": 'modes=[{"n": freq, "A": amp, "phi": phase}, ...]',
    },
    VectorType.GEOMETRIC: {
        "required": {"ratio"},
        "optional": {"c"},
        "description": "ratio=<base of geometric sequence>",
    },
}