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
    """
    vector_type: str
    N: int
    m: int
    gate_count: int
    complexity: str
    validated: bool
    params: dict
    circuit_code: str = ""

    def __str__(self) -> str:
        lines = [
            f"PyEncode  v{__version__}",
            f"  Vector type : {self.vector_type}",
            f"  N           : {self.N}  (m = {self.m} qubits)",
            f"  Gate count  : {self.gate_count}",
            f"  Complexity  : {self.complexity}",
            f"  Validated   : {'yes' if self.validated else 'no'}",
        ]
        if self.params:
            lines.append(f"  Parameters  : {self.params}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Complexity table
# ---------------------------------------------------------------------------

_COMPLEXITY = {
    VectorType.STEP:    "O(m)",
    VectorType.SQUARE:  "O(m)",
    VectorType.SPARSE:  "O(s\u00b7m)",
    VectorType.FOURIER: "O(m\u00b2)",
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
}
