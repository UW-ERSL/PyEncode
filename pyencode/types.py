"""
pyencode.types
==============
Typed constructor classes, EncodingInfo dataclass, and related constants.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .recognizer import VectorType, LoadPattern

# Backward compat alias
LoadType = VectorType

__version__ = "0.4.0"


# ---------------------------------------------------------------------------
# Typed constructor classes for encode_params
# ---------------------------------------------------------------------------

class _VectorObj:
    """Base class for typed vector constructors."""
    vector_type: VectorType
    params: dict

    def __repr__(self):
        p = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.vector_type.name}({p})"


class DISCRETE(_VectorObj):
    """DISCRETE(k, P) — single nonzero entry at index k."""
    def __init__(self, k, P=1.0):
        self.vector_type = VectorType.DISCRETE
        self.params = {"k": int(k), "P": float(P)}


class UNIFORM(_VectorObj):
    """UNIFORM(c) — constant vector."""
    def __init__(self, c=1.0):
        self.vector_type = VectorType.UNIFORM
        self.params = {"c": float(c)}


class STEP(_VectorObj):
    """STEP(k_s, c) — prefix f[:k_s] = c."""
    def __init__(self, k_s, c=1.0):
        self.vector_type = VectorType.STEP
        self.params = {"k_s": int(k_s), "c": float(c)}


class SQUARE(_VectorObj):
    """SQUARE(k1, k2, c) — segment f[k1:k2] = c."""
    def __init__(self, k1, k2, c=1.0):
        self.vector_type = VectorType.SQUARE
        self.params = {"k1": int(k1), "k2": int(k2), "c": float(c)}


class SINE(_VectorObj):
    """SINE(n, A, phi=0) — sinusoidal mode."""
    def __init__(self, n, A=1.0, phi=0):
        self.vector_type = VectorType.SINE
        self.params = {"n": int(n), "A": float(A), "phi": float(phi)}


class COSINE(_VectorObj):
    """COSINE(n, A, phi=0) — cosine mode."""
    def __init__(self, n, A=1.0, phi=0):
        self.vector_type = VectorType.COSINE
        self.params = {"n": int(n), "A": float(A), "phi": float(phi)}


class MULTI_DISCRETE(_VectorObj):
    """MULTI_DISCRETE(vectors=[DISCRETE(...), ...]) — multiple point loads."""
    def __init__(self, vectors):
        self.vector_type = VectorType.MULTI_DISCRETE
        loads = []
        for v in vectors:
            if not isinstance(v, DISCRETE):
                raise TypeError(
                    f"MULTI_DISCRETE expects a list of DISCRETE objects, "
                    f"got {type(v).__name__}."
                )
            loads.append({"k": v.params["k"], "P": v.params["P"]})
        self.params = {"loads": loads}


class MULTI_SINE(_VectorObj):
    """MULTI_SINE(modes=[SINE(...), ...]) — sum of sinusoidal modes."""
    def __init__(self, modes):
        self.vector_type = VectorType.MULTI_SINE
        mode_list = []
        for m in modes:
            if not isinstance(m, SINE):
                raise TypeError(
                    f"MULTI_SINE expects a list of SINE objects, "
                    f"got {type(m).__name__}."
                )
            mode_list.append({"n": m.params["n"], "A": m.params["A"]})
        self.params = {"modes": mode_list}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class EncodingInfo:
    """
    Metadata about the encoding produced by the ``encode_*`` functions.

    Attributes
    ----------
    vector_type : str
        Name of the recognised vector pattern.
    N : int
        Number of vector components (must be a power of 2).
    m : int
        Number of qubits  (m = log2(N)).
    gate_count : int
        Total number of gates in the returned circuit.
    complexity : str
        Asymptotic gate complexity class (e.g. "O(m)", "O(m^2)").
    validated : bool
        True if the circuit was validated (statevector simulation).
    params : dict
        Extracted vector parameters (amplitudes, mode numbers, etc.).
    circuit_code : str
        Standalone Python/Qiskit source that builds the same circuit.
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
    VectorType.DISCRETE:        "O(m)",
    VectorType.UNIFORM:         "O(m)",
    VectorType.STEP:            "O(m)",
    VectorType.SQUARE:          "O(m)",
    VectorType.SINE:            "O(m\u00b2)",
    VectorType.COSINE:          "O(m\u00b2)",
    VectorType.MULTI_DISCRETE:  "O(m \u00b7 L)",
    VectorType.MULTI_SINE:      "O(m\u00b2)",
    VectorType.UNKNOWN:         "O(2^m)",
}


# ---------------------------------------------------------------------------
# Parameter schemas (for encode_params validation)
# ---------------------------------------------------------------------------

_PARAM_SCHEMAS = {
    VectorType.DISCRETE: {
        "required": {"k"},
        "optional": {"P"},
        "description": "k=<index>",
    },
    VectorType.UNIFORM: {
        "required": set(),
        "optional": {"c"},
        "description": "(no parameters required beyond N)",
    },
    VectorType.STEP: {
        "required": {"k_s"},
        "optional": {"c"},
        "description": "k_s=<step end index>",
    },
    VectorType.SQUARE: {
        "required": {"k1", "k2"},
        "optional": {"c"},
        "description": "k1=<start>, k2=<end>",
    },
    VectorType.SINE: {
        "required": {"n"},
        "optional": {"A", "phi"},
        "description": "n=<mode> (phi=<phase>, default 0)",
    },
    VectorType.COSINE: {
        "required": {"n"},
        "optional": {"A", "phi"},
        "description": "n=<mode> (phi=<phase>, default 0)",
    },
    VectorType.MULTI_DISCRETE: {
        "required": {"loads"},
        "optional": set(),
        "description": 'loads=[{"k": idx, "P": amp}, ...]',
    },
    VectorType.MULTI_SINE: {
        "required": {"modes"},
        "optional": set(),
        "description": 'modes=[{"n": mode, "A": amp}, ...]',
    },
}
