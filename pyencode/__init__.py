"""
pyencode
==========
A Python compiler for quantum amplitude encoding of structured load vectors.

Analogous to PyQUBO for QUBO formulation, PyEncode translates Python
code that constructs a structured state vector directly into an
efficient Qiskit quantum circuit — without ever materialising the
exponentially large amplitude vector.

Quick start
-----------
>>> from pyencode import encode
>>>
>>> circuit, info = encode('''
... import numpy as np
... N = 8
... f = np.zeros(N)
... f[3] = 5.0
... ''')
>>> print(info)
>>> circuit.draw('text')

Supported load patterns
-----------------------
  Point load      :  f[k] = P
  Uniform load    :  f = np.ones(N) * c
  Step load       :  f[:k_s] = c
  Sinusoidal      :  f = A * np.sin(n * np.pi * x / L)
  Case A          :  multiple disjoint point loads
  Case B          :  sum of sinusoidal modes
  Case C          :  uniform load + single point perturbation

For unrecognised patterns the compiler falls back to the Möttönen
general state-preparation routine (requires the load vector to be
passed explicitly via the ``fallback_vector`` argument).

References
----------
  Möttönen et al., Quantum Inf. Comput. 5(6), 467-473, 2005.
  Shende, Markov, Bullock, IEEE TCAD 25(6), 1000-1010, 2006.
  Harrow, Hassidim, Lloyd, Phys. Rev. Lett. 103, 150502, 2009.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union

from qiskit import QuantumCircuit

from .recognizer import recognise, LoadPattern, LoadType
from .synthesizer import synthesize
from .emitter import emit_code


__all__ = ["encode", "EncodingInfo"]
__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class EncodingInfo:
    """
    Metadata about the encoding produced by :func:`encode`.

    Attributes
    ----------
    load_type : str
        Human-readable name of the recognised load pattern.
    N : int
        Number of force-vector components (must be a power of 2).
    m : int
        Number of qubits  (m = log2(N)).
    gate_count : int
        Total number of gates in the returned circuit.
    gate_complexity : str
        Asymptotic gate complexity class (e.g. "O(m)", "O(m²)").
    fallback : bool
        True if the Möttönen fallback was used (pattern unrecognised).
    params : dict
        Extracted load parameters (amplitudes, mode numbers, etc.).
    circuit_code : str
        Standalone Python/Qiskit source code that builds the same
        circuit.  Can be copy-pasted, saved to a .py file, or tweaked
        without PyEncode installed.
    """
    load_type: str
    N: int
    m: int
    gate_count: int
    gate_complexity: str
    fallback: bool
    params: dict
    circuit_code: str = ""

    def __str__(self) -> str:
        lines = [
            f"PyEncode  v{__version__}",
            f"  Load type   : {self.load_type}",
            f"  N           : {self.N}  (m = {self.m} qubits)",
            f"  Gate count  : {self.gate_count}",
            f"  Complexity  : {self.gate_complexity}",
            f"  Fallback    : {'yes (Shende)' if self.fallback else 'no'}",
        ]
        if self.params:
            lines.append(f"  Parameters  : {self.params}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Complexity table
# ---------------------------------------------------------------------------

_COMPLEXITY = {
    LoadType.POINT_LOAD:        "O(m)",
    LoadType.UNIFORM_LOAD:      "O(m)",
    LoadType.STEP_LOAD:         "O(m)",
    LoadType.SQUARE_LOAD:       "O(m)",
    LoadType.SINUSOIDAL_LOAD:   "O(m²)",
    LoadType.COSINE_LOAD:       "O(m²)",
    LoadType.MULTI_POINT_LOAD:  "O(m · L)",
    LoadType.MULTI_SIN_LOAD:    "O(m²)",
    LoadType.UNIFORM_SPIKE_LOAD: "O(2^m) [impl]; O(m²) analytical",
    LoadType.UNKNOWN:           "O(2^m)  [Shende fallback]",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode(
    code: str,
    fallback_vector: Optional[np.ndarray] = None,
    *,
    validate: bool = False,
    atol: float = 1e-6,
) -> tuple[QuantumCircuit, EncodingInfo]:
    """
    Compile Python force-vector construction code into a Qiskit circuit.

    Parameters
    ----------
    code : str
        Python source code that constructs a 1-D NumPy load vector.
        The vector must have length N = 2^m for some integer m ≥ 1.
        The variable holding the vector should be named ``f``
        (or ``N`` / ``n_nodes`` for the grid size variable).

    fallback_vector : np.ndarray, optional
        If the pattern is not recognised and you want to use the Möttönen
        fallback, supply the explicit amplitude vector here.
        Shape must be (N,).

    validate : bool, optional (default False)
        If True, execute the circuit on Qiskit's statevector simulator
        and verify that the output state matches the expected amplitudes
        to within ``atol``.  Raises ``ValueError`` on mismatch.
        Requires the load vector to be computable (either recognisable
        or provided via ``fallback_vector``).

    atol : float, optional (default 1e-6)
        Absolute tolerance used when ``validate=True``.

    Returns
    -------
    circuit : QuantumCircuit
        The compiled quantum circuit preparing the encoded state.
    info : EncodingInfo
        Metadata about the encoding (load type, gate count, etc.).

    Raises
    ------
    ValueError
        If N is not a power of 2, or if ``validate=True`` and the circuit
        does not produce the expected state within tolerance.
    RuntimeError
        If the pattern is unrecognised and no ``fallback_vector`` is given.

    Examples
    --------
    Point load:

    >>> circuit, info = encode('''
    ... import numpy as np
    ... N = 8
    ... f = np.zeros(N)
    ... f[3] = 5.0
    ... ''')
    >>> print(info)

    Uniform load:

    >>> circuit, info = encode('''
    ... import numpy as np
    ... N = 16
    ... f = np.ones(N) * 2.5
    ... ''')

    Sinusoidal load:

    >>> circuit, info = encode('''
    ... import numpy as np
    ... N = 8
    ... L = 1.0
    ... x = np.linspace(0, L, N)
    ... f = np.sin(np.pi * x / L)
    ... ''')
    """
    # 1. Pattern recognition
    pattern: LoadPattern = recognise(code)

    # 2. Handle unrecognised pattern
    if pattern.load_type == LoadType.UNKNOWN:
        if fallback_vector is None:
            raise RuntimeError(
                "PyEncode: load pattern not recognised and no fallback_vector "
                "provided.\n"
                "Either restructure the code to match a supported pattern, or "
                "pass the explicit load vector as fallback_vector=f."
            )
        N = len(fallback_vector)
        if N < 2 or (N & (N - 1)) != 0:
            raise ValueError(f"fallback_vector length {N} is not a power of 2.")
        pattern.N = N
        pattern.params = {"amplitudes": fallback_vector.astype(complex)}

    N = pattern.N
    m = int(round(math.log2(N)))

    # 3. Circuit synthesis
    circuit = synthesize(pattern)

    # 4. Optional validation against statevector simulator
    if validate:
        _validate_circuit(circuit, pattern, fallback_vector, atol)

    # 5. Build info object — transpile to primitive basis for honest gate count
    try:
        from qiskit import transpile as qk_transpile
        t = qk_transpile(circuit,
                         basis_gates=['cx', 'u', 'x', 'h', 'ry', 'rz', 'rx', 'p'],
                         optimization_level=0)
        total_gates = sum(t.count_ops().values())
    except Exception:
        total_gates = sum(circuit.decompose(reps=10).count_ops().values())
    complexity = _COMPLEXITY.get(pattern.load_type, "unknown")
    fallback = (pattern.load_type == LoadType.UNKNOWN)

    info = EncodingInfo(
        load_type=pattern.load_type.name,
        N=N,
        m=m,
        gate_count=total_gates,
        gate_complexity=complexity,
        fallback=fallback,
        params=_sanitise_params(pattern.params),
        circuit_code=emit_code(pattern),
    )

    return circuit, info


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_circuit(
    circuit: QuantumCircuit,
    pattern: LoadPattern,
    fallback_vector: Optional[np.ndarray],
    atol: float,
):
    """
    Run the circuit on Qiskit's statevector simulator and compare to the
    expected normalised amplitude vector.
    """
    try:
        from qiskit.quantum_info import Statevector
    except ImportError:
        raise ImportError(
            "Statevector validation requires qiskit-terra >= 0.20. "
            "Install with: pip install qiskit"
        )

    sv = Statevector(circuit)
    simulated = np.array(sv)

    # Build the expected vector
    expected = _build_expected_vector(pattern, fallback_vector)
    if expected is None:
        return  # cannot validate without reference

    norm = np.linalg.norm(expected)
    if norm < 1e-14:
        raise ValueError("Expected amplitude vector is zero.")
    expected = expected / norm

    # Compare magnitudes (global phase is irrelevant)
    if not np.allclose(np.abs(simulated), np.abs(expected), atol=atol):
        max_err = np.max(np.abs(np.abs(simulated) - np.abs(expected)))
        raise ValueError(
            f"Validation failed: max amplitude error = {max_err:.2e} > atol={atol}.\n"
            "The synthesised circuit does not match the expected load vector."
        )


def _build_expected_vector(
    pattern: LoadPattern,
    fallback_vector: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Return the expected real amplitude vector for validation."""
    N = pattern.N

    if fallback_vector is not None:
        return fallback_vector.astype(float)

    lt = pattern.load_type
    p  = pattern.params

    if lt == LoadType.POINT_LOAD:
        f = np.zeros(N)
        f[p["k"]] = p["P"]
        return f

    if lt == LoadType.UNIFORM_LOAD:
        return np.full(N, p["c"])

    if lt == LoadType.STEP_LOAD:
        f = np.zeros(N)
        f[:p["k_s"]] = p["c"]
        return f

    if lt in (LoadType.SINUSOIDAL_LOAD, LoadType.COSINE_LOAD):
        k = np.arange(N)
        return p["A"] * np.sin(p["n"] * math.pi * k / N)

    if lt == LoadType.MULTI_POINT_LOAD:
        f = np.zeros(N)
        for load in p["loads"]:
            f[load["k"]] = load["P"]
        return f

    if lt == LoadType.MULTI_SIN_LOAD:
        k = np.arange(N)
        f = np.zeros(N)
        for mode in p["modes"]:
            f += mode["A"] * np.sin(mode["n"] * math.pi * k / N)
        return f

    if lt == LoadType.UNIFORM_SPIKE_LOAD:
        f = np.full(N, p["c"])
        f[p["k"]] = p["delta"]
        return f

    return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sanitise_params(params: dict) -> dict:
    """Convert numpy arrays to plain Python for display in EncodingInfo."""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            out[k] = f"<array shape={v.shape}>"
        else:
            out[k] = v
    return out