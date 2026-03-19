"""
pyencode
==========
A toolkit for quantum amplitude encoding of structured load vectors.

PyEncode synthesises efficient Qiskit quantum circuits for structured
state vectors (point loads, sinusoidal modes, step functions, etc.)
using closed-form circuit templates instead of general O(2^m) routines.

Three entry points
------------------

**encode_params** — specify the vector type and parameters directly:

>>> circuit, info = encode_params(DISCRETE(k=3, P=5.0), N=8)
>>> circuit, info = encode_params(SINE(n=3, A=1.0), N=64)

**encode_vector** — pass a vector, optionally declare the type:

>>> f = np.zeros(8); f[3] = 5.0
>>> circuit, info = encode_vector(f)                        # auto-detect
>>> circuit, info = encode_vector(f, vector_type="DISCRETE") # with hint

**encode_python** — compile Python source code into a circuit:

>>> circuit, info = encode_python('''
... import numpy as np
... N = 8; f = np.zeros(N); f[3] = 5.0
... ''')

Supported vector patterns
--------------------------
  DISCRETE       :  f[k] = P
  UNIFORM        :  f = ones(N) * c
  STEP           :  f[:k_s] = c
  SQUARE         :  f[k1:k2] = c
  SINE           :  f = A * sin(n * pi * k / N + phi)
  COSINE         :  f = A * cos(n * pi * k / N + phi)
  MULTI_DISCRETE :  multiple disjoint point loads
  MULTI_SINE     :  sum of sinusoidal modes

References
----------
  Mottonen et al., Quantum Inf. Comput. 5(6), 467-473, 2005.
  Shende, Markov, Bullock, IEEE TCAD 25(6), 1000-1010, 2006.
  Harrow, Hassidim, Lloyd, Phys. Rev. Lett. 103, 150502, 2009.
"""

import builtins as _builtins
import inspect
import math
import textwrap
import warnings
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union

from qiskit import QuantumCircuit

from .recognizer import recognise, LoadPattern, VectorType
from .synthesizer import synthesize
from .emitter import emit_code
from .extractor import extract, auto_detect

# Backward compat
LoadType = VectorType


__all__ = [
    "encode_params",
    "encode_vector",
    "encode_python",
    "EncodingInfo",
    "VectorType",
    "LoadType",
    # Constructor classes
    "DISCRETE",
    "UNIFORM",
    "STEP",
    "SQUARE",
    "SINE",
    "COSINE",
    "MULTI_DISCRETE",
    "MULTI_SINE",
]
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


# ===================================================================
# PUBLIC API
# ===================================================================

# -------------------------------------------------------------------
# encode_params — direct parameter specification via typed constructors
# -------------------------------------------------------------------

def encode_params(
    vector_obj: Union[_VectorObj, list, "VectorType", str],
    N: int,
    *,
    validate: bool = False,
    tol: float = 1e-6,
    # Backward-compatible kwargs for string/enum usage
    **params,
) -> tuple[QuantumCircuit, EncodingInfo]:
    """
    Synthesise a circuit from a typed vector constructor and parameters.

    Parameters
    ----------
    vector_obj : VectorObj, list, VectorType, or str
        A typed constructor (e.g. ``SINE(n=3, A=1.0)``), a list of
        constructors for composite vectors, or a VectorType enum / string
        for backward compatibility.
    N : int
        Vector length (must be a power of 2).
    validate : bool
        If True, simulate the circuit and verify the output state.
    tol : float
        Tolerance for validation.
    **params
        Type-specific parameters (only for string/enum backward compat).

    Examples
    --------
    >>> circuit, info = encode_params(DISCRETE(k=3, P=5.0), N=8)
    >>> circuit, info = encode_params(SINE(n=3, A=1.0), N=64)
    >>> circuit, info = encode_params(
    ...     [SINE(n=1, A=2.0), SINE(n=3, A=1.0)], N=64)
    """
    if N < 2 or (N & (N - 1)) != 0:
        raise ValueError(f"N={N} is not a power of 2.")

    # Handle typed constructor objects
    if isinstance(vector_obj, _VectorObj):
        vtype = vector_obj.vector_type
        validated_params = _validate_params(vtype, N, vector_obj.params)
        pattern = LoadPattern(vtype, N=N, params=validated_params)
        return _synthesise_and_build_info(
            pattern, fallback_vector=None,
            validate=validate, tol=tol,
        )

    # Handle list of constructors (composite)
    if isinstance(vector_obj, list):
        return _encode_composite(vector_obj, N, validate=validate, tol=tol)

    # Backward compat: string or VectorType enum
    vtype = _normalise_vector_type(vector_obj)
    validated_params = _validate_params(vtype, N, params)
    pattern = LoadPattern(vtype, N=N, params=validated_params)
    return _synthesise_and_build_info(
        pattern, fallback_vector=None,
        validate=validate, tol=tol,
    )


def _encode_composite(
    components: list,
    N: int,
    validate: bool,
    tol: float,
) -> tuple[QuantumCircuit, EncodingInfo]:
    """Synthesise a composite vector from a list of typed constructors."""
    if not components:
        raise ValueError("Empty component list.")

    # Validate all are VectorObj instances
    for i, comp in enumerate(components):
        if not isinstance(comp, _VectorObj):
            raise TypeError(
                f"Component {i} is not a VectorObj, got {type(comp).__name__}."
            )

    # Check if this is really a MULTI_DISCRETE or MULTI_SINE
    all_discrete = all(isinstance(c, DISCRETE) for c in components)
    all_sine = all(isinstance(c, SINE) for c in components)

    if all_discrete:
        loads = [{"k": c.params["k"], "P": c.params["P"]} for c in components]
        combined = MULTI_DISCRETE(
            vectors=[c for c in components]
        )
        validated_params = _validate_params(VectorType.MULTI_DISCRETE, N, combined.params)
        pattern = LoadPattern(VectorType.MULTI_DISCRETE, N=N, params=validated_params)
        return _synthesise_and_build_info(
            pattern, fallback_vector=None,
            validate=validate, tol=tol,
        )

    if all_sine:
        combined = MULTI_SINE(
            modes=[c for c in components]
        )
        validated_params = _validate_params(VectorType.MULTI_SINE, N, combined.params)
        pattern = LoadPattern(VectorType.MULTI_SINE, N=N, params=validated_params)
        return _synthesise_and_build_info(
            pattern, fallback_vector=None,
            validate=validate, tol=tol,
        )

    # General composite: build combined vector, try auto-detect
    f = np.zeros(N)
    for comp in components:
        f += _build_component_vector(comp, N)

    norm = np.linalg.norm(f)
    if norm < 1e-14:
        raise ValueError("Composite vector is the zero vector.")

    # Try auto-detect on the combined vector
    try:
        detected_type, detected_params = auto_detect(f, tol=tol)
        pattern = LoadPattern(detected_type, N=N, params=detected_params)
        return _synthesise_and_build_info(
            pattern, fallback_vector=None,
            validate=validate, tol=tol,
        )
    except ValueError:
        pass

    # No pattern matched: Shende fallback with warning
    warnings.warn(
        "Composite vector does not match any known pattern: "
        "falling back to Shende O(2^m) synthesis.",
        stacklevel=3,
    )
    pattern = LoadPattern(VectorType.UNKNOWN, N=N,
                          params={"amplitudes": (f / norm).astype(complex)})
    return _synthesise_and_build_info(
        pattern, fallback_vector=f,
        validate=validate, tol=tol,
    )


def _build_component_vector(comp: _VectorObj, N: int) -> np.ndarray:
    """Materialise a single component vector from its constructor."""
    p = comp.params
    k = np.arange(N)

    if comp.vector_type == VectorType.DISCRETE:
        f = np.zeros(N); f[p["k"]] = p.get("P", 1.0); return f
    if comp.vector_type == VectorType.UNIFORM:
        return np.full(N, p.get("c", 1.0))
    if comp.vector_type == VectorType.STEP:
        f = np.zeros(N); f[:p["k_s"]] = p.get("c", 1.0); return f
    if comp.vector_type == VectorType.SQUARE:
        f = np.zeros(N); f[p["k1"]:p["k2"]] = p.get("c", 1.0); return f
    if comp.vector_type == VectorType.SINE:
        return p.get("A", 1.0) * np.sin(2 * math.pi * p["n"] * k / N + p.get("phi", 0.0))
    if comp.vector_type == VectorType.COSINE:
        return p.get("A", 1.0) * np.cos(2 * math.pi * p["n"] * k / N + p.get("phi", 0.0))

    raise TypeError(f"Cannot materialise component of type {comp.vector_type.name}.")


# -------------------------------------------------------------------
# encode_vector — vector input, optional type hint
# -------------------------------------------------------------------

def encode_vector(
    f: np.ndarray,
    *,
    vector_type: Optional[Union[VectorType, str]] = None,
    validate: bool = False,
    tol: float = 1e-6,
) -> tuple[QuantumCircuit, EncodingInfo]:
    """
    Synthesise a circuit from an explicit load vector.

    If ``vector_type`` is given, parameters are extracted numerically
    and verified against the declared type.  If omitted, the vector
    type is auto-detected.

    Parameters
    ----------
    f : np.ndarray
        Load vector of length N (must be a power of 2).
    vector_type : VectorType or str, optional
        Declared vector type.  If None, auto-detected.
    validate : bool
        If True, simulate and verify.
    tol : float
        Tolerance for extraction and validation.

    Examples
    --------
    >>> f = np.zeros(8); f[3] = 5.0
    >>> circuit, info = encode_vector(f)                          # auto-detect
    >>> circuit, info = encode_vector(f, vector_type="DISCRETE")  # with hint
    """
    f = np.asarray(f, dtype=float).ravel()
    N = len(f)

    if N < 2:
        raise ValueError(f"Vector length {N} is too small, need >= 2.")
    if N & (N - 1) != 0:
        raise ValueError(f"Vector length {N} is not a power of 2.")

    if vector_type is not None:
        # Declared type: extract params for that specific type
        vector_type = _normalise_vector_type(vector_type)
        params = extract(f, vector_type, tol=tol)
    else:
        # Auto-detect
        vector_type, params = auto_detect(f, tol=tol)

    pattern = LoadPattern(vector_type, N=N, params=params)

    # Verify reconstruction
    expected = _build_expected_vector(pattern, fallback_vector=None)
    if expected is not None:
        norm_f = np.linalg.norm(f)
        norm_expected = np.linalg.norm(expected)
        if norm_f > 1e-14 and norm_expected > 1e-14:
            diff = np.linalg.norm(f / norm_f - expected / norm_expected)
            if diff > tol:
                raise ValueError(
                    f"Verification failed for vector_type={vector_type.name}: "
                    f"||f_normalised - f_reconstructed|| = {diff:.2e} > "
                    f"tol={tol}.  The vector may not match the declared type."
                )

    return _synthesise_and_build_info(
        pattern, fallback_vector=None,
        validate=validate, tol=tol,
    )


# -------------------------------------------------------------------
# encode_python — compile Python source code
# -------------------------------------------------------------------

def encode_python(
    code,
    *,
    vector_type: Optional[Union[VectorType, str]] = None,
    N: Optional[int] = None,
    validate: bool = False,
    tol: float = 1e-6,
) -> tuple[QuantumCircuit, EncodingInfo]:
    """
    Compile Python source code into a Qiskit circuit.

    ``code`` accepts a string of valid Python source or a callable;
    in the latter case the source is retrieved automatically via
    ``inspect.getsource``.

    If ``vector_type`` is not given, the AST recogniser infers the
    pattern from the code structure.  If ``vector_type`` is given,
    the code is *executed* to produce a vector, then parameters
    are extracted numerically.

    Parameters
    ----------
    code : str or callable
        Python source that constructs a load vector named ``f``,
        or a callable whose source will be inspected.
    vector_type : VectorType or str, optional
        If given, bypass AST recognition: execute the code and
        extract parameters numerically for the declared type.
    N : int, optional
        Vector length hint (used when AST cannot determine N).
    validate : bool
        If True, simulate and verify.
    tol : float
        Tolerance for extraction and validation.

    Examples
    --------
    AST recognition (automatic):

    >>> circuit, info = encode_python('''
    ... import numpy as np
    ... N = 8; f = np.zeros(N); f[3] = 5.0
    ... ''')

    With type hint (executes code, handles loops etc.):

    >>> circuit, info = encode_python('''
    ... import numpy as np
    ... N = 8; f = np.zeros(N)
    ... for i in range(N):
    ...     if i == 3: f[i] = 5.0
    ... ''', vector_type="DISCRETE")
    """
    # Accept callable: retrieve source via inspect.getsource
    if callable(code) and not isinstance(code, str):
        code = textwrap.dedent(inspect.getsource(code))

    if vector_type is not None:
        # Path 2: Execute code -> get vector -> extract params
        vector_type = _normalise_vector_type(vector_type)
        f = _execute_code(code)

        vec_N = len(f)
        if vec_N < 2 or (vec_N & (vec_N - 1)) != 0:
            raise ValueError(
                f"Executed code produced vector of length {vec_N}, "
                f"which is not a power of 2."
            )

        params = extract(f, vector_type, tol=tol)
        pattern = LoadPattern(vector_type, N=vec_N, params=params)

        # Classical validation (mandatory on Path 2)
        expected = _build_expected_vector(pattern, fallback_vector=None)
        if expected is not None:
            norm_f = np.linalg.norm(f)
            norm_expected = np.linalg.norm(expected)
            if norm_f > 1e-14 and norm_expected > 1e-14:
                diff = np.linalg.norm(f / norm_f - expected / norm_expected)
                if diff > tol:
                    raise ValueError(
                        f"Verification failed for vector_type={vector_type.name}: "
                        f"the executed code does not produce a vector "
                        f"matching the declared type "
                        f"(normalised diff = {diff:.2e})."
                    )

        return _synthesise_and_build_info(
            pattern, fallback_vector=None,
            validate=validate, tol=tol,
        )

    # Path 1: AST recognition (no vector_type hint)
    pattern: LoadPattern = recognise(code)

    if pattern.load_type == VectorType.UNKNOWN:
        # Fallback: execute the code to get the vector, use Shende
        warnings.warn(
            "PyEncode: pattern not recognised by the AST recogniser. "
            "Falling back to Shende's O(2^m) general state preparation.",
            stacklevel=2,
        )
        try:
            f = _execute_code(code)
            vec_N = len(f)
            if vec_N < 2 or (vec_N & (vec_N - 1)) != 0:
                raise ValueError(
                    f"Executed code produced vector of length {vec_N}, "
                    f"which is not a power of 2."
                )
            pattern.N = vec_N
            pattern.params = {"amplitudes": f.astype(complex)}
        except Exception as e:
            raise RuntimeError(
                "PyEncode: load pattern not recognised and code execution "
                f"failed: {e}\n"
                "Either restructure the code to match a supported "
                "pattern or use vector_type= to declare the pattern."
            ) from e

    return _synthesise_and_build_info(
        pattern, fallback_vector=None,
        validate=validate, tol=tol,
    )


# ===================================================================
# INTERNAL HELPERS
# ===================================================================

# -------------------------------------------------------------------
# Code execution (for encode_python with vector_type hint)
# -------------------------------------------------------------------

def _execute_code(code: str) -> np.ndarray:
    """Execute user code and return the resulting ``f`` vector."""
    namespace = {"__builtins__": _builtins, "np": np, "numpy": np, "math": math}
    exec(compile(code, "<encode_python>", "exec"), namespace)

    if "f" not in namespace:
        raise RuntimeError(
            "Executed code does not define variable 'f'.  "
            "The code must assign the load vector to a variable "
            "named 'f'."
        )
    f = np.asarray(namespace["f"], dtype=float).ravel()
    return f


# -------------------------------------------------------------------
# Vector type normalisation
# -------------------------------------------------------------------

# Mapping from old names to new enum values for backward compat
_OLD_NAME_MAP = {
    "POINT_LOAD": VectorType.DISCRETE,
    "UNIFORM_LOAD": VectorType.UNIFORM,
    "STEP_LOAD": VectorType.STEP,
    "SQUARE_LOAD": VectorType.SQUARE,
    "SINUSOIDAL_LOAD": VectorType.SINE,
    "COSINE_LOAD": VectorType.COSINE,
    "MULTI_POINT_LOAD": VectorType.MULTI_DISCRETE,
    "MULTI_SIN_LOAD": VectorType.MULTI_SINE,
    "UNIFORM_SPIKE_LOAD": VectorType.UNIFORM_SPIKE,
}


def _normalise_vector_type(vector_type: Union[VectorType, str]) -> VectorType:
    """Convert string to VectorType enum; reject UNKNOWN."""
    if isinstance(vector_type, str):
        upper = vector_type.upper()
        # Try new names first
        try:
            vector_type = VectorType[upper]
        except KeyError:
            # Try old names for backward compat
            if upper in _OLD_NAME_MAP:
                vector_type = _OLD_NAME_MAP[upper]
            else:
                valid = [vt.name for vt in VectorType if vt != VectorType.UNKNOWN]
                raise ValueError(
                    f"Unknown vector_type '{vector_type}'.  "
                    f"Valid values: {valid}"
                )

    if vector_type == VectorType.UNKNOWN:
        raise ValueError(
            "vector_type=UNKNOWN is not valid.  "
            "Specify a concrete vector type like DISCRETE, "
            "SINE, etc."
        )
    return vector_type


# -------------------------------------------------------------------
# Parameter validation (for encode_params)
# -------------------------------------------------------------------

def _validate_params(vector_type: VectorType, N: int, params: dict) -> dict:
    """Validate user-supplied parameters against the schema."""
    schema = _PARAM_SCHEMAS.get(vector_type)
    if schema is None:
        raise TypeError(
            f"vector_type={vector_type.name} is not supported for "
            f"direct parameter specification."
        )

    required = schema["required"]
    optional = schema["optional"]
    allowed = required | optional
    description = schema["description"]

    missing = required - set(params.keys())
    if missing:
        raise TypeError(
            f"Missing required parameters for {vector_type.name}: "
            f"{sorted(missing)}.  Expected: {description}"
        )

    unexpected = set(params.keys()) - allowed
    if unexpected:
        raise TypeError(
            f"Unexpected parameters for {vector_type.name}: "
            f"{sorted(unexpected)}.  Expected: {description}"
        )

    result = dict(params)

    # Defaults — amplitude / constant params are absorbed by
    # normalisation and default to 1.0 when omitted.
    if vector_type in (VectorType.SINE, VectorType.COSINE):
        result.setdefault("phi", 0.0)
        result.setdefault("A", 1.0)
    if vector_type == VectorType.DISCRETE:
        result.setdefault("P", 1.0)
    if vector_type in (VectorType.UNIFORM, VectorType.STEP, VectorType.SQUARE):
        result.setdefault("c", 1.0)

    # Type-specific validation
    if vector_type == VectorType.DISCRETE:
        k = int(result["k"])
        if k < 0 or k >= N:
            raise ValueError(f"k={k} out of range [0, {N}).")
        result["k"] = k
        result["P"] = float(result["P"])

    elif vector_type == VectorType.UNIFORM:
        result["c"] = float(result["c"])

    elif vector_type == VectorType.STEP:
        k_s = int(result["k_s"])
        if k_s < 1 or k_s >= N:
            raise ValueError(f"k_s={k_s} out of range [1, {N}).")
        result["k_s"] = k_s
        result["c"] = float(result["c"])

    elif vector_type == VectorType.SQUARE:
        k1 = int(result["k1"])
        k2 = int(result["k2"])
        if k1 < 0 or k2 > N or k1 >= k2:
            raise ValueError(f"Invalid range [{k1}, {k2}) for N={N}.")
        result["k1"] = k1
        result["k2"] = k2
        result["c"] = float(result["c"])

    elif vector_type in (VectorType.SINE, VectorType.COSINE):
        n = int(result["n"])
        if n < 1:
            raise ValueError(f"Mode n={n} must be >= 1.")
        result["n"] = n
        result["A"] = float(result["A"])
        result["phi"] = float(result["phi"])

    elif vector_type == VectorType.MULTI_DISCRETE:
        loads = result["loads"]
        if not isinstance(loads, list) or len(loads) < 2:
            raise TypeError("loads must be a list of >= 2 entries.")
        validated_loads = []
        for entry in loads:
            k = int(entry["k"])
            if k < 0 or k >= N:
                raise ValueError(f"Load index k={k} out of range [0, {N}).")
            validated_loads.append({"k": k, "P": float(entry["P"])})
        result["loads"] = validated_loads

    elif vector_type == VectorType.MULTI_SINE:
        modes = result["modes"]
        if not isinstance(modes, list) or len(modes) < 2:
            raise TypeError("modes must be a list of >= 2 entries.")
        validated_modes = []
        for entry in modes:
            n = int(entry["n"])
            if n < 1:
                raise ValueError(f"Mode n={n} must be >= 1.")
            validated_modes.append({"n": n, "A": float(entry["A"])})
        result["modes"] = validated_modes

    return result


# -------------------------------------------------------------------
# Shared synthesis + info builder
# -------------------------------------------------------------------

def _synthesise_and_build_info(
    pattern: LoadPattern,
    fallback_vector: Optional[np.ndarray],
    validate: bool,
    tol: float,
) -> tuple[QuantumCircuit, EncodingInfo]:
    """Synthesise circuit from pattern and build EncodingInfo."""
    N = pattern.N
    m = int(round(math.log2(N)))

    circuit = synthesize(pattern)

    validated = False
    if validate:
        _validate_circuit(circuit, pattern, fallback_vector, tol)
        validated = True

    try:
        from qiskit import transpile as qk_transpile
        t = qk_transpile(circuit,
                         basis_gates=['cx', 'u', 'x', 'h', 'ry', 'rz', 'rx', 'p'],
                         optimization_level=0)
        total_gates = sum(t.count_ops().values())
    except Exception:
        total_gates = sum(circuit.decompose(reps=10).count_ops().values())

    complexity = _COMPLEXITY.get(pattern.load_type, "unknown")

    info = EncodingInfo(
        vector_type=pattern.load_type.name,
        N=N,
        m=m,
        gate_count=total_gates,
        complexity=complexity,
        validated=validated,
        params=_sanitise_params(pattern.params),
        circuit_code=emit_code(pattern),
    )

    return circuit, info


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

def _validate_circuit(
    circuit: QuantumCircuit,
    pattern: LoadPattern,
    fallback_vector: Optional[np.ndarray],
    tol: float,
):
    """Simulate and compare to expected amplitude vector."""
    from qiskit.quantum_info import Statevector

    sv = Statevector(circuit)
    simulated = np.array(sv)

    expected = _build_expected_vector(pattern, fallback_vector)
    if expected is None:
        return

    norm = np.linalg.norm(expected)
    if norm < 1e-14:
        raise ValueError("Expected amplitude vector is zero.")
    expected = expected / norm

    if not np.allclose(np.abs(simulated), np.abs(expected), atol=tol):
        max_err = np.max(np.abs(np.abs(simulated) - np.abs(expected)))
        raise ValueError(
            f"Validation failed: max amplitude error = {max_err:.2e} "
            f"> tol={tol}."
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

    if lt == VectorType.DISCRETE:
        f = np.zeros(N); f[p["k"]] = p.get("P", 1.0); return f

    if lt == VectorType.UNIFORM:
        return np.full(N, p.get("c", 1.0))

    if lt == VectorType.STEP:
        f = np.zeros(N); f[:p["k_s"]] = p.get("c", 1.0); return f

    if lt == VectorType.SQUARE:
        f = np.zeros(N); f[p["k1"]:p["k2"]] = p.get("c", 1.0); return f

    if lt == VectorType.SINE:
        k = np.arange(N)
        return p.get("A", 1.0) * np.sin(2 * math.pi * p["n"] * k / N + p.get("phi", 0.0))

    if lt == VectorType.COSINE:
        k = np.arange(N)
        return p.get("A", 1.0) * np.cos(2 * math.pi * p["n"] * k / N + p.get("phi", 0.0))

    if lt == VectorType.MULTI_DISCRETE:
        f = np.zeros(N)
        for load in p["loads"]:
            f[load["k"]] = load["P"]
        return f

    if lt == VectorType.MULTI_SINE:
        k = np.arange(N)
        f = np.zeros(N)
        for mode in p["modes"]:
            f += mode["A"] * np.sin(2 * math.pi * mode["n"] * k / N)
        return f

    return None


# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------

def _sanitise_params(params: dict) -> dict:
    """Convert numpy arrays to plain Python for display."""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            out[k] = f"<array shape={v.shape}>"
        else:
            out[k] = v
    return out
