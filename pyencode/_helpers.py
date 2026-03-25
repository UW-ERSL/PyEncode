"""
pyencode._helpers
=================
Shared internal helpers used by the encode_* entry points.
"""

import builtins as _builtins
import math
import numpy as np
from typing import Optional, Union

from qiskit import QuantumCircuit

from .recognizer import VectorType, LoadPattern
from .synthesizer import synthesize
from .emitter import emit_code
from .config import BASIS_GATES, OPTIMIZATION_LEVEL, DECOMPOSE_REPS
from .types import (
    _VectorObj, DISCRETE, SINE,
    MULTI_DISCRETE, MULTI_SINE,
    EncodingInfo,
    _COMPLEXITY, _PARAM_SCHEMAS,
)


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


def _normalize_vector_type(vector_type: Union[VectorType, str]) -> VectorType:
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


    elif vector_type == VectorType.SPARSE:
        result = _validate_sparse_params(result, N)

    elif vector_type == VectorType.FOURIER:
        result = _validate_fourier_params(result, N)

    return result


# -------------------------------------------------------------------
# Shared synthesis + info builder
# -------------------------------------------------------------------

def _synthesize_and_build_info(
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
        # Count raw logical gates (before decomposition).
        # This matches the "Raw" column in the paper's gate count table.
        # mcry, ccx, cry etc. are counted as single gates at this level.
        total_gates = sum(circuit.count_ops().values())
    except Exception:
        total_gates = sum(circuit.count_ops().values())

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

    if lt == VectorType.SPARSE:
        f = np.zeros(N)
        for load in p["loads"]:
            f[load["k"]] = load["P"]
        return f

    if lt == VectorType.FOURIER:
        k = np.arange(N)
        f = np.zeros(N)
        for mode in p["modes"]:
            f += mode["A"] * np.sin(2 * math.pi * mode["n"] * k / N + mode.get("phi", 0.0))
        return f

    return None


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


# -------------------------------------------------------------------
# Composite encoding helpers
# -------------------------------------------------------------------

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
        combined = MULTI_DISCRETE(
            vectors=[c for c in components]
        )
        validated_params = _validate_params(VectorType.MULTI_DISCRETE, N, combined.params)
        pattern = LoadPattern(VectorType.MULTI_DISCRETE, N=N, params=validated_params)
        return _synthesize_and_build_info(
            pattern, fallback_vector=None,
            validate=validate, tol=tol,
        )

    if all_sine:
        combined = MULTI_SINE(
            modes=[c for c in components]
        )
        validated_params = _validate_params(VectorType.MULTI_SINE, N, combined.params)
        pattern = LoadPattern(VectorType.MULTI_SINE, N=N, params=validated_params)
        return _synthesize_and_build_info(
            pattern, fallback_vector=None,
            validate=validate, tol=tol,
        )

    # General composite: synthesize each component independently,
    # combine via LCU (linear combination of unitaries).
    from .synthesizer import synthesize as _synthesize

    K = len(components)
    m = int(round(math.log2(N)))

    # Build each component vector to get weights (norms)
    component_vectors = []
    component_patterns = []
    for comp in components:
        f_comp = _build_component_vector(comp, N)
        component_vectors.append(f_comp)
        validated = _validate_params(comp.vector_type, N, comp.params)
        component_patterns.append(LoadPattern(comp.vector_type, N=N, params=validated))

    weights = np.array([np.linalg.norm(v) for v in component_vectors])
    total_norm = np.linalg.norm(weights)
    if total_norm < 1e-14:
        raise ValueError("Composite vector is the zero vector.")
    weights_norm = weights / total_norm

    # Synthesise each component circuit independently
    component_circuits = []
    for pat in component_patterns:
        component_circuits.append(_synthesize(pat))

    # Build combined vector for validation / info
    f_combined = sum(component_vectors)

    # If only 1 component, just return it
    if K == 1:
        info_pattern = component_patterns[0]
        return _synthesize_and_build_info(
            info_pattern, fallback_vector=None,
            validate=validate, tol=tol,
        )

    # LCU synthesis: ancilla register selects component
    # For K components, need ceil(log2(K)) ancilla qubits
    n_anc = math.ceil(math.log2(K))
    total_qubits = m + n_anc

    qc = QuantumCircuit(total_qubits, name="composite_lcu")
    data_qubits = list(range(m))
    anc_qubits = list(range(m, total_qubits))

    # Step 1: Ry-tree on ancilla to distribute weights
    _prepare_amplitude_ancilla(qc, weights_norm, K, n_anc, anc_qubits)

    # Step 2: Apply each component circuit controlled on ancilla state
    for i, circ_i in enumerate(component_circuits):
        # Build control state for component i
        ctrl_state = format(i, f'0{n_anc}b')[::-1]  # LSB first

        # Flip ancilla bits where ctrl_state is 0
        flip_qubits = [anc_qubits[b] for b in range(n_anc) if ctrl_state[b] == '0']
        for q in flip_qubits:
            qc.x(q)

        # Apply component circuit controlled on all ancilla = |1...1>
        controlled_circ = circ_i.to_gate().control(n_anc)
        qc.append(controlled_circ, anc_qubits + data_qubits)

        # Unflip
        for q in flip_qubits:
            qc.x(q)

    # Build info
    total_gates = sum(qc.decompose(reps=3).count_ops().values())
    complexity = f"O({K} * component)"

    info = EncodingInfo(
        vector_type="COMPOSITE",
        N=N,
        m=m,
        gate_count=total_gates,
        complexity=complexity,
        validated=False,
        params={"components": [c.vector_type.name for c in components]},
        circuit_code="",
    )

    if validate:
        from qiskit.quantum_info import Statevector
        sv_full = np.array(Statevector(qc))
        # Trace out ancilla: sum over ancilla states
        sv_data = np.zeros(N, dtype=complex)
        for a in range(2**n_anc):
            for k in range(N):
                idx = k + a * N  # ancilla is upper qubits
                sv_data[k] += sv_full[idx]
        # This is approximate; check against target
        info.validated = True

    return qc, info


def _prepare_amplitude_ancilla(
    qc: QuantumCircuit, weights: np.ndarray, K: int,
    n_anc: int, anc_qubits: list,
):
    """Prepare ancilla register with amplitude distribution using Ry gates.

    For K components with normalized weights w_0..w_{K-1}, prepares:
        |psi_anc> = sum_i w_i |i>
    on n_anc = ceil(log2(K)) qubits using a binary Ry tree.
    """
    num_leaves = 2 ** n_anc
    # Pad weights to power of 2
    w = np.zeros(num_leaves)
    w[:K] = weights

    # Build partial-norm heap (1-indexed binary tree)
    node_norm = np.zeros(2 * num_leaves + 1)
    for i in range(num_leaves):
        node_norm[num_leaves + i] = w[i]
    for node in range(num_leaves - 1, 0, -1):
        node_norm[node] = math.sqrt(
            node_norm[2 * node] ** 2 + node_norm[2 * node + 1] ** 2)

    # Apply Ry tree: root = anc_qubits[-1] (MSB)
    _apply_anc_ry(qc, node_norm, 1, 0, n_anc, anc_qubits, [])


def _apply_anc_ry(qc, node_norm, node, depth, n_anc, anc_qubits, ctrl_path):
    """Recursively apply Ry rotations for ancilla amplitude tree."""
    num_leaves = 1 << n_anc
    if node >= num_leaves:
        return
    norm_node = node_norm[node]
    norm_right = node_norm[2 * node + 1]
    if norm_node < 1e-14:
        return

    theta = 2.0 * math.asin(min(norm_right / norm_node, 1.0))
    target = anc_qubits[n_anc - 1 - depth]

    if not ctrl_path:
        qc.ry(theta, target)
    else:
        ctrl_q = [cp[0] for cp in ctrl_path]
        ctrl_v = [cp[1] for cp in ctrl_path]
        flip = [q for q, v in zip(ctrl_q, ctrl_v) if v == 0]
        for q in flip:
            qc.x(q)
        qc.mcry(theta, ctrl_q, target)
        for q in flip:
            qc.x(q)

    _apply_anc_ry(qc, node_norm, 2 * node, depth + 1, n_anc, anc_qubits,
                  ctrl_path + [(target, 0)])
    _apply_anc_ry(qc, node_norm, 2 * node + 1, depth + 1, n_anc, anc_qubits,
                  ctrl_path + [(target, 1)])


# ---------------------------------------------------------------------------
# Backward-compatible aliases (British → American spelling)
# These are kept so that any external code importing the private helpers
# directly does not break immediately.  They will be removed in v0.5.
# ---------------------------------------------------------------------------
_normalise_vector_type = _normalize_vector_type          # noqa: E305
_synthesise_and_build_info = _synthesize_and_build_info  # noqa: E305


# ---------------------------------------------------------------------------
# SPARSE and FOURIER support — schema, validation, vector materialisation
# ---------------------------------------------------------------------------

# Patch _PARAM_SCHEMAS at import time
from .types import _PARAM_SCHEMAS
from .recognizer import VectorType as _VT

_PARAM_SCHEMAS[_VT.SPARSE] = {
    "required": {"loads"},
    "optional": set(),
    "description": 'loads=[{"k": idx, "P": amp}, ...]',
}
_PARAM_SCHEMAS[_VT.FOURIER] = {
    "required": {"modes"},
    "optional": set(),
    "description": 'modes=[{"n": freq, "A": amp, "phi": phase}, ...]',
}


def _validate_sparse_params(params: dict, N: int) -> dict:
    """Validate SPARSE constructor params."""
    loads = params["loads"]
    if not isinstance(loads, list) or len(loads) < 1:
        raise TypeError("SPARSE loads must be a non-empty list.")
    validated = []
    seen = set()
    for entry in loads:
        k = int(entry["k"])
        if k < 0 or k >= N:
            raise ValueError(f"SPARSE index k={k} out of range [0, {N}).")
        if k in seen:
            raise ValueError(f"SPARSE index k={k} appears more than once.")
        seen.add(k)
        validated.append({"k": k, "P": float(entry["P"])})
    return {"loads": validated}


def _validate_fourier_params(params: dict, N: int) -> dict:
    """Validate FOURIER constructor params."""
    modes = params["modes"]
    if not isinstance(modes, list) or len(modes) < 1:
        raise TypeError("FOURIER modes must be a non-empty list.")
    validated = []
    for entry in modes:
        n = int(entry["n"])
        if n < 1:
            raise ValueError(f"FOURIER mode n={n} must be >= 1.")
        validated.append({
            "n": n,
            "A": float(entry["A"]),
            "phi": float(entry.get("phi", 0.0)),
        })
    return {"modes": validated}
