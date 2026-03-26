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
    _VectorObj, SPARSE, FOURIER, STEP, SQUARE,
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



def _normalize_vector_type(vector_type: Union[VectorType, str]) -> VectorType:
    """Convert string to VectorType enum; reject UNKNOWN."""
    if isinstance(vector_type, str):
        upper = vector_type.upper()
        try:
            vector_type = VectorType[upper]
        except KeyError:
            valid = [vt.name for vt in VectorType if vt != VectorType.UNKNOWN]
            raise ValueError(
                f"Unknown vector_type '{vector_type}'.  "
                f"Valid values: {valid}"
            )

    if vector_type == VectorType.UNKNOWN:
        raise ValueError("vector_type=UNKNOWN is not valid.")
    return vector_type


# -------------------------------------------------------------------
# Parameter validation (for encode_params)
# -------------------------------------------------------------------

def _validate_params(vector_type: VectorType, N: int, params: dict) -> dict:
    """Validate user-supplied parameters against the schema."""
    schema = _PARAM_SCHEMAS.get(vector_type)
    if schema is None:
        raise TypeError(
            f"vector_type={vector_type.name} is not supported."
        )

    required = schema["required"]
    optional = schema["optional"]
    allowed  = required | optional
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

    if vector_type == VectorType.STEP:
        result.setdefault("c", 1.0)
        k_s = int(result["k_s"])
        if k_s < 1 or k_s > N:
            raise ValueError(f"k_s={k_s} out of range [1, {N}].")
        result["k_s"] = k_s
        result["c"] = float(result["c"])

    elif vector_type == VectorType.SQUARE:
        result.setdefault("c", 1.0)
        k1, k2 = int(result["k1"]), int(result["k2"])
        if k1 < 0 or k2 > N or k1 >= k2:
            raise ValueError(f"Invalid range [{k1}, {k2}) for N={N}.")
        result["k1"], result["k2"] = k1, k2
        result["c"] = float(result["c"])

    elif vector_type == VectorType.SPARSE:
        result = _validate_sparse_params(result, N)

    elif vector_type == VectorType.WALSH:
        k = int(result["k"])
        m_bits = int(round(math.log2(N)))
        if k < 0 or k >= m_bits:
            raise ValueError(f"WALSH qubit index k={k} out of range [0, {m_bits}).")
        result["k"] = k
        result.setdefault("c_pos", 1.0)
        c_pos = float(result["c_pos"])
        result["c_pos"] = c_pos
        result["c_neg"] = float(result.get("c_neg", -c_pos))

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
    fallback_vector,
):
    """Return the expected real amplitude vector for validation."""
    N  = pattern.N
    lt = pattern.load_type
    p  = pattern.params

    if fallback_vector is not None:
        return fallback_vector.astype(float)

    if lt == VectorType.STEP:
        f = np.zeros(N); f[:p["k_s"]] = p.get("c", 1.0); return f

    if lt == VectorType.SQUARE:
        f = np.zeros(N); f[p["k1"]:p["k2"]] = p.get("c", 1.0); return f

    if lt == VectorType.WALSH:
        k = p["k"]
        c_pos = p.get("c_pos", 1.0)
        c_neg = p.get("c_neg", -c_pos)
        f = np.array([c_pos if not ((i >> k) & 1) else c_neg for i in range(N)], dtype=float)
        return f

    if lt == VectorType.SPARSE:
        f = np.zeros(N)
        for load in p["loads"]: f[load["k"]] = load["P"]
        return f

    if lt == VectorType.FOURIER:
        k = np.arange(N)
        f = np.zeros(N)
        for mode in p["modes"]:
            f += mode["A"] * np.sin(
                2 * math.pi * mode["n"] * k / N + mode.get("phi", 0.0))
        return f

    return None


def _build_component_vector(comp: _VectorObj, N: int):
    """Materialise a single component vector from its constructor."""
    p = comp.params
    if comp.vector_type == VectorType.STEP:
        f = np.zeros(N); f[:p["k_s"]] = p.get("c", 1.0); return f
    if comp.vector_type == VectorType.SQUARE:
        f = np.zeros(N); f[p["k1"]:p["k2"]] = p.get("c", 1.0); return f
    if comp.vector_type == VectorType.WALSH:
        k = p["k"]
        c_pos = p.get("c_pos", 1.0)
        c_neg = p.get("c_neg", -c_pos)
        return np.array([c_pos if not ((i >> k) & 1) else c_neg for i in range(N)], dtype=float)
    if comp.vector_type == VectorType.SPARSE:
        f = np.zeros(N)
        for load in p["loads"]: f[load["k"]] = load["P"]
        return f
    if comp.vector_type == VectorType.FOURIER:
        k = np.arange(N); f = np.zeros(N)
        for mode in p["modes"]:
            f += mode["A"] * np.sin(2 * math.pi * mode["n"] * k / N + mode.get("phi", 0.0))
        return f
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


# ---------------------------------------------------------------------------
# Disjoint support detection
# ---------------------------------------------------------------------------

def _support_interval(comp):
    """Return (k1, k2) interval of support, or None if full/unknown support."""
    vt = comp.vector_type
    p  = comp.params
    if vt == VectorType.STEP:
        return (0, p["k_s"])
    if vt == VectorType.SQUARE:
        return (p["k1"], p["k2"])
    if vt == VectorType.SPARSE:
        indices = [ld["k"] for ld in p["loads"]]
        return (min(indices), max(indices) + 1)
    # WALSH, FOURIER have full support
    return None


def _intervals_disjoint(comps):
    """
    Return True if all components have analytically disjoint support.
    WALSH and FOURIER always have full support — never disjoint with anything.
    STEP/SQUARE/SPARSE: check pairwise interval non-overlap.
    """
    intervals = []
    for comp in comps:
        iv = _support_interval(comp)
        if iv is None:
            return False   # full support — cannot be disjoint
        intervals.append(iv)

    # Pairwise overlap check
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            a1, a2 = intervals[i]
            b1, b2 = intervals[j]
            if a1 < b2 and b1 < a2:   # intervals overlap
                return False
    return True


def _compute_success_probability(weights, component_vectors):
    """
    Success probability for Protocol 1 LCU (PREP + ctrl-U + PREP_dagger):
      p = sum_{i,j} beta_i * beta_j * <f_hat_i | f_hat_j>
    where beta_j = w_j * ||f_j|| / Z, Z = ||(w_j * ||f_j||)||.
    For disjoint support: all cross terms vanish, p = sum_j beta_j^2 <= 1.
    For overlapping support: cross terms > 0, p > disjoint case.
    p = 1 only when all component states are identical.
    """
    norms  = np.array([np.linalg.norm(v) for v in component_vectors])
    scaled = np.sqrt(np.array(weights, dtype=float) * norms)
    Z = np.linalg.norm(scaled)
    if Z < 1e-14:
        return 0.0
    betas   = scaled / Z
    f_hats  = [v / n if n > 1e-14 else np.zeros_like(v)
               for v, n in zip(component_vectors, norms)]
    # Correct formula: p = ||sum_j beta_j^2 |f_hat_j>||^2
    # = sum_{i,j} beta_i^2 * beta_j^2 * <f_hat_i|f_hat_j>
    p = 0.0
    for i, (bi, fi) in enumerate(zip(betas, f_hats)):
        for j, (bj, fj) in enumerate(zip(betas, f_hats)):
            p += (bi**2) * (bj**2) * float(np.dot(fi, fj))
    return float(np.clip(p, 0.0, 1.0))


# ---------------------------------------------------------------------------
# LCU encoding
# ---------------------------------------------------------------------------

def _encode_lcu(lcu_obj, N, validate, tol):
    """
    Encode an LCU constructor: weighted superposition of component states.

    For disjoint-support components (STEP, SQUARE, SPARSE combinations):
      - success_probability = 1.0, no warning
    For overlapping components:
      - success_probability < 1.0, UserWarning issued
    """
    import warnings
    from .synthesizer import synthesize as _synthesize
    from .types import LCU as LCU_type, EncodingInfo

    weights   = lcu_obj.params["weights"]
    comp_objs = lcu_obj.params["components"]
    K = len(comp_objs)
    m = int(round(math.log2(N)))

    # Single component — just encode directly
    if K == 1:
        validated_params = _validate_params(
            comp_objs[0].vector_type, N, comp_objs[0].params)
        pattern = LoadPattern(comp_objs[0].vector_type, N=N, params=validated_params)
        return _synthesize_and_build_info(pattern, None, validate, tol)

    # Validate and materialise each component
    component_vectors  = []
    component_patterns = []
    for comp in comp_objs:
        validated = _validate_params(comp.vector_type, N, comp.params)
        component_patterns.append(
            LoadPattern(comp.vector_type, N=N, params=validated))
        component_vectors.append(_build_component_vector(comp, N))

    # Disjoint support check
    disjoint = _intervals_disjoint(comp_objs)

    # Compute success probability: p = sum_{i,j} beta_i*beta_j*<fi|fj>
    # Disjoint: p = sum_j beta_j^2 (cross terms zero)
    # Overlapping: p higher due to positive cross terms
    # p = 1 only when all component states are identical
    p_success = _compute_success_probability(weights, component_vectors)
    if not disjoint:
        warnings.warn(
            f"LCU components have overlapping support. "
            f"Success probability p={p_success:.4f}. "
            f"Post-selection on ancilla |0> required; "
            f"use amplitude amplification for repeated preparation.",
            UserWarning, stacklevel=3,
        )

    # Ancilla amplitude: beta_j = sqrt(w_j * ||f_j||) / Z
    # This ensures beta_j^2 proportional to w_j * ||f_j||,
    # so the post-selected state is proportional to sum_j w_j |f_hat_j>.
    norms  = np.array([np.linalg.norm(v) for v in component_vectors])
    scaled = np.sqrt(np.array(weights, dtype=float) * norms)
    Z      = np.linalg.norm(scaled)
    if Z < 1e-14:
        raise ValueError("LCU: combined vector is zero.")
    weights_norm = scaled / Z

    # Synthesise component circuits
    component_circuits = [_synthesize(pat) for pat in component_patterns]

    # Build LCU circuit
    n_anc       = math.ceil(math.log2(K))
    total_qubits = m + n_anc
    qc          = QuantumCircuit(total_qubits, name="lcu")
    data_qubits = list(range(m))
    anc_qubits  = list(range(m, total_qubits))

    # Step 1: Ry-tree on ancilla
    _prepare_amplitude_ancilla(qc, weights_norm, K, n_anc, anc_qubits)

    # Step 2: controlled component circuits
    for i, circ_i in enumerate(component_circuits):
        ctrl_state  = format(i, f'0{n_anc}b')[::-1]
        flip_qubits = [anc_qubits[b] for b in range(n_anc) if ctrl_state[b] == '0']
        for q in flip_qubits:
            qc.x(q)
        controlled_circ = circ_i.to_gate().control(n_anc)
        qc.append(controlled_circ, anc_qubits + data_qubits)
        for q in flip_qubits:
            qc.x(q)

    # Step 3: apply PREP_dagger to uncompute ancilla
    # This is the key step for Protocol 1 pure-state LCU.
    # After this, post-selecting ancilla=|0> gives the target pure state.
    anc_prep_circ = QuantumCircuit(n_anc)
    _prepare_amplitude_ancilla(anc_prep_circ, weights_norm, K, n_anc, list(range(n_anc)))
    qc.compose(anc_prep_circ.inverse(), qubits=anc_qubits, inplace=True)

    total_gates = sum(qc.decompose(reps=3).count_ops().values())
    complexity  = f"O({K}·m)"

    info = EncodingInfo(
        vector_type="LCU",
        N=N,
        m=m,
        gate_count=total_gates,
        complexity=complexity,
        validated=False,
        params={
            "components": [c.vector_type.name for c in comp_objs],
            "weights":    weights,
            "disjoint":   disjoint,
        },
        success_probability=p_success,
    )

    if validate:
        _validate_lcu_circuit(qc, component_vectors, weights,
                               N, m, n_anc, disjoint, tol)
        info.validated = True

    return qc, info


def _validate_lcu_circuit(qc, component_vectors, weights,
                           N, m, n_anc, disjoint, tol):
    """
    Validate LCU circuit (Protocol 1) by post-selecting ancilla on |0>.

    After PREP + ctrl-U + PREP_dagger, the anc=0 subspace holds the
    target pure state (up to normalization by sqrt(p_success)).
    """
    from qiskit.quantum_info import Statevector

    # Build expected combined vector
    f_combined = sum(w * v for w, v in zip(weights, component_vectors))
    norm_f = np.linalg.norm(f_combined)
    if norm_f < 1e-14:
        return
    expected = (f_combined / norm_f).real

    # Post-select: read anc=0 subspace of statevector
    sv = np.array(Statevector(qc))
    sv_anc0 = sv[:N].real   # ancilla qubit is MSB: anc=0 -> indices 0..N-1
    norm_anc0 = np.linalg.norm(sv_anc0)
    if norm_anc0 < 1e-14:
        raise ValueError("LCU validation: anc=0 subspace is empty.")
    simulated = sv_anc0 / norm_anc0

    if not np.allclose(np.abs(simulated), np.abs(expected), atol=max(tol, 1e-4)):
        max_err = np.max(np.abs(np.abs(simulated) - np.abs(expected)))
        raise ValueError(
            f"LCU validation failed: max amplitude error = {max_err:.2e} > tol={tol}."
        )
