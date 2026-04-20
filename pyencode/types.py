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
    """SQUARE(k1, k2, c) — interval uniform superposition [k1, k2). O(m^2) gates.

    Extends STEP to arbitrary intervals via a Draper QFT-based constant adder.
    O(m) for k1=0 or power-of-2-aligned blocks.
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
    """GEOMETRIC(ratio, start=0, c=1.0) — geometric sequence, optionally offset.

    Prepares a state proportional to:
      f_i = c * ratio^(i - start)   for i in [start, N),
      f_i = 0                       for i in [0, start).

    With start=0 (default), the vector is multiplicatively separable over
    the bits of i:
      f_i = c * ratio^(sum_j b_j * 2^j) = c * prod_j ratio^(b_j * 2^j)
    so the quantum state is a product state prepared by m independent
    R_y rotations — one per qubit, zero entangling gates, depth 1.

    With start > 0, there are two implementation tiers:

    Tier 1 (power-of-2-aligned): If the window width w = N - start is a 
    power of 2 AND start is a multiple of w, uses an efficient construction
    with geometric product state on lower qubits + X gates on upper qubits.
    Gate count: log2(w) + popcount(start/w), zero two-qubit gates, depth 1.

    Tier 2 (arbitrary offset): For any other start value, uses sparse
    encoding techniques to synthesize the geometric amplitudes directly.
    Gate count: O(w*m) where w = N - start, includes two-qubit gates.

    Parameters
    ----------
    ratio : float
        Base of the geometric sequence. Must satisfy 0 < ratio and ratio != 1.
        Typical values: 0 < ratio < 1 for decay, ratio > 1 for growth.
    start : int, optional (default 0)
        Starting index of the geometric sequence. Amplitudes below this
        index are zero. Any value 0 <= start < N is supported.
        Tier 1 examples at N=64: start=0, 32, 48, 56, 60, 62, 63.
        Tier 2 examples at N=64: start=4, 10, 17, etc.
    c : float, optional
        Leading amplitude (default 1.0). Only affects normalization.

    Examples
    --------
    >>> circuit, info = encode(GEOMETRIC(ratio=0.95), N=64)
    >>> circuit, info = encode(GEOMETRIC(ratio=0.5), N=16)
    >>> circuit, info = encode(GEOMETRIC(ratio=0.9, start=32), N=64)  # tier 1
    >>> circuit, info = encode(GEOMETRIC(ratio=0.8, start=4), N=256)  # tier 2
    """
    def __init__(self, ratio, start=0, c=1.0):
        self.vector_type = VectorType.GEOMETRIC
        ratio = float(ratio)
        if ratio <= 0:
            raise ValueError(f"GEOMETRIC ratio must be positive, got {ratio}.")
        if abs(ratio - 1.0) < 1e-14:
            raise ValueError(
                "GEOMETRIC ratio=1.0 is a uniform vector; use STEP(k_s=N) instead."
            )
        start = int(start)
        if start < 0:
            raise ValueError(f"GEOMETRIC start must be non-negative, got {start}.")
        self.params = {"ratio": ratio, "start": start, "c": float(c)}


class POPCOUNT(_VectorObj):
    """POPCOUNT(r, c) — amplitudes depend only on Hamming weight.  O(m) gates, depth 1.

    Prepares a product state proportional to f_i = c * r^popcount(i), where
    popcount(i) is the number of 1-bits in the binary representation of i.

    Circuit: m identical R_y(theta) gates applied in parallel, with
      theta = 2 * arctan(r),
    yielding the single-qubit state (|0> + r|1>) / sqrt(1+r^2) on each wire.
    The tensor product gives amplitude r^popcount(i) at basis state |i>.

    Properties:
      - Gate count: m single-qubit R_y gates
      - Two-qubit gate count: 0
      - Depth: 1 (all rotations are on disjoint qubits, execute in parallel)

    Physical interpretation.  If p = r^2 / (1 + r^2) is the per-qubit
    excitation probability, then |f_i|^2 follows a Bernoulli product
    distribution and the marginal distribution over popcount(i) is
    Binomial(m, p).  This structure arises in:
      - Ising models with uniform transverse field
      - Depolarising error channels (uniform per-qubit flip rate)
      - Hamming-weight-structured Hamiltonian coefficient vectors

    Parameters
    ----------
    r : float
        Ratio of |1>-amplitude to |0>-amplitude on each qubit.
        Must satisfy r > 0.  r = 1 gives the uniform superposition.
    c : float, optional
        Leading amplitude (default 1.0).  Only affects normalisation.

    Examples
    --------
    >>> # Binomial-structured state with per-qubit weight ratio 0.5
    >>> circuit, info = encode(POPCOUNT(r=0.5), N=64)
    >>> # info.complexity    -> "O(m)"
    >>> # info.gate_count    -> 6  (= m)
    >>> # info.gate_count_2q -> 0
    >>> # info.circuit_depth -> 1
    """
    def __init__(self, r, c=1.0):
        self.vector_type = VectorType.POPCOUNT
        r = float(r)
        if r <= 0:
            raise ValueError(
                f"POPCOUNT r must be positive, got {r}. "
                f"r=0 is a point mass at index 0; use SPARSE([(0, 1.0)]) instead."
            )
        self.params = {"r": r, "c": float(c)}


class STAIRCASE(_VectorObj):
    """STAIRCASE(r, c) — sparse geometric staircase on unary indices.  O(m) gates.

    Prepares a state with exactly m+1 nonzero amplitudes, supported on the
    "unary" indices  U = {2^k - 1 : k = 0, 1, ..., m} = {0, 1, 3, 7, ..., 2^m - 1}.
    The amplitudes form a clean geometric progression:

        f_{2^k - 1} = c * r^k   for k = 0, 1, ..., m,

    so adjacent nonzero amplitudes have ratio exactly r.

    Circuit: one R_y on qubit 0 followed by (m-1) cascaded CR_y gates in
    a linear nearest-neighbour chain (q0 -> q1 -> q2 -> ... -> q_{m-1}),
    each with a per-step angle chosen so the geometric-r property holds
    exactly.  Total: m single- and two-qubit rotations.

      Gate count     : m   (1 R_y + (m-1) CR_y)
      Entangling     : (m-1) CR_y   -> O(m) CX after decomposition
      Depth          : O(m)

    Physical motivation.  Hierarchical matrix (H-matrix) representations
    assign a coefficient to each dyadic scale; loading m+1 per-scale
    coefficients in geometric progression is a common PREP oracle for
    Pauli expansions of tridiagonal Toeplitz operators (discretised
    Laplacian, heat equation, wavelet-Galerkin stiffness matrices).

    Parameters
    ----------
    r : float
        Ratio between consecutive nonzero amplitudes.  Must satisfy r > 0
        and r != 1 (r=1 is a SPARSE uniform-weight state on unary indices).
    c : float, optional
        Leading amplitude (default 1.0).  Only affects normalisation.

    Examples
    --------
    >>> # Geometric staircase with ratio 0.5 on N=8 (k = 0..3)
    >>> circuit, info = encode(STAIRCASE(r=0.5), N=8)
    >>> # Nonzero indices: 0, 1, 3, 7   with amplitudes proportional to 1, 0.5, 0.25, 0.125

    References
    ----------
    Hackbusch, "A sparse matrix arithmetic based on H-matrices",
    Computing 62, 1999.
    """
    def __init__(self, r, c=1.0):
        self.vector_type = VectorType.STAIRCASE
        r = float(r)
        if r <= 0:
            raise ValueError(
                f"STAIRCASE r must be positive, got {r}. "
                f"Use SPARSE([(0, 1.0)]) for the degenerate r=0 case."
            )
        if abs(r - 1.0) < 1e-14:
            raise ValueError(
                "STAIRCASE r=1.0 gives equal amplitudes on unary indices; "
                "use SPARSE with uniform weights on {0,1,3,...,2^m-1} instead."
            )
        self.params = {"r": r, "c": float(c)}


class POLYNOMIAL(_VectorObj):
    """POLYNOMIAL(coeffs, normalize_domain=True) — degree-d polynomial amplitudes.

    Prepares the state proportional to

        f(i) = sum_{j=0}^{d} coeffs[j] * x(i)^j

    where x(i) = i/(N-1) if ``normalize_domain=True`` (evaluation on the
    unit interval) and x(i) = i otherwise.  Covers RAMP (d=1), QUADRATIC
    (d=2), POISEUILLE (d=2 parabolic), CUBIC (d=3), etc.

    Construction (Walsh-sparse loading).  A polynomial of degree d in the
    grid variable has Walsh-Hadamard spectrum supported only on indices of
    Hamming weight at most d (Welch et al., 2014; Gonzalez-Conde et al.,
    2024, remark after Lemma 1; Mac Williams & Sloane, Chapter 13).
    The number of nonzero Walsh coefficients is

        s = sum_{k=0}^{d} C(m, k)  =  O(m^d).

    The synthesiser computes these s coefficients classically, loads them
    as a sparse state using the Gleinig-Hoefler algorithm (O(s * m) gates),
    and applies a single layer of Hadamards H^{otimes m} to transform the
    Walsh-coefficient register into the target polynomial state.

      Total gate count : O(m * s) = O(m^{d+1})
      Exact (no approximation).

    Parameters
    ----------
    coeffs : list or numpy array of floats
        Polynomial coefficients [c_0, c_1, ..., c_d], so that
        f(x) = c_0 + c_1 x + c_2 x^2 + ... + c_d x^d.
    normalize_domain : bool, optional
        If True (default), evaluate at x = i/(N-1) (unit-interval grid).
        If False, evaluate at x = i (raw indices).

    Examples
    --------
    >>> # Ramp (d=1): f(i) = i/(N-1)
    >>> circuit, info = encode(POLYNOMIAL(coeffs=[0.0, 1.0]), N=16)

    >>> # Poiseuille parabolic profile (d=2): f(i) = 1 - (2 i/(N-1) - 1)^2
    >>> #   Expanding: f = 4 x - 4 x^2  after centring.
    >>> circuit, info = encode(POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]), N=32)

    >>> # Cubic: f(x) = x (1 - x)(1 - 2x) on x in [0, 1]
    >>> circuit, info = encode(POLYNOMIAL(coeffs=[0.0, 1.0, -3.0, 2.0]), N=64)

    References
    ----------
    Welch, Greenbaum, Mostame, Aspuru-Guzik, New J. Phys. 16, 033040 (2014).
    Gonzalez-Conde, Watts, Rodriguez-Grasa, Sanz, Quantum 8, 1297 (2024).
    """
    def __init__(self, coeffs, normalize_domain=True):
        self.vector_type = VectorType.POLYNOMIAL
        coeffs = [float(c) for c in coeffs]
        if len(coeffs) == 0:
            raise ValueError("POLYNOMIAL requires at least one coefficient.")
        # Strip trailing zeros so degree is correct
        while len(coeffs) > 1 and abs(coeffs[-1]) < 1e-14:
            coeffs = coeffs[:-1]
        # Reject the zero polynomial
        if all(abs(c) < 1e-14 for c in coeffs):
            raise ValueError("POLYNOMIAL zero polynomial has no encoding.")
        self.params = {
            "coeffs": coeffs,
            "normalize_domain": bool(normalize_domain),
        }


class TENSOR(_VectorObj):
    """TENSOR([(VectorObj_1, N_1), (VectorObj_2, N_2), ...]) — disjoint-subregister composition.

    Prepares a tensor-product state where each component acts on its own
    subregister.  The total amplitude vector is the outer product of the
    component vectors, so the total qubit count is the sum of the
    subregister qubit counts and the total N is the product.

    Prepared state:
        |psi> = |f^(1)> otimes |f^(2)> otimes ... otimes |f^(K)>

    Unlike LCU (which SUPERPOSES components via an ancilla), TENSOR places
    components on DISJOINT qubit registers.  No ancilla is needed, the
    success probability is 1, and there is no post-selection.

    Gate count: sum of component gate counts (no additional gates).
    Depth: max of component depths (components on disjoint qubits execute
    in parallel).
    Two-qubit gates: sum of component two-qubit gate counts.

    Physical motivation.  Separable 2D/3D source terms (e.g., discretised
    Poisson RHS of the form sin(2 pi n x) sin(2 pi p y)) split exactly
    into a tensor product of 1D states.  TENSOR formalises the
    ``circ_x.tensor(circ_y)`` idiom as a named pattern.

    Parameters
    ----------
    components : list of (VectorObj, N_i) tuples
        Each tuple declares a subregister: the constructor and its vector
        length.  Each N_i must be a power of 2.  The total encode() N
        must equal the product of all N_i.

    Examples
    --------
    >>> # Separable 2D Poisson source: sin(2 pi x) * sin(2 pi y)
    >>> circuit, info = encode(
    ...     TENSOR([(FOURIER(modes=[(1, 1.0, 0)]), 32),
    ...             (FOURIER(modes=[(3, 1.0, 0)]), 32)]),
    ...     N=32 * 32)
    >>> # info.gate_count -> sum of component gate counts

    >>> # 3D source: Gaussian-like profile factorisable across axes
    >>> circuit, info = encode(
    ...     TENSOR([(GEOMETRIC(ratio=0.9), 16),
    ...             (GEOMETRIC(ratio=0.9), 16),
    ...             (GEOMETRIC(ratio=0.9), 16)]),
    ...     N=16 * 16 * 16)
    """
    def __init__(self, components):
        self.vector_type = VectorType.TENSOR
        if not components:
            raise ValueError("TENSOR requires at least one component.")
        comp_list = []
        sizes = []
        for item in components:
            try:
                obj, n_i = item
            except (TypeError, ValueError):
                raise TypeError(
                    f"TENSOR expects (VectorObj, N_i) tuples, got {item!r}."
                )
            if not isinstance(obj, _VectorObj):
                raise TypeError(
                    f"TENSOR component must be a VectorObj, got {type(obj).__name__}."
                )
            n_i = int(n_i)
            if n_i < 2 or (n_i & (n_i - 1)) != 0:
                raise ValueError(
                    f"TENSOR subregister size must be a power of 2 >= 2, got {n_i}."
                )
            comp_list.append(obj)
            sizes.append(n_i)
        self.params = {"components": comp_list, "sizes": sizes}


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
        May differ from the raw circuit depth visible via print(circuit).
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
        else:
            lines.append(f"  Success prob: 1.0")
        if self.vector is not None:
            lines.append(f"  Vector      : numpy array, shape ({self.N},)")
        if self.params:
            lines.append(f"  Parameters  : {self.params}")
        if self.circuit_code:
            lines.append(f"  Circuit code: {len(self.circuit_code)} chars (info.circuit_code)")
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
    VectorType.POPCOUNT: "O(m)",
    VectorType.STAIRCASE: "O(m)",
    VectorType.POLYNOMIAL: "O(m^(d+1))",
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
        "optional": {"c", "start"},
        "description": "ratio=<base of geometric sequence>, start=<starting index (default 0)>",
    },
    VectorType.POPCOUNT: {
        "required": {"r"},
        "optional": {"c"},
        "description": "r=<per-qubit amplitude ratio>",
    },
    VectorType.STAIRCASE: {
        "required": {"r"},
        "optional": {"c"},
        "description": "r=<geometric ratio between consecutive unary amplitudes>",
    },
    VectorType.POLYNOMIAL: {
        "required": {"coeffs"},
        "optional": {"normalize_domain"},
        "description": "coeffs=[c_0, c_1, ..., c_d]",
    },
}