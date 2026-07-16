"""
pyencode.types
==============
Typed constructor classes and EncodingInfo dataclass.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .recognizer import PatternKind

__version__ = "3.0.0"


# ---------------------------------------------------------------------------
# Internal: amplitude type coercion
# ---------------------------------------------------------------------------

def _coerce_amp(z):
    """
    Internal coercion for amplitude-like parameters (c, c0, c1, P, r, w, ...).

    The library accepts both real and complex amplitudes via the same
    constructor signature; this helper keeps the *storage type* matched
    to the *value*: a real-valued input round-trips as a Python float
    (preserving all backward-compatible introspection like ``float(P)``
    and ``params["c"] > 0``), and a genuinely complex input is stored
    as a Python complex.

    The dispatch rule is exactness of the imaginary part: complex(1+0j)
    coerces to float(1.0); complex(1+1e-300j) stays complex.
    """
    if isinstance(z, (int, float)):
        return float(z)
    cz = complex(z)
    if cz.imag == 0.0:
        return cz.real
    return cz


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _Pattern:
    """Base class for typed vector constructors."""
    kind: PatternKind
    params: dict

    def __repr__(self):
        p = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.kind.name}({p})"


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

class STEP(_Pattern):
    """STEP(k_e, c) — prefix uniform superposition [0, k_e). O(m) gates.

    Implements the Shukla-Vedula (2024) interval circuit.
    STEP(k_e=N) produces the full uniform superposition H^m|0>.

    The amplitude ``c`` may be real or complex. A non-real ``c`` adds a
    global phase ``arg(c)`` to the prepared state; this is unobservable
    in isolation but matters when STEP is used as a SUM component.
    """
    def __init__(self, k_e, c=1.0):
        self.kind = PatternKind.STEP
        self.params = {"k_e": int(k_e), "c": _coerce_amp(c)}


class SQUARE(_Pattern):
    """SQUARE(k_s, k_e, c) — interval uniform superposition [k_s, k_e). O(m^2) gates.

    Extends STEP to arbitrary intervals via a Draper QFT-based constant adder.
    O(m) for k_s=0 or power-of-2-aligned blocks.
    SQUARE(k_s=0, k_e=k_e) is identical to STEP(k_e).

    The amplitude ``c`` may be real or complex. A non-real ``c`` adds a
    global phase ``arg(c)`` to the prepared state.
    """
    def __init__(self, k_s, k_e, c=1.0):
        self.kind = PatternKind.SQUARE
        self.params = {"k_s": int(k_s), "k_e": int(k_e), "c": _coerce_amp(c)}


class WALSH(_Pattern):
    """WALSH(k, c0, c1) — generalized Walsh function. O(m) gates.

    Prepares a two-level piecewise-constant state with period P = 2^(k+1):
      amplitude proportional to c0 where bit k of i is 0,
      amplitude proportional to c1 where bit k of i is 1.

    Circuit: R_y(theta) on qubit k, then H on all m qubits. Total: m+1 gates.
    When c1 = -c0 (default), R_y(pi) = X and this reduces to the
    standard Walsh function (signed uniform superposition).

    The constants c0 and c1 may be real or complex. A complex pair gives
    the prepared state an observable relative phase ``arg(c1) - arg(c0)``
    between the two halves of the register.

    Parameters
    ----------
    k : int
        Qubit index (0 = LSB). Period P = 2^(k+1). Must satisfy 0 <= k < m.
    c0 : float or complex, optional
        Amplitude on the b_k=0 half. Default 1.0.
    c1 : float or complex, optional
        Amplitude on the b_k=1 half. Default -c0 (standard Walsh).

    Examples
    --------
    >>> circuit, info = encode(WALSH(k=1), N=8)                      # standard: +1/-1
    >>> circuit, info = encode(WALSH(k=1, c0=2.0), N=8)              # standard: +2/-2
    >>> circuit, info = encode(WALSH(k=2, c0=1.0, c1=4.0), N=8)      # generalized: two positive levels
    >>> circuit, info = encode(WALSH(k=0, c0=1.0, c1=1j), N=8)       # complex: relative phase pi/2
    """
    def __init__(self, k, c0=1.0, c1=None):
        self.kind = PatternKind.WALSH
        c0 = _coerce_amp(c0)
        c1 = _coerce_amp(c1) if c1 is not None else -c0
        self.params = {"k": int(k), "c0": c0, "c1": c1}

class SPARSE(_Pattern):
    """SPARSE([(x1, a1), (x2, a2), ...]) — s point masses at arbitrary indices.

    Implements the Gleinig-Hoefler algorithm. Gate complexity O(s * m).
    The s=1 case reduces to a single computational-basis state (X gates only).

    Amplitudes may be real or complex. The Gleinig-Hoefler tree uses an
    Ry merge for purely real amplitudes (gate count unchanged) and adds
    a per-merge controlled-phase gate for complex amplitudes.

    Example
    -------
    >>> circuit, info = encode(SPARSE([(19, 1.0)]), N=64)
    >>> circuit, info = encode(SPARSE([(1, 3.0), (6, 4.0)]), N=8)
    >>> circuit, info = encode(SPARSE([(1, 1+1j), (6, 1-1j)]), N=8)
    """
    def __init__(self, entries):
        self.kind = PatternKind.SPARSE
        loads = []
        for item in entries:
            try:
                x, a = item
            except (TypeError, ValueError):
                raise TypeError(
                    f"SPARSE expects a list of (index, amplitude) tuples, "
                    f"got {item!r}."
                )
            loads.append({"k": int(x), "P": _coerce_amp(a)})
        if len(loads) == 0:
            raise ValueError("SPARSE requires at least one entry.")
        self.params = {"loads": loads}

    def __repr__(self):
        # Override the default _Pattern.__repr__: the constructor takes
        # `entries=[(k, v), ...]` but the internal params dict uses key
        # 'loads' (list of {"k":..., "P":...} dicts).  Without this
        # override, ``eval(repr(obj))`` fails with an unknown-kwarg error,
        # which breaks every emitter that serialises components via {obj!r}.
        # Real-valued amplitudes render as floats so repr round-trips
        # through eval() without becoming ``complex(0.5, 0.0)``.
        entries = []
        for ld in self.params["loads"]:
            P = ld["P"]
            if isinstance(P, complex) and P.imag == 0.0:
                entries.append((ld["k"], P.real))
            else:
                entries.append((ld["k"], P))
        return f"SPARSE({entries!r})"


class FOURIER(_Pattern):
    """FOURIER(modes=[(n, A, phi), ...]) — T sinusoidal modes via inverse QFT.

    Gate complexity O(m^2) for any T.
    Single-mode T=1 subsumes SINE (phi=0) and COSINE (phi=pi/2).

    Example
    -------
    >>> circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=16)
    >>> circuit, info = encode(FOURIER(modes=[(1, 1.0, 0), (3, 0.5, 0)]), N=16)
    """
    def __init__(self, modes):
        self.kind = PatternKind.FOURIER
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
            # Reject non-integer frequencies. The DFT-pair construction in
            # _synth_fourier is exact only when n is an integer aligned with
            # the discrete grid; non-integer n makes the analytical signal
            # non-periodic on N samples, so its true DFT spreads across every
            # bin (spectral leakage) and the prepared state would silently
            # disagree with the formula A*sin(2*pi*n*i/N + phi).
            try:
                n_int = int(n)
            except (TypeError, ValueError):
                raise TypeError(
                    f"FOURIER mode n={n!r} must be an integer."
                )
            if n_int != n:
                raise ValueError(
                    f"FOURIER mode n={n!r} must be an integer; "
                    f"non-integer frequencies are not aligned with the "
                    f"discrete grid and would cause silent spectral "
                    f"leakage. Use encode_mps for arbitrary smooth signals."
                )
            mode_list.append({"n": n_int, "A": float(A), "phi": float(phi)})
        if len(mode_list) == 0:
            raise ValueError("FOURIER requires at least one mode.")
        self.params = {"modes": mode_list}


class GEOMETRIC(_Pattern):
    """GEOMETRIC(r, k_s=0, k_e=None, c=1.0) — geometric sequence on [k_s, k_e).

    Prepares a state proportional to:
      f_i = c * r^(i - k_s)   for i in [k_s, k_e),
      f_i = 0                 elsewhere.

    The default k_e=None means "use N" (i.e. the support is [k_s, N)),
    matching the historical behavior. An explicit k_e gives an upper bound
    on the support window, mirroring the SQUARE(k_s, k_e, c) API.

    With k_s=0 and k_e=N, the vector is multiplicatively separable over
    the bits of i:
      f_i = c * r^(sum_j b_j * 2^j) = c * prod_j r^(b_j * 2^j)
    so the quantum state is a product state prepared by m independent
    R_y rotations — one per qubit, zero entangling gates, depth 1.

    For a restricted window [k_s, k_e), there are two implementation tiers:

    Tier 1 (power-of-2-aligned): If the window width w = k_e - k_s is a
    power of 2 AND k_s is a multiple of w, the construction uses a
    geometric product state on the lower log2(w) qubits plus X gates on
    the upper qubits to fix the window position.
    Gate count: log2(w) + popcount(k_s/w), zero two-qubit gates, depth 1.

    Tier 2 (arbitrary window): For any other (k_s, k_e), uses dyadic
    decomposition of [k_s, k_e) and Gleinig-Hoefler anchor loading.
    Cost: O(L*m^2) basis gates with L <= m blocks; includes two-qubit gates.

    Parameters
    ----------
    r : float or complex
        Base (common ratio) of the geometric sequence. Must satisfy
        r != 0 and r != 1. Typical real values: 0 < r < 1 for decay,
        r > 1 for growth. Complex r yields oscillating geometrics; in
        particular, r = exp(i*omega) gives a discrete plane wave
        f_i = exp(i*omega*i), the elementary Fourier basis function.
    k_s : int, optional (default 0)
        Starting index (inclusive). Amplitudes below this index are zero.
        Must satisfy 0 <= k_s < N.
    k_e : int or None, optional (default None)
        Ending index (exclusive). Amplitudes at or above this index are
        zero. None means k_e = N (full window from k_s to end of register).
        Must satisfy k_s < k_e <= N.
    c : float or complex, optional
        Leading amplitude (default 1.0). Affects normalization, and (when
        complex) the global phase of the prepared state.

    Examples
    --------
    >>> circuit, info = encode(GEOMETRIC(r=0.95), N=64)               # full window
    >>> circuit, info = encode(GEOMETRIC(r=0.5, k_s=0, k_e=8), N=16)  # tier 1: lower half
    >>> circuit, info = encode(GEOMETRIC(r=0.9, k_s=32), N=64)        # tier 1: upper half
    >>> circuit, info = encode(GEOMETRIC(r=0.8, k_s=4, k_e=10), N=16) # tier 2
    >>> import math
    >>> circuit, info = encode(GEOMETRIC(r=complex(math.cos(0.5),    # plane wave
    ...                                            math.sin(0.5))), N=16)
    """
    def __init__(self, r, k_s=0, k_e=None, c=1.0):
        self.kind = PatternKind.GEOMETRIC
        r = _coerce_amp(r)
        if abs(r) < 1e-14:
            raise ValueError(f"GEOMETRIC r must be nonzero, got {r}.")
        if abs(r - 1.0) < 1e-14:
            raise ValueError(
                "GEOMETRIC r=1.0 is a uniform vector; use STEP or SQUARE instead."
            )
        k_s = int(k_s)
        if k_s < 0:
            raise ValueError(f"GEOMETRIC k_s must be non-negative, got {k_s}.")
        if k_e is not None:
            k_e = int(k_e)
            if k_e <= k_s:
                raise ValueError(
                    f"GEOMETRIC requires k_s < k_e; got k_s={k_s}, k_e={k_e}."
                )
        # Store k_e as None when unset; _validate_params resolves to N.
        self.params = {"r": r, "k_s": k_s, "k_e": k_e, "c": _coerce_amp(c)}


class HAMMING(_Pattern):
    """HAMMING(r, c) — amplitudes depend only on Hamming weight.  O(m) gates, depth 1.

    Prepares a product state proportional to f_i = c * r^{wt(i)}, where
    wt(i) is the Hamming weight of i --- the number of 1-bits in the
    binary representation of i (also known as the population count,
    `popcount`).

    Circuit: m identical R_y(theta) gates applied in parallel, with
      theta = 2 * arctan(r),
    yielding the single-qubit state (|0> + r|1>) / sqrt(1+r^2) on each wire.
    The tensor product gives amplitude r^{wt(i)} at basis state |i>.

    Properties:
      - Gate count: m single-qubit R_y gates
      - Two-qubit gate count: 0
      - Depth: 1 (all rotations are on disjoint qubits, execute in parallel)

    Physical interpretation.  If p = r^2 / (1 + r^2) is the per-qubit
    excitation probability, then |f_i|^2 follows a Bernoulli product
    distribution and the marginal distribution over wt(i) is
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
    >>> circuit, info = encode(HAMMING(r=0.5), N=64)
    >>> # info.complexity    -> "O(m)"
    >>> # info.gate_count    -> 6  (= m)
    >>> # info.gate_count_2q -> 0
    >>> # info.circuit_depth -> 1

    References
    ----------
    Nielsen & Chuang, *Quantum Computation and Quantum Information*,
    Cambridge University Press, 2010 (§10.5, Hamming weight).
    """
    def __init__(self, r, c=1.0):
        self.kind = PatternKind.HAMMING
        r = _coerce_amp(r)
        if abs(r) < 1e-14:
            raise ValueError(
                f"HAMMING r must be nonzero, got {r}. "
                f"r=0 is a point mass at index 0; use SPARSE([(0, 1.0)]) instead."
            )
        self.params = {"r": r, "c": _coerce_amp(c)}


class STAIRCASE(_Pattern):
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
        self.kind = PatternKind.STAIRCASE
        r = _coerce_amp(r)
        if abs(r) < 1e-14:
            raise ValueError(
                f"STAIRCASE r must be nonzero, got {r}. "
                f"Use SPARSE([(0, 1.0)]) for the degenerate r=0 case."
            )
        if abs(r - 1.0) < 1e-14:
            raise ValueError(
                "STAIRCASE r=1.0 gives equal amplitudes on unary indices; "
                "use SPARSE with uniform weights on {0,1,3,...,2^m-1} instead."
            )
        self.params = {"r": r, "c": _coerce_amp(c)}


class DICKE(_Pattern):
    """DICKE(k, c=1.0) — uniform superposition over all weight-k basis states.

    Prepares the Dicke state

        |D^m_k> = C(m,k)^(-1/2) * sum_{|S|=k} |e_S>,

    i.e. the equal superposition of every computational-basis state of
    Hamming weight k on m qubits.  The amplitude vector is

        f_i = c * 1[wt(i) == k],

    zero off the weight-k sphere and constant on it.  Unlike HAMMING
    (which is a product state with geometric decay across weight classes),
    DICKE is genuinely entangled and supported on a single weight class.

    Construction (Bärtschi-Eidenbenz, FCT 2019).  Initialise with X gates
    on the top k' qubits (where k' = min(k, m-k)) to get |0^(m-k') 1^k'>,
    then apply a cascade of split-cyclic-shift unitaries

        U_{m,k'} = (product_{l=m..k'+1} SCS^l_{k'}) . (product_{l=k'..2} SCS^l_{l-1}).

    Each SCS^l_j block is built from one 2-qubit gate (i) and (j-1) 3-qubit
    gates (ii), consisting of a CNOT, a (controlled-)R_y rotation with
    angle 2 arccos(sqrt(j'/l)), and a CNOT.  When k > m/2 we use the
    Dicke symmetry

        |D^m_k> = X^{otimes m} |D^m_{m-k}>

    and append a final X-layer on all m qubits.  This halves the cascade
    cost whenever k > m/2 and is absorbed by the transpiler with no extra
    two-qubit gates.  Ancilla-free, deterministic, unit success
    probability.

      Gate count     : O(k' * (m - k'))  CX after transpilation
      Depth          : O(m)
      Complexity     : O(k * (m - k)), i.e. O(m) for k in {1, m-1}
                       up to O(m^2) at k = m/2.

    Special cases k = 0 and k = m reduce to |0...0> and |1...1>,
    handled with zero or m X gates respectively.

    Physical motivation.  Dicke states arise as:
      - The Hartree-Fock reference state on m spin-orbitals with k
        electrons (up to orbital ordering) in quantum chemistry VQE.
      - The uniform feasible initialiser for Grover-mixer QAOA on
        k-hot constrained problems (max-k-cover, k-densest-subgraph,
        portfolio selection).
      - Occupation-number states of k indistinguishable bosons in m
        modes.
      - Volume-constrained initialisers in discrete topology
        optimisation (``k of m elements active'').

    Parameters
    ----------
    k : int
        Hamming weight of the target states.  Must satisfy 0 <= k <= m
        where m = log2(N).
    c : float, optional
        Leading amplitude (default 1.0).  Only affects normalisation.

    Examples
    --------
    >>> # |D^4_2>: uniform over the six weight-2 basis states {0011, 0101,
    >>> #                                   0110, 1001, 1010, 1100}
    >>> circuit, info = encode(DICKE(k=2), N=16)
    >>> # info.complexity  -> "O(k*(m-k))"

    References
    ----------
    Bärtschi, A. & Eidenbenz, S., "Deterministic Preparation of Dicke
    States", *Fundamentals of Computation Theory* (FCT 2019),
    LNCS 11651, 126–139.  doi:10.1007/978-3-030-25027-0_9.

    Bärtschi, A. & Eidenbenz, S., "Short-Depth Circuits for Dicke State
    Preparation", *IEEE QCE 2022*.  doi:10.1109/QCE53715.2022.00027.
    """
    def __init__(self, k, c=1.0):
        self.kind = PatternKind.DICKE
        k = int(k)
        if k < 0:
            raise ValueError(f"DICKE k must be non-negative, got {k}.")
        self.params = {"k": k, "c": _coerce_amp(c)}


class POLYNOMIAL(_Pattern):
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
        self.kind = PatternKind.POLYNOMIAL
        coeffs = [_coerce_amp(c) for c in coeffs]
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


class TENSOR(_Pattern):
    """TENSOR([(pattern_1, N_1), (pattern_2, N_2), ...]) — disjoint-subregister composition.

    Prepares a tensor-product state where each component acts on its own
    subregister.  The total amplitude vector is the outer product of the
    component vectors, so the total qubit count is the sum of the
    subregister qubit counts and the total N is the product.

    Prepared state:
        |psi> = |f^(1)> otimes |f^(2)> otimes ... otimes |f^(K)>

    Unlike SUM (which SUPERPOSES components via an ancilla), TENSOR places
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
    components : list of (pattern, N_i) tuples
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
    ...     TENSOR([(GEOMETRIC(r=0.9), 16),
    ...             (GEOMETRIC(r=0.9), 16),
    ...             (GEOMETRIC(r=0.9), 16)]),
    ...     N=16 * 16 * 16)
    """
    def __init__(self, components):
        self.kind = PatternKind.TENSOR
        if not components:
            raise ValueError("TENSOR requires at least one component.")
        comp_list = []
        sizes = []
        for item in components:
            try:
                obj, n_i = item
            except (TypeError, ValueError):
                raise TypeError(
                    f"TENSOR expects (pattern, N_i) tuples, got {item!r}."
                )
            if not isinstance(obj, _Pattern):
                raise TypeError(
                    f"TENSOR component must be a pattern, got {type(obj).__name__}."
                )
            n_i = int(n_i)
            if n_i < 2 or (n_i & (n_i - 1)) != 0:
                raise ValueError(
                    f"TENSOR subregister size must be a power of 2 >= 2, got {n_i}."
                )
            comp_list.append(obj)
            sizes.append(n_i)
        self.params = {"components": comp_list, "sizes": sizes}


class SUM(_Pattern):
    """SUM([(w1, pattern1), (w2, pattern2), ...]) — weighted superposition.

    Prepares a weighted superposition of structured component states:
      |psi> ∝ sum_j w_j |f^(j)>

    For components with pairwise-disjoint support, prefer ``PARTITION``
    instead — it prepares the same state with no ancilla and success
    probability 1.  ``SUM`` is the general-purpose alternative that
    handles overlapping or weighted combinations via an ancilla register.

    Implementation: the Linear Combination of Unitaries (LCU) technique
    of Childs & Wiebe 2012, using ceil(log2(r)) ancilla qubits and
    controlled component circuits.

    Success probability.  ``SUM`` always post-selects the ancilla on
    |0>, so p < 1 in general -- including for disjoint components.
    Disjointness removes the cross terms, giving the closed form

        p = sum_j beta_j^4,    beta_j^2 = |w_j| ||f^(j)|| / Z,
        Z = sum_j |w_j| ||f^(j)||,

    which equals 1 only when a single component carries all the weight.
    Equal weights over r components give p = 1/r; unequal weights give
    something in between (e.g. weights 1 and 4 over two disjoint
    normalized components give beta^2 = 1/5, 4/5 and p = 17/25 = 0.68).
    Overlapping components additionally pick up cross terms, so p then
    depends on the relative geometry of the component states; a warning
    is issued in that case since amplitude amplification may be
    warranted.

    If you need p = 1 for disjoint components, use ``PARTITION``, which
    prepares the same state ancilla-free and deterministically.

    Parameters
    ----------
    components : list of (weight, pattern) tuples
        Unnormalized weights and typed constructors.
        Weights may be real or complex; only ``w == 0`` is rejected.
        For complex weights ``w_j = |w_j| e^{i phi_j}``, the LCU
        construction loads the magnitude distribution into the ancilla
        register's amplitudes (real positive) and threads the phase
        ``phi_j`` through SELECT as a controlled global phase.

    Examples
    --------
    >>> # Piecewise-constant: two disjoint intervals.
    >>> # Disjoint support removes the cross terms but does NOT give p = 1:
    >>> # beta^2 = 1/5, 4/5  ->  p = (1/5)^2 + (4/5)^2 = 17/25 = 0.68.
    >>> circuit, info = encode(
    ...     SUM([(1.0, SQUARE(k_s=0,  k_e=8,  c=1.0)),
    ...          (4.0, SQUARE(k_s=8,  k_e=16, c=1.0))]), N=16)
    >>> # info.success_probability -> 0.68
    >>> # PARTITION prepares this same state ancilla-free with p = 1.

    >>> # Mixed patterns: overlapping (p < 1, warning issued)
    >>> circuit, info = encode(
    ...     SUM([(1.0, STEP(k_e=8, c=1.0)),
    ...          (1.0, FOURIER(modes=[(1, 1.0, 0)]))]), N=16)
    >>> # UserWarning: overlapping support, p < 1.0

    >>> # Complex weights: superposition with relative phase
    >>> circuit, info = encode(
    ...     SUM([(1.0,  SQUARE(k_s=0, k_e=8, c=1.0)),
    ...          (1.0j, SQUARE(k_s=8, k_e=16, c=1.0))]), N=16)

    References
    ----------
    Childs & Wiebe, *Quantum Inf. Comput.* 12(11-12), 2012.
    Berry, Childs, Cleve, Kothari & Somma, *Phys. Rev. Lett.* 114(9), 2015.
    """
    def __init__(self, components):
        self.kind = PatternKind.SUM
        if not components:
            raise ValueError("SUM requires at least one component.")
        weights = []
        objs = []
        for item in components:
            try:
                w, obj = item
            except (TypeError, ValueError):
                raise TypeError(
                    f"SUM expects (weight, pattern) tuples, got {item!r}."
                )
            if not isinstance(obj, _Pattern):
                raise TypeError(
                    f"SUM component must be a pattern, got {type(obj).__name__}."
                )
            w = _coerce_amp(w)
            if abs(w) < 1e-14:
                raise ValueError(
                    f"SUM weights must be nonzero, got {w}."
                )
            weights.append(w)
            objs.append(obj)
        self.params = {"weights": weights, "components": objs}


def LCU(components):
    """Deprecated alias for ``SUM``.

    ``LCU`` has been renamed to ``SUM`` to describe the mathematical
    object (a weighted sum of structured states) rather than the
    implementation technique.  The underlying algorithm remains the
    Linear Combination of Unitaries method of Childs & Wiebe 2012;
    see ``SUM`` for full details.

    This alias is kept for backward compatibility and will be removed
    in a future release.  New code should use ``SUM``.
    """
    import warnings
    warnings.warn(
        "LCU is deprecated and will be removed in a future release; "
        "use SUM instead.  (The implementation is unchanged -- SUM is "
        "the new name for the same constructor.)",
        DeprecationWarning,
        stacklevel=2,
    )
    return SUM(components)


class PARTITION(_Pattern):
    """PARTITION([comp1, comp2, ...]) — disjoint-support composition.

    Prepares a state whose support is the *union* of component supports,
    provided those supports are pairwise disjoint.  The composition is
    deterministic: no ancilla, no post-selection, success probability 1.

    This is the ancilla-free counterpart of SUM.  When components happen
    to have disjoint support, PARTITION is strictly cheaper than SUM and
    avoids the need for amplitude amplification.

    Algorithm
    ---------
    Each component is decomposed into atoms of one of two kinds:
      - singleton atom  (a_k, value_k)         -- a SPARSE point mass,
      - dyadic block    (a_k, j_k, c_at_a, r)  -- a power-of-2-aligned
                                                  block of width 2^j_k,
                                                  with internal amplitude
                                                  c_at_a * r^(i - a_k)
                                                  for i in [a_k, a_k + 2^j_k).

    Component -> atoms:
      SPARSE([(x_i, v_i)])           : one singleton per (x_i, v_i).
      STEP(k_e, c), SQUARE(k_s, k_e, c): dyadic decomposition of [0, k_e)
                                       (resp. [k_s, k_e)), ratio = 1.
      GEOMETRIC(r, k_s, c)     : dyadic decomposition of [k_s, N),
                                       ratio inherited.

    Circuit:
      1. Gleinig-Hoefler sparse state preparation on the L anchor points
         {|a_k>} with weights w_k = sign_k * |c_at_a_k| * sqrt(block_norm)
         where block_norm = (r^(2*2^j_k) - 1) / (r^2 - 1) for a dyadic
         block (or 1 for a singleton).
      2. For each block with j_k >= 1, a multi-controlled R_y per free
         bit, controlled on upper qubits matching (a_k >> j_k), spreads
         the anchor amplitude across the block according to its internal
         ratio r.

    Cost: O(L * m) total gates, where L is the total number of atoms
    summed across components.  No ancilla, success_probability == 1.

    Disallowed components: FOURIER, WALSH, HAMMING, STAIRCASE, POLYNOMIAL
    have full or dense support and cannot participate in a disjoint
    partition.  Use SUM instead if such a component is required.

    Parameters
    ----------
    components : list of _Pattern
        Bounded-support constructors (SPARSE, STEP, SQUARE, GEOMETRIC).
        Supports must be pairwise disjoint under the chosen N; the
        framework verifies this and raises ValueError on overlap.

    Examples
    --------
    >>> # Sparse prefix + geometric tail on 256 indices
    >>> circuit, info = encode(
    ...     PARTITION([
    ...         SPARSE([(2, 0.3), (5, 0.5), (7, 0.7)]),
    ...         GEOMETRIC(r=0.8, k_s=11),
    ...     ]),
    ...     N=256)
    >>> info.success_probability   # 1.0, no post-selection

    >>> # Multi-interval piecewise-constant
    >>> circuit, info = encode(
    ...     PARTITION([SQUARE(k_s=0,  k_e=4,  c=1.0),
    ...                SQUARE(k_s=8,  k_e=12, c=2.0),
    ...                SQUARE(k_s=14, k_e=16, c=3.0)]),
    ...     N=16)

    References
    ----------
    Gleinig & Hoefler, DAC 2021          (sparse state preparation step).
    Bentley & Saxe, J. Algorithms, 1980  (dyadic interval decomposition).
    """
    def __init__(self, components):
        self.kind = PatternKind.PARTITION
        if not components:
            raise ValueError("PARTITION requires at least one component.")
        comp_list = []
        for item in components:
            if not isinstance(item, _Pattern):
                raise TypeError(
                    f"PARTITION component must be a pattern "
                    f"(SPARSE, STEP, SQUARE, or GEOMETRIC), got "
                    f"{type(item).__name__}."
                )
            if item.kind not in (PatternKind.SPARSE, PatternKind.STEP,
                                        PatternKind.SQUARE,
                                        PatternKind.GEOMETRIC):
                raise TypeError(
                    f"PARTITION component type {item.kind.name} has "
                    f"full or dense support and cannot be part of a disjoint "
                    f"partition.  Allowed types: SPARSE, STEP, SQUARE, "
                    f"GEOMETRIC.  Use SUM instead for overlapping or "
                    f"dense-support combinations."
                )
            comp_list.append(item)
        self.params = {"components": comp_list}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class EncodingInfo:
    """
    Metadata returned by encode().

    Attributes
    ----------
    pattern_name : str
        Name of the recognized pattern (e.g. "SPARSE", "STEP", "WALSH").
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
        p in (0,1] for SUM).
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
    pattern_name: str
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
            f"  Pattern     : {self.pattern_name}",
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
    PatternKind.STEP:    "O(m)",
    PatternKind.SQUARE:  "O(m²)",   # general; O(m) for k_s=0 or aligned blocks
    PatternKind.WALSH:   "O(m)",
    PatternKind.SPARSE:  "O(s\u00b7m)",
    PatternKind.FOURIER: "O(m\u00b2)",
    PatternKind.GEOMETRIC: "O(m)",
    PatternKind.HAMMING: "O(m)",
    PatternKind.STAIRCASE: "O(m)",
    PatternKind.DICKE:     "O(k*(m-k))",
    PatternKind.POLYNOMIAL: "O(m^(d+1))",
    PatternKind.PARTITION: "O(L\u00b7m)",
    PatternKind.UNKNOWN: "O(2^m)",
}


# ---------------------------------------------------------------------------
# Parameter schemas (for encode() validation)
# ---------------------------------------------------------------------------

_PARAM_SCHEMAS = {
    PatternKind.STEP: {
        "required": {"k_e"},
        "optional": {"c"},
        "description": "k_e=<prefix end index>",
    },
    PatternKind.SQUARE: {
        "required": {"k_s", "k_e"},
        "optional": {"c"},
        "description": "k_s=<k_s>, k_e=<end>",
    },
    PatternKind.WALSH: {
        "required": {"k"},
        "optional": {"c0", "c1"},
        "description": "k=<qubit index>, c0=<amplitude on b_k=0 half>, c1=<amplitude on b_k=1 half>",
    },
    PatternKind.SPARSE: {
        "required": {"loads"},
        "optional": set(),
        "description": 'loads=[{"k": idx, "P": amp}, ...]',
    },
    PatternKind.FOURIER: {
        "required": {"modes"},
        "optional": set(),
        "description": 'modes=[{"n": freq, "A": amp, "phi": phase}, ...]',
    },
    PatternKind.GEOMETRIC: {
        "required": {"r"},
        "optional": {"c", "k_s", "k_e"},
        "description": "r=<base (common ratio) of geometric sequence>, k_s=<starting index (default 0)>, k_e=<ending index, default N>",
    },
    PatternKind.HAMMING: {
        "required": {"r"},
        "optional": {"c"},
        "description": "r=<per-qubit amplitude ratio>",
    },
    PatternKind.STAIRCASE: {
        "required": {"r"},
        "optional": {"c"},
        "description": "r=<geometric ratio between consecutive unary amplitudes>",
    },
    PatternKind.DICKE: {
        "required": {"k"},
        "optional": {"c"},
        "description": "k=<Hamming weight of target states, 0 <= k <= m>",
    },
    PatternKind.POLYNOMIAL: {
        "required": {"coeffs"},
        "optional": {"normalize_domain"},
        "description": "coeffs=[c_0, c_1, ..., c_d]",
    },
}