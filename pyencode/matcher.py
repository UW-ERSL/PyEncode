"""
pyencode.matcher
================
Reverse lookup: given a materialized amplitude vector, find which exact
pattern family best represents it.

``encode`` goes forward (typed declaration -> circuit).  ``match_vector``
goes backward: it takes a numerical vector and, for each exact pattern
family, fits that family's free parameters to the vector, scores the fit,
and returns the closest matches ranked by error.  Each returned match is a
ready-to-``encode`` constructor, so the typical workflow is::

    from pyencode import match_vector, encode

    matches = match_vector(v)          # v is a length-N numpy array
    print_matches(matches)             # human-readable ranking
    circuit, info = encode(matches[0].pattern, N=len(v))

Scoring
-------
Quantum state preparation is insensitive to a global scale and phase (the
state is normalized, and the leading amplitude ``c`` of every pattern is a
free parameter).  So the fit metric is scale- and phase-invariant: for a
candidate vector ``w`` the optimal complex scale ``alpha`` minimizing
``||v - alpha w||`` is projected out, and the reported ``rel_error`` is

    rel_error = ||v - alpha w|| / ||v|| = sqrt(1 - |<w, v>|^2 / (||w||^2 ||v||^2)),

which lies in [0, 1] — 0 is a perfect structural match.  ``fidelity`` is
the complementary ``1 - rel_error**2`` (the squared cosine overlap).

Patterns whose error ties (e.g. several families reproduce the vector
exactly) are broken by predicted gate count, so the cheapest exact
encoding ranks first.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .types import (
    STEP, SQUARE, WALSH, SPARSE, FOURIER,
    GEOMETRIC, HAMMING, STAIRCASE, DICKE, POLYNOMIAL,
)
from .predictor import predict_gates


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class PatternMatch:
    """One fitted pattern family and how well it matches the target vector.

    Attributes
    ----------
    pattern_name : str
        Family name, e.g. ``"GEOMETRIC"``.
    pattern : _Pattern
        A ready-to-``encode`` constructor with the fitted parameters (the
        leading amplitude is scaled so the pattern's vector best matches the
        target in magnitude, not just in shape).
    rel_error : float
        Scale/phase-invariant relative error in [0, 1]; 0 is exact.
    fidelity : float
        ``1 - rel_error**2`` — the squared overlap with the target.
    params : dict
        The fitted parameters (same as ``pattern.params``).
    gate_count : Optional[int]
        Predicted total transpiled gate count from ``predict_gates``, or
        ``None`` if prediction was unavailable. Used to break error ties.
    """
    pattern_name: str
    pattern: object
    rel_error: float
    fidelity: float
    params: dict = field(default_factory=dict)
    gate_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def match_vector(v, top_k: int = 3, families=None, max_fourier_modes: int = 4,
                 max_poly_degree: int = 3):
    """Rank the exact pattern families by how well they match *v*.

    Parameters
    ----------
    v : array_like
        Target amplitude vector. Length N must be a power of two. Real or
        complex; it need not be normalized (the metric is scale-invariant).
    top_k : int, optional
        Number of best matches to return (default 3). Pass ``None`` for all.
    families : list of str, optional
        Restrict the search to these family names (e.g. ``["STEP",
        "GEOMETRIC"]``). Default: try all families.
    max_fourier_modes : int, optional
        Maximum number of sinusoidal modes to fit for the FOURIER family
        (greedy, dominant bins first). Default 4.
    max_poly_degree : int, optional
        Highest polynomial degree to try for the POLYNOMIAL family.
        Default 3.

    Returns
    -------
    list of PatternMatch
        Sorted by ``rel_error`` ascending, ties broken by predicted gate
        count. Length ``min(top_k, n_families)``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyencode import match_vector, encode
    >>> v = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)   # STEP(k_e=4)
    >>> matches = match_vector(v)
    >>> matches[0].pattern_name
    'STEP'
    >>> circuit, info = encode(matches[0].pattern, N=len(v))
    """
    v = np.asarray(v)
    if v.dtype.kind not in "fc":
        v = v.astype(float)
    N = v.size
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError(
            f"match_vector requires a power-of-two length vector, got N={N}."
        )
    vnorm2 = float(np.vdot(v, v).real)
    if vnorm2 <= 0.0:
        raise ValueError("match_vector requires a nonzero vector.")
    m = int(round(math.log2(N)))

    fitters = {
        "SPARSE":     _fit_sparse,
        "STEP":       _fit_step,
        "SQUARE":     _fit_square,
        "WALSH":      _fit_walsh,
        "FOURIER":    lambda v, N, m: _fit_fourier(v, N, m, max_fourier_modes),
        "GEOMETRIC":  _fit_geometric,
        "HAMMING":    _fit_hamming,
        "STAIRCASE":  _fit_staircase,
        "DICKE":      _fit_dicke,
        "POLYNOMIAL": lambda v, N, m: _fit_polynomial(v, N, m, max_poly_degree),
    }

    if families is not None:
        unknown = set(families) - set(fitters)
        if unknown:
            raise ValueError(f"Unknown families: {sorted(unknown)}")
        selected = [f for f in fitters if f in set(families)]
    else:
        selected = list(fitters)

    matches: list[PatternMatch] = []
    for name in selected:
        try:
            pattern = fitters[name](v, N, m)
        except Exception as exc:                       # pragma: no cover
            warnings.warn(f"match_vector: {name} fit failed ({exc}); skipped.")
            pattern = None
        if pattern is None:
            continue
        rel_error = _rel_error_of(v, pattern, N, vnorm2)
        if rel_error is None:
            continue
        gate_count = _safe_gate_count(pattern, N)
        matches.append(PatternMatch(
            pattern_name=name,
            pattern=pattern,
            rel_error=rel_error,
            fidelity=max(0.0, 1.0 - rel_error ** 2),
            params=pattern.params,
            gate_count=gate_count,
        ))

    # Primary key: error (rounded so near-exact ties group together).
    # Secondary key: cheaper circuit wins the tie.
    matches.sort(key=lambda mt: (round(mt.rel_error, 9),
                                 mt.gate_count if mt.gate_count is not None
                                 else float("inf")))
    if top_k is not None:
        matches = matches[:top_k]
    return matches


def print_matches(matches) -> None:
    """Print a ranked table of :class:`PatternMatch` results."""
    print(format_matches(matches))


def format_matches(matches) -> str:
    """Return a human-readable ranked table of matches."""
    if not matches:
        return "(no pattern matched)"
    lines = [f"{'rank':>4}  {'pattern':<11}  {'rel_error':>10}  "
             f"{'fidelity':>9}  {'gates':>6}  params"]
    for i, mt in enumerate(matches, 1):
        gates = "?" if mt.gate_count is None else str(mt.gate_count)
        params = _format_params(mt.params)
        lines.append(f"{i:>4}  {mt.pattern_name:<11}  {mt.rel_error:>10.3e}  "
                     f"{mt.fidelity:>9.6f}  {gates:>6}  {params}")
    return "\n".join(lines)


def _format_params(params: dict) -> str:
    def fmt(x):
        if isinstance(x, complex):
            return f"{x:.4g}"
        if isinstance(x, float):
            return f"{x:.4g}"
        return repr(x)
    parts = []
    for k, val in params.items():
        if isinstance(val, list):
            parts.append(f"{k}=[{len(val)} entries]")
        else:
            parts.append(f"{k}={fmt(val)}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score(v, w, vnorm2: float):
    """Best scale/phase-invariant fit of candidate *w* to target *v*.

    Returns ``(rel_error, alpha)`` where ``alpha`` is the optimal complex
    scale and ``rel_error`` = ||v - alpha w|| / ||v|| in [0, 1]. If *w* is
    the zero vector, returns ``(1.0, 0.0)``.
    """
    wnorm2 = float(np.vdot(w, w).real)
    if wnorm2 <= 0.0:
        return 1.0, 0.0
    overlap = np.vdot(w, v)                       # sum(conj(w) * v)
    cos2 = (abs(overlap) ** 2) / (wnorm2 * vnorm2)
    cos2 = min(1.0, max(0.0, cos2))
    alpha = overlap / wnorm2
    return math.sqrt(1.0 - cos2), alpha


def _rel_error_of(v, pattern, N: int, vnorm2: float):
    """Relative error of a fitted *pattern* against *v* via its built vector."""
    from ._helpers import _build_component_vector
    try:
        w = _build_component_vector(pattern, N)
    except Exception:                                  # pragma: no cover
        return None
    rel_error, _ = _score(v, np.asarray(w), vnorm2)
    return rel_error


def _safe_gate_count(pattern, N: int):
    try:
        return int(predict_gates(pattern, N)["gate_count"])
    except Exception:                                  # pragma: no cover
        return None


def _maximize_1d(f, lo, hi, iters=60):
    """Golden-section search for the argmax of scalar *f* on [lo, hi]."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = f(d)
    return (lo + hi) / 2.0


def _fit_ratio(build, v, vnorm2, ranges, n_grid=80):
    """Fit a single real ratio r maximizing fidelity of ``build(r)`` to v.

    *ranges* is a list of (lo, hi) intervals to scan (e.g. positive and
    negative r). A coarse grid brackets the global optimum, then golden
    section refines it. Returns the best r found.
    """
    def cos2(r):
        return 1.0 - _score(v, build(r), vnorm2)[0] ** 2

    best_r, best = None, -1.0
    for lo, hi in ranges:
        grid = np.linspace(lo, hi, n_grid)
        for r in grid:
            c = cos2(float(r))
            if c > best:
                best, best_r = c, float(r)
        step = (hi - lo) / (n_grid - 1)
        r_ref = _maximize_1d(cos2, max(lo, best_r - step),
                             min(hi, best_r + step))
        c_ref = cos2(r_ref)
        if c_ref > best:
            best, best_r = c_ref, r_ref
    return best_r


def _lstsq_residual(v, basis, vnorm2):
    """Least-squares projection of v onto columns of *basis*.

    Returns ``(coeffs, rel_error)``. *basis* is a (N, k) array.
    """
    coeffs, *_ = np.linalg.lstsq(basis, v, rcond=None)
    residual = v - basis @ coeffs
    rel_error = math.sqrt(max(0.0, float(np.vdot(residual, residual).real) / vnorm2))
    return coeffs, rel_error


# ---------------------------------------------------------------------------
# Per-family fitters.  Each returns a constructor (or None if inapplicable).
# ---------------------------------------------------------------------------

def _fit_sparse(v, N, m):
    """SPARSE reproduces any vector exactly from its nonzero entries."""
    nz = np.flatnonzero(np.abs(v) > 1e-12)
    if nz.size == 0:
        return None
    entries = [(int(k), complex(v[k]) if np.iscomplexobj(v) else float(v[k]))
               for k in nz]
    return SPARSE(entries)


def _fit_step(v, N, m):
    """STEP: best prefix [0, k_e) with uniform amplitude."""
    vnorm2 = float(np.vdot(v, v).real)
    prefix = np.concatenate([[0.0 + 0.0j if np.iscomplexobj(v) else 0.0],
                             np.cumsum(v)])
    best_ke, best = None, -1.0
    for k_e in range(1, N + 1):
        ov = prefix[k_e]
        cos2 = (abs(ov) ** 2) / (k_e * vnorm2)
        if cos2 > best:
            best, best_ke = cos2, k_e
    alpha = prefix[best_ke] / best_ke
    return STEP(k_e=best_ke, c=_real_if_real(alpha, v))


def _fit_square(v, N, m):
    """SQUARE: best interval [k_s, k_e) with uniform amplitude (O(N^2))."""
    if N > 4096:
        # The brute-force interval search is O(N^2); fall back to STEP-shaped
        # search (k_s = 0) for very large N to stay responsive.
        return _fit_step(v, N, m)
    vnorm2 = float(np.vdot(v, v).real)
    zero = 0.0 + 0.0j if np.iscomplexobj(v) else 0.0
    prefix = np.concatenate([[zero], np.cumsum(v)])
    best, best_ks, best_ke = -1.0, 0, 1
    lengths_all = np.arange(1, N + 1)
    for k_s in range(N):
        seg = prefix[k_s + 1:] - prefix[k_s]            # sums of [k_s, k_e)
        lengths = lengths_all[: N - k_s]
        cos2 = (np.abs(seg) ** 2) / (lengths * vnorm2)
        j = int(np.argmax(cos2))
        if cos2[j] > best:
            best, best_ks, best_ke = float(cos2[j]), k_s, k_s + j + 1
    width = best_ke - best_ks
    alpha = (prefix[best_ke] - prefix[best_ks]) / width
    return SQUARE(k_s=best_ks, k_e=best_ke, c=_real_if_real(alpha, v))


def _fit_walsh(v, N, m):
    """WALSH: best single bit k splitting the register into two levels."""
    vnorm2 = float(np.vdot(v, v).real)
    idx = np.arange(N)
    best_k, best_err = None, math.inf
    best_c0 = best_c1 = None
    for k in range(m):
        bit = (idx >> k) & 1
        e0 = (bit == 0).astype(float)
        e1 = (bit == 1).astype(float)
        basis = np.column_stack([e0, e1]).astype(v.dtype)
        (c0, c1), rel_error = _lstsq_residual(v, basis, vnorm2)
        if rel_error < best_err:
            best_err, best_k = rel_error, k
            best_c0, best_c1 = c0, c1
    if best_k is None:
        return None
    return WALSH(k=best_k, c0=_real_if_real(best_c0, v),
                 c1=_real_if_real(best_c1, v))


def _fit_fourier(v, N, m, max_modes):
    """FOURIER: greedily fit up to *max_modes* dominant sinusoidal modes."""
    if np.iscomplexobj(v):
        # FOURIER's (n, A, phi) parameterization yields a real signal; only
        # fit the real part's structure (complex v will just score poorly).
        v_fit = v.real.astype(float)
    else:
        v_fit = v.astype(float)
    vnorm2 = float(np.dot(v_fit, v_fit))
    if vnorm2 <= 0.0:
        return None
    k = np.arange(N)
    # Power per integer frequency n via orthogonal sin/cos projection.
    powers = []
    for n in range(1, N // 2 + 1):
        s = np.sin(2 * math.pi * n * k / N)
        c = np.cos(2 * math.pi * n * k / N)
        sn2 = float(np.dot(s, s))
        cn2 = float(np.dot(c, c))
        b = float(np.dot(s, v_fit)) / sn2 if sn2 > 1e-12 else 0.0   # sin coeff
        a = float(np.dot(c, v_fit)) / cn2 if cn2 > 1e-12 else 0.0   # cos coeff
        power = (b ** 2) * sn2 + (a ** 2) * cn2
        powers.append((power, n, a, b))
    powers.sort(reverse=True)
    modes = []
    for power, n, a, b in powers[:max_modes]:
        if power <= 1e-12 * vnorm2:
            break
        A = math.hypot(a, b)
        phi = math.atan2(a, b)        # A sin(.) cos(phi)=b, A cos(.) -> a
        modes.append((int(n), float(A), float(phi)))
    if not modes:
        return None
    return FOURIER(modes=modes)


def _fit_geometric(v, N, m):
    """GEOMETRIC (full window): f_i = c * r^i, fit real ratio r."""
    vnorm2 = float(np.vdot(v, v).real)
    idx = np.arange(N, dtype=float)

    def build(r):
        return r ** idx

    r = _fit_ratio(build, v, vnorm2,
                   ranges=[(0.02, 2.5), (-2.5, -0.02)])
    if r is None or abs(r - 1.0) < 1e-3:
        return None
    _, alpha = _score(v, build(r), vnorm2)
    return GEOMETRIC(r=_real_if_real(r, v), c=_real_if_real(alpha, v))


def _fit_hamming(v, N, m):
    """HAMMING: f_i = c * r^{wt(i)}, fit real ratio r."""
    vnorm2 = float(np.vdot(v, v).real)
    pops = np.array([bin(i).count("1") for i in range(N)], dtype=float)

    def build(r):
        return r ** pops

    r = _fit_ratio(build, v, vnorm2,
                   ranges=[(0.02, 2.5), (-2.5, -0.02)])
    if r is None:
        return None
    _, alpha = _score(v, build(r), vnorm2)
    return HAMMING(r=_real_if_real(r, v), c=_real_if_real(alpha, v))


def _fit_staircase(v, N, m):
    """STAIRCASE: f_{2^k - 1} = c * r^k on unary indices, fit real ratio r."""
    vnorm2 = float(np.vdot(v, v).real)
    unary = [(1 << k) - 1 for k in range(m + 1)]

    def build(r):
        zero = 0.0 + 0.0j if np.iscomplexobj(v) else 0.0
        w = np.full(N, zero)
        for k, ix in enumerate(unary):
            w[ix] = r ** k
        return w

    r = _fit_ratio(build, v, vnorm2,
                   ranges=[(0.02, 2.5), (-2.5, -0.02)])
    if r is None or abs(r - 1.0) < 1e-3:
        return None
    _, alpha = _score(v, build(r), vnorm2)
    return STAIRCASE(r=_real_if_real(r, v), c=_real_if_real(alpha, v))


def _fit_dicke(v, N, m):
    """DICKE: uniform over weight-k indices, fit the best weight class k."""
    vnorm2 = float(np.vdot(v, v).real)
    pops = np.array([bin(i).count("1") for i in range(N)])
    best_k, best, best_alpha = None, -1.0, None
    for k in range(m + 1):
        w = (pops == k).astype(v.dtype)
        rel_error, alpha = _score(v, w, vnorm2)
        cos2 = 1.0 - rel_error ** 2
        if cos2 > best:
            best, best_k, best_alpha = cos2, k, alpha
    if best_k is None:
        return None
    return DICKE(k=best_k, c=_real_if_real(best_alpha, v))


def _fit_polynomial(v, N, m, max_degree):
    """POLYNOMIAL: least-squares fit over degrees 1..max_degree, best wins."""
    vnorm2 = float(np.vdot(v, v).real)
    if N > 1:
        x = np.arange(N, dtype=float) / (N - 1)
    else:
        x = np.arange(N, dtype=float)
    best_coeffs, best_err = None, math.inf
    for d in range(1, min(max_degree, N - 1) + 1):
        basis = np.column_stack([x ** j for j in range(d + 1)]).astype(v.dtype)
        coeffs, rel_error = _lstsq_residual(v, basis, vnorm2)
        if rel_error < best_err - 1e-12:
            best_err, best_coeffs = rel_error, coeffs
    if best_coeffs is None:
        return None
    coeffs = [_real_if_real(c, v) for c in best_coeffs]
    return POLYNOMIAL(coeffs=coeffs)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _real_if_real(z, v):
    """Cast a fitted amplitude to float when the target is real-valued."""
    if np.iscomplexobj(v):
        return complex(z)
    return float(np.real(z))
