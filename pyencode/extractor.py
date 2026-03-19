"""
pyencode.extractor
====================
Numerical parameter extraction from a concrete load vector.

Given a vector *f* (numpy array of length *N*) and optionally a
declared :class:`VectorType`, extract the pattern parameters by
numerical analysis (projection, peak detection, etc.) and return
them as a ``dict`` compatible with :class:`LoadPattern`.

Two entry points:

- ``extract(f, vector_type, atol)`` — extract parameters assuming a
  known load type.  Raises if the vector doesn't match.

- ``auto_detect(f, atol)`` — try all load types and return the
  best-fitting ``(VectorType, params)`` pair, preferring simpler
  patterns (lower gate complexity) when multiple match.
"""

import math
import numpy as np

from .recognizer import VectorType


# ── public entry point ────────────────────────────────────────────

def extract(f: np.ndarray, vector_type: VectorType, tol: float = 1e-6) -> dict:
    """
    Extract parameters from vector *f* for the given *vector_type*.

    Parameters
    ----------
    f : np.ndarray
        The concrete load vector, shape ``(N,)``, where *N* is a
        power of 2.
    vector_type : VectorType
        The declared vector type.
    tol : float
        Absolute tolerance for zero / equality checks.

    Returns
    -------
    dict
        Parameters matching the schema for the given *vector_type*.

    Raises
    ------
    ValueError
        If the vector does not match the declared vector type, with a
        diagnostic message explaining *why*.
    """
    fn = _EXTRACTORS.get(vector_type)
    if fn is None:
        valid = [vt.name for vt in _EXTRACTORS]
        raise ValueError(
            f"vector_type={vector_type.name} is not supported for "
            f"extraction.  Supported types: {valid}"
        )
    return fn(f, tol)


# ── individual extractors ─────────────────────────────────────────

def _extract_point_load(f: np.ndarray, atol: float) -> dict:
    """Exactly one nonzero entry."""
    nonzero = np.where(np.abs(f) > atol)[0]
    if len(nonzero) == 0:
        raise ValueError(
            "Vector does not match declared DISCRETE: "
            "all entries are zero."
        )
    if len(nonzero) != 1:
        raise ValueError(
            f"Vector does not match declared DISCRETE: "
            f"found {len(nonzero)} nonzero entries at indices "
            f"{nonzero.tolist()}, expected exactly 1."
        )
    k = int(nonzero[0])
    return {"k": k, "P": float(f[k])}


def _extract_uniform_load(f: np.ndarray, atol: float) -> dict:
    """All entries approximately equal."""
    c = float(np.mean(f))
    max_dev = float(np.max(np.abs(f - c)))
    if max_dev > atol:
        raise ValueError(
            f"Vector does not match declared UNIFORM: "
            f"max deviation from mean = {max_dev:.2e}, "
            f"expected < {atol:.2e}.  Values range from "
            f"{float(np.min(f)):.6f} to {float(np.max(f)):.6f}."
        )
    return {"c": c}


def _extract_step_load(f: np.ndarray, atol: float) -> dict:
    """Prefix of equal nonzero values, remainder zero."""
    near_zero = np.abs(f) < atol

    if np.all(near_zero):
        raise ValueError(
            "Vector does not match declared STEP: "
            "all entries are zero."
        )
    if not np.any(near_zero):
        raise ValueError(
            "Vector does not match declared STEP: "
            "no zero entries found (tail must be zero).  "
            "Did you mean UNIFORM?"
        )

    k_s = int(np.argmax(near_zero))           # first zero index
    if k_s == 0:
        raise ValueError(
            "Vector does not match declared STEP: "
            "f[0] is zero (step must start from index 0)."
        )

    c = float(f[0])
    prefix_dev = float(np.max(np.abs(f[:k_s] - c)))
    if prefix_dev > atol:
        raise ValueError(
            f"Vector does not match declared STEP: "
            f"prefix f[0:{k_s}] is not constant "
            f"(max deviation = {prefix_dev:.2e})."
        )

    tail_max = float(np.max(np.abs(f[k_s:])))
    if tail_max > atol:
        raise ValueError(
            f"Vector does not match declared STEP: "
            f"tail f[{k_s}:] is not zero "
            f"(max |f[i]| = {tail_max:.2e}).  "
            f"Did you mean SQUARE?"
        )

    return {"k_s": k_s, "c": c}


def _extract_square_load(f: np.ndarray, atol: float) -> dict:
    """Contiguous block of equal nonzero values, zeros outside."""
    nonzero_idx = np.where(np.abs(f) > atol)[0]

    if len(nonzero_idx) == 0:
        raise ValueError(
            "Vector does not match declared SQUARE: "
            "all entries are zero."
        )

    k1 = int(nonzero_idx[0])
    k2 = int(nonzero_idx[-1]) + 1
    expected_count = k2 - k1

    if len(nonzero_idx) != expected_count:
        raise ValueError(
            f"Vector does not match declared SQUARE: "
            f"nonzero entries are not contiguous.  "
            f"Found {len(nonzero_idx)} nonzero entries spanning "
            f"indices [{k1}, {k2}), expected {expected_count}."
        )

    c = float(f[k1])
    block_dev = float(np.max(np.abs(f[k1:k2] - c)))
    if block_dev > atol:
        raise ValueError(
            f"Vector does not match declared SQUARE: "
            f"block f[{k1}:{k2}] is not constant "
            f"(max deviation = {block_dev:.2e})."
        )

    return {"k1": k1, "k2": k2, "c": c}


# ── sinusoidal / cosine via projection ────────────────────────────

def _fit_sinusoidal(f: np.ndarray, atol: float, *, use_cos: bool) -> dict:
    """
    Shared engine for SINE and COSINE.

    For each candidate mode *n* in ``[1, N)``, project *f* onto
    ``{sin(n pi k/N), cos(n pi k/N)}`` and solve the 2x2 system for
    *A* and *phi*.  Return the best-fitting ``(n, A, phi)`` if the
    relative residual is below *atol*.

    Parameters
    ----------
    use_cos : bool
        If False, fit ``A sin(n pi k/N + phi)``.
        If True,  fit ``A cos(n pi k/N + phi)``.
    """
    label = "COSINE" if use_cos else "SINE"
    N = len(f)
    k = np.arange(N)
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-14:
        raise ValueError(
            f"Vector does not match declared {label}: vector is zero."
        )

    best_n = None
    best_residual = np.inf
    best_A = None
    best_phi = None

    for n_cand in range(1, N):
        basis = 2 * math.pi * n_cand * k / N
        s = np.sin(basis)
        c = np.cos(basis)

        ss = np.dot(s, s)
        cc = np.dot(c, c)
        sc = np.dot(s, c)
        det = ss * cc - sc * sc
        if abs(det) < 1e-14:
            continue

        fs = np.dot(f, s)
        fc = np.dot(f, c)

        if use_cos:
            # f ~ A cos(basis + phi) = A cos(phi) cos(.) - A sin(phi) sin(.)
            a_coeff = (ss * fc - sc * fs) / det    # A cos(phi)
            b_coeff = (cc * fs - sc * fc) / det    # -A sin(phi)
            A_cand = math.sqrt(a_coeff ** 2 + b_coeff ** 2)
            if A_cand < 1e-14:
                continue
            phi_cand = math.atan2(-b_coeff, a_coeff)
            reconstruction = A_cand * np.cos(basis + phi_cand)
        else:
            # f ~ A sin(basis + phi) = A cos(phi) sin(.) + A sin(phi) cos(.)
            a_coeff = (cc * fs - sc * fc) / det    # A cos(phi)
            b_coeff = (ss * fc - sc * fs) / det    # A sin(phi)
            A_cand = math.sqrt(a_coeff ** 2 + b_coeff ** 2)
            if A_cand < 1e-14:
                continue
            phi_cand = math.atan2(b_coeff, a_coeff)
            reconstruction = A_cand * np.sin(basis + phi_cand)

        residual = np.linalg.norm(f - reconstruction)
        if residual < best_residual:
            best_residual = residual
            best_n = n_cand
            best_A = A_cand
            best_phi = phi_cand

    if best_n is None:
        raise ValueError(
            f"Vector does not match declared {label}: "
            f"no viable candidate mode found."
        )

    rel_residual = best_residual / f_norm
    if rel_residual > atol:
        raise ValueError(
            f"Vector does not match declared {label}: "
            f"best fit is n={best_n} with relative residual "
            f"{rel_residual:.2e} > {atol:.2e}."
        )

    # Snap phi to 0 if negligibly small
    if abs(best_phi) < atol:
        best_phi = 0.0

    return {"n": best_n, "A": best_A, "phi": best_phi}


def _extract_sinusoidal_load(f: np.ndarray, atol: float) -> dict:
    return _fit_sinusoidal(f, atol, use_cos=False)


def _extract_cosine_load(f: np.ndarray, atol: float) -> dict:
    return _fit_sinusoidal(f, atol, use_cos=True)


# ── multi-point ───────────────────────────────────────────────────

def _extract_multi_point_load(f: np.ndarray, atol: float) -> dict:
    """Two or more nonzero entries, rest zero.  Must be sparse."""
    N = len(f)
    nonzero = np.where(np.abs(f) > atol)[0]

    if len(nonzero) < 2:
        raise ValueError(
            f"Vector does not match declared MULTI_DISCRETE: "
            f"found {len(nonzero)} nonzero entries, expected >= 2.  "
            f"Did you mean DISCRETE?"
        )

    # Sparsity check: at most half the entries should be nonzero
    if len(nonzero) > N // 2:
        raise ValueError(
            f"Vector does not match declared MULTI_DISCRETE: "
            f"{len(nonzero)} of {N} entries are nonzero — "
            f"vector is too dense for a multi-point pattern."
        )

    # Verify the remaining entries are truly zero
    zero_mask = np.ones(N, dtype=bool)
    zero_mask[nonzero] = False
    if np.any(zero_mask):
        max_zero_err = float(np.max(np.abs(f[zero_mask])))
        if max_zero_err > atol:
            raise ValueError(
                f"Vector does not match declared MULTI_DISCRETE: "
                f"'zero' entries have max |f[i]| = {max_zero_err:.2e}.  "
                f"Vector is not sparse."
            )

    loads = [{"k": int(i), "P": float(f[i])} for i in nonzero]
    return {"loads": loads}


# ── multi-sin (iterative projection / subtraction) ────────────────

def _extract_multi_sin_load(f: np.ndarray, atol: float) -> dict:
    """Sum of sinusoidal modes (no per-mode phase)."""
    N = len(f)
    k = np.arange(N)
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-14:
        raise ValueError(
            "Vector does not match declared MULTI_SINE: "
            "vector is zero."
        )

    residual = f.copy()
    modes = []
    used_freqs: set = set()

    for _ in range(min(N // 2, 8)):      # practical limit: 8 modes
        best_n = None
        best_A = 0.0

        for n_cand in range(1, N):
            if n_cand in used_freqs:
                continue
            basis = np.sin(2 * math.pi * n_cand * k / N)
            norm_sq = np.dot(basis, basis)
            if norm_sq < 1e-14:
                continue
            A_cand = np.dot(residual, basis) / norm_sq
            if abs(A_cand) > abs(best_A):
                best_A = float(A_cand)
                best_n = n_cand

        if best_n is None or abs(best_A) < atol * f_norm:
            break

        modes.append({"n": best_n, "A": best_A})
        used_freqs.add(best_n)
        residual = residual - best_A * np.sin(2 * math.pi * best_n * k / N)

    if len(modes) < 2:
        raise ValueError(
            f"Vector does not match declared MULTI_SINE: "
            f"found {len(modes)} significant sinusoidal mode(s), "
            f"expected >= 2.  Did you mean SINE?"
        )

    rel_residual = float(np.linalg.norm(residual) / f_norm)
    if rel_residual > atol:
        raise ValueError(
            f"Vector does not match declared MULTI_SINE: "
            f"after extracting {len(modes)} modes, relative residual "
            f"= {rel_residual:.2e} > {atol:.2e}."
        )

    return {"modes": modes}


# ── uniform + spike ───────────────────────────────────────────────

def _extract_uniform_spike_load(f: np.ndarray, atol: float) -> dict:
    """Uniform background *c* with single outlier *delta* at index *k*."""
    N = len(f)
    c_estimate = float(np.median(f))

    deviations = np.abs(f - c_estimate)
    k = int(np.argmax(deviations))
    delta = float(f[k])

    # Recompute c excluding the outlier
    mask = np.ones(N, dtype=bool)
    mask[k] = False
    c = float(np.mean(f[mask]))

    # Verify remaining entries are uniform
    max_dev = float(np.max(np.abs(f[mask] - c)))
    if max_dev > atol:
        raise ValueError(
            f"Vector does not match declared UNIFORM_SPIKE: "
            f"after removing outlier at index {k}, max deviation "
            f"from background = {max_dev:.2e} > {atol:.2e}."
        )

    # Verify the spike is actually different from background
    if abs(delta - c) < atol:
        raise ValueError(
            "Vector does not match declared UNIFORM_SPIKE: "
            "no outlier found (all values approximately equal).  "
            "Did you mean UNIFORM?"
        )

    return {"c": c, "k": k, "delta": delta}


# ── dispatch table ────────────────────────────────────────────────

_EXTRACTORS = {
    VectorType.DISCRETE:         _extract_point_load,
    VectorType.UNIFORM:       _extract_uniform_load,
    VectorType.STEP:          _extract_step_load,
    VectorType.SQUARE:        _extract_square_load,
    VectorType.SINE:    _extract_sinusoidal_load,
    VectorType.COSINE:        _extract_cosine_load,
    VectorType.MULTI_DISCRETE:   _extract_multi_point_load,
    VectorType.MULTI_SINE:     _extract_multi_sin_load,
    VectorType.UNIFORM_SPIKE: _extract_uniform_spike_load,
}


# ── auto-detection ───────────────────────────────────────────────

# Preference order: simpler / cheaper patterns first.
# When multiple types fit the vector perfectly, we pick the one
# that comes first in this list.
_DETECT_ORDER = [
    VectorType.DISCRETE,         # O(m)
    VectorType.UNIFORM,       # O(m)
    VectorType.STEP,          # O(m)
    VectorType.SQUARE,        # O(m)
    VectorType.UNIFORM_SPIKE, # O(m^2) analytical
    VectorType.SINE,    # O(m^2)
    VectorType.COSINE,        # O(m^2)
    VectorType.MULTI_DISCRETE,   # O(m*L)
    VectorType.MULTI_SINE,     # O(m^2)
]


def _reconstruct(vector_type: VectorType, N: int, params: dict) -> np.ndarray:
    """Build the expected vector from extracted parameters."""
    k = np.arange(N)
    p = params

    if vector_type == VectorType.DISCRETE:
        f = np.zeros(N); f[p["k"]] = p["P"]; return f

    if vector_type == VectorType.UNIFORM:
        return np.full(N, p["c"])

    if vector_type == VectorType.STEP:
        f = np.zeros(N); f[:p["k_s"]] = p["c"]; return f

    if vector_type == VectorType.SQUARE:
        f = np.zeros(N); f[p["k1"]:p["k2"]] = p["c"]; return f

    if vector_type == VectorType.SINE:
        return p["A"] * np.sin(2 * math.pi * p["n"] * k / N + p.get("phi", 0.0))

    if vector_type == VectorType.COSINE:
        return p["A"] * np.cos(2 * math.pi * p["n"] * k / N + p.get("phi", 0.0))

    if vector_type == VectorType.MULTI_DISCRETE:
        f = np.zeros(N)
        for ld in p["loads"]:
            f[ld["k"]] = ld["P"]
        return f

    if vector_type == VectorType.MULTI_SINE:
        f = np.zeros(N)
        for mode in p["modes"]:
            f += mode["A"] * np.sin(2 * math.pi * mode["n"] * k / N)
        return f

    if vector_type == VectorType.UNIFORM_SPIKE:
        f = np.full(N, p["c"]); f[p["k"]] = p["delta"]; return f

    return np.zeros(N)


def auto_detect(f: np.ndarray, tol: float = 1e-6) -> tuple[VectorType, dict]:
    """
    Try all known load types and return the best match.

    Parameters
    ----------
    f : np.ndarray
        Load vector of length N (power of 2).
    atol : float
        Tolerance for extraction and reconstruction checks.

    Returns
    -------
    (VectorType, dict)
        The detected load type and its extracted parameters.

    Raises
    ------
    ValueError
        If no load type matches the vector.
    """
    N = len(f)
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-14:
        raise ValueError("Cannot auto-detect load type: vector is zero.")

    for lt in _DETECT_ORDER:
        fn = _EXTRACTORS.get(lt)
        if fn is None:
            continue
        try:
            params = fn(f, tol)
        except ValueError:
            continue

        # Verify reconstruction
        recon = _reconstruct(lt, N, params)
        recon_norm = np.linalg.norm(recon)
        if recon_norm < 1e-14:
            continue
        diff = np.linalg.norm(f / f_norm - recon / recon_norm)
        if diff <= tol:
            # ── sin / cos disambiguation ─────────────────────
            # cos(x) ≡ sin(x + π/2), so the sinusoidal fitter
            # always matches cosines.  When both fit, prefer
            # whichever form has |phi| closer to 0.
            if lt == VectorType.SINE:
                try:
                    cos_p = _extract_cosine_load(f, tol)
                    cos_r = _reconstruct(VectorType.COSINE, N, cos_p)
                    cn = np.linalg.norm(cos_r)
                    if cn > 1e-14:
                        cd = np.linalg.norm(f / f_norm - cos_r / cn)
                        if cd <= tol and abs(cos_p.get("phi", 0.0)) < abs(params.get("phi", 0.0)):
                            return VectorType.COSINE, cos_p
                except ValueError:
                    pass

            return lt, params

    raise ValueError(
        "No known load type matches the vector.  "
        "Use encode_vector(f, vector_type=...) with an explicit type."
    )
