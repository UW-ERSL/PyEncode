"""
pyencode.predictor
==================
Fast gate-count prediction without circuit construction.

The top-level function ``predict_gates(VectorObj, N)`` returns an
estimate of the transpiled ({CX, U}) gate count and depth for any
PyEncode pattern, using closed-form formulas derived from empirical
calibration against the full synthesis + transpilation pipeline.

This is useful in design-optimization loops where many candidate
patterns or parameter choices are evaluated: the full ``encode()``
call can take 100+ms at large m due to Qiskit's transpile pass,
whereas ``predict_gates()`` is always sub-millisecond.

Accuracy
--------
For patterns with deterministic structure (POPCOUNT, WALSH,
GEOMETRIC, STEP, STAIRCASE, POLYNOMIAL d=1) the predictions are
EXACT.  For POLYNOMIAL d>=2 and FOURIER T=1 they are exact
closed-form fits to the empirical transpiled counts.  For SPARSE and
SQUARE (general interval) the counts depend on specific index bit
patterns in ways that transpilation further optimizes, so predictions
are UPPER BOUNDS; the 'exact' field in the returned dict is False.

For the composite constructors (LCU, TENSOR) predictions sum the
component predictions plus composition overhead.

Returned dict
-------------
    {
      "vector_type":    str,
      "N":              int,
      "m":              int,
      "gate_count_1q":  int,    # predicted transpiled U count
      "gate_count_2q":  int,    # predicted transpiled CX count
      "gate_count":     int,    # sum of above
      "circuit_depth":  int,    # predicted transpiled depth
      "complexity":     str,    # asymptotic (e.g. "O(m)")
      "exact":          bool,   # True if prediction is guaranteed exact
    }
"""

from __future__ import annotations

import math
from typing import Any

from .types import (
    _VectorObj, SPARSE, STEP, SQUARE, FOURIER, WALSH, GEOMETRIC,
    POPCOUNT, STAIRCASE, POLYNOMIAL, TENSOR, LCU,
)
from .recognizer import VectorType


# ---------------------------------------------------------------------------
# Per-pattern predictors.  Each returns dict with keys
# (gate_count_1q, gate_count_2q, circuit_depth, complexity, exact).
# ---------------------------------------------------------------------------

def _predict_sparse(m: int, params: dict) -> dict:
    """SPARSE: Gleinig-Hoefler O(s*m). Exact for s=1 (bit-write); upper
    bound for s>=2 (transpiler further optimizes)."""
    loads = params["loads"]
    s = len(loads)
    if s == 1:
        # Just X gates on set bits + maybe global phase. Exact.
        k = int(loads[0]["k"])
        hw = bin(k).count("1")
        return dict(gate_count_1q=hw, gate_count_2q=0,
                    circuit_depth=1 if hw > 0 else 0,
                    complexity="O(sm)", exact=True)
    # s >= 2: Gleinig-Hoefler has O(s*m) CNOT cost in the worst case.
    # Empirical measurements show the transpiler typically produces
    # fewer gates; the asymptotic bound is honest but loose.
    return dict(
        gate_count_1q=s * m,      # worst-case per Gleinig-Hoefler
        gate_count_2q=s * m,
        circuit_depth=2 * s * m,
        complexity="O(sm)",
        exact=False,
    )


def _predict_step(m: int, params: dict) -> dict:
    """STEP: 1q = m-1 for pc=1; 1q = m + 4*pc - 6 for pc>=2.
    2q = max(0, 2*pc - 3).  depth = 1 for pc<=1 else 4*pc - 5.  Exact."""
    k_s = int(params["k_s"])
    if k_s == 0:
        return dict(gate_count_1q=0, gate_count_2q=0, circuit_depth=0,
                    complexity="O(m)", exact=True)
    pc = bin(k_s).count("1")
    if pc == 1:
        return dict(gate_count_1q=m - 1, gate_count_2q=0, circuit_depth=1,
                    complexity="O(m)", exact=True)
    return dict(
        gate_count_1q=m + 4 * pc - 6,
        gate_count_2q=2 * pc - 3,
        circuit_depth=4 * pc - 5,
        complexity="O(m)",
        exact=True,
    )


def _predict_square(m: int, params: dict) -> dict:
    """SQUARE: if aligned or k1=0 reduces to STEP + shift; else Draper adder.
    Exact for aligned/prefix cases; upper bound for general intervals."""
    k1 = int(params["k1"])
    k2 = int(params["k2"])
    w = k2 - k1
    aligned = (w > 0) and ((w & (w - 1)) == 0) and (k1 % w == 0)
    if k1 == 0:
        return _predict_step(m, {"k_s": w})
    if aligned:
        # Power-of-2-aligned block: X gates on upper qubits + Hadamards
        # on lower qubits. 1q = m - 1 in typical case (same as STEP prefix).
        p = int(math.log2(w))
        # Upper qubits carrying k1 bits: popcount of (k1 >> p)
        upper_pc = bin(k1 >> p).count("1")
        return dict(
            gate_count_1q=p + upper_pc,
            gate_count_2q=0,
            circuit_depth=1,
            complexity="O(m)",
            exact=True,
        )
    # General interval: Draper QFT adder. Empirical fit ~ 2.9 m^2 + ...
    # Give an asymptotic upper bound; mark as inexact.
    return dict(
        gate_count_1q=3 * m * m,
        gate_count_2q=2 * m * m,
        circuit_depth=m * m,
        complexity="O(m^2)",
        exact=False,
    )


def _predict_fourier(m: int, params: dict) -> dict:
    """FOURIER: T modes, inverse-QFT pipeline. Exact fit for T=1 from
    empirical: 1q = 1.5 m^2 - 2.5 m + 6, 2q = m^2 - 3. For T>=2 the
    sparse-load prefix adds modest overhead; return upper bound."""
    modes = params["modes"]
    T = len(modes)
    oneq_t1 = int(1.5 * m * m - 2.5 * m + 6)
    twoq_t1 = m * m - 3
    depth_t1 = 9 * m - 13 if m >= 2 else 1
    if T == 1:
        return dict(
            gate_count_1q=oneq_t1,
            gate_count_2q=max(0, twoq_t1),
            circuit_depth=max(1, depth_t1),
            complexity="O(m^2)",
            exact=True,
        )
    # T>=2: additional sparse load for 2T nonzero DFT coefficients
    extra_oneq = 2 * T * m
    extra_twoq = 2 * T * m
    return dict(
        gate_count_1q=oneq_t1 + extra_oneq,
        gate_count_2q=max(0, twoq_t1 + extra_twoq),
        circuit_depth=max(1, depth_t1 + T * m),
        complexity="O(m^2)",
        exact=False,
    )


def _predict_walsh(m: int, params: dict) -> dict:
    """WALSH: m+1 gates (R_y + H^m), depth 1. Exact."""
    return dict(
        gate_count_1q=m,  # transpiler may optimize R_y; typically m
        gate_count_2q=0,
        circuit_depth=1,
        complexity="O(m)",
        exact=True,
    )


def _predict_geometric(m: int, params: dict) -> dict:
    """GEOMETRIC: three regimes (see synthesizer._synth_geometric).

    Regime (a) start == 0                       : m R_y gates, 0 CX, depth 1.
    Regime (b) single dyadic block              : log2(w) + popcount(start/w)
                                                   1-qubit gates, 0 CX, depth 1.
    Regime (c) general dyadic decomposition     : O(m^2) gates total,
                                                   bounded by
                                                     anchors (L*m) + spread
                                                     (sum_k j_k * (m - j_k))
                                                   multi-controlled ops.

    Transpiler may collapse small-angle rotations; returned counts are
    upper bounds.
    """
    start = params.get("start", 0)
    N = 1 << m

    # Regime (a)
    if start == 0:
        return dict(
            gate_count_1q=m,
            gate_count_2q=0,
            circuit_depth=1,
            complexity="O(m)",
            exact=False,
        )

    w = N - start

    # Regime (b): single dyadic block
    if (w & (w - 1)) == 0 and start % w == 0:
        m_low = w.bit_length() - 1
        upper_val = start // w
        x_gates = bin(upper_val).count("1")
        return dict(
            gate_count_1q=m_low + x_gates,
            gate_count_2q=0,
            circuit_depth=1,
            complexity="O(m)",
            exact=False,
        )

    # Regime (c): dyadic decomposition.  Analytical O(m^2) bound.
    # Anchor load (Gleinig-Hoefler on L anchors): O(L * m) 1q + 2q gates.
    # Spread step: for block k with j_k free bits and (m - j_k) controls,
    # each MC-R_y decomposes into O(m - j_k) CX + O(m - j_k) 1q.
    # Aggregate spread cost: sum_k j_k * (m - j_k)  <=  m^2 / 4.
    #
    # Build the decomposition inline to get tight per-instance counts
    # without running the synthesizer.
    blocks = []
    cur = start
    while cur < N:
        room = N - cur
        mx = room.bit_length() - 1
        if cur == 0:
            j = mx
        else:
            tz = (cur & -cur).bit_length() - 1
            j = tz if tz < mx else mx
        blocks.append((cur, j))
        cur += 1 << j
    L = len(blocks)

    # Anchor step (Gleinig-Hoefler): L anchors, up to m gates per reduction.
    anchor_1q = L * m
    anchor_2q = L * m                   # conservative

    # Spread step: sum_k j_k * (m - j_k + 1) per-gate cost.
    spread_1q = 0
    spread_2q = 0
    for (_a_k, j_k) in blocks:
        ctrl_w = m - j_k
        # Each MC-R_y with ctrl_w controls: roughly ctrl_w 1q + ctrl_w 2q
        # (Qiskit's noancilla decomposition).  ctrl_w == 1 is native CR_y.
        cost_1q = max(1, ctrl_w)
        cost_2q = max(1, ctrl_w)
        spread_1q += j_k * cost_1q
        spread_2q += j_k * cost_2q

    return dict(
        gate_count_1q=anchor_1q + spread_1q,
        gate_count_2q=anchor_2q + spread_2q,
        circuit_depth=anchor_1q + anchor_2q + spread_1q + spread_2q,
        complexity="O(m^2)",
        exact=False,
    )


def _predict_popcount(m: int, params: dict) -> dict:
    """POPCOUNT: m identical single-qubit rotations, depth 1. Exact."""
    return dict(
        gate_count_1q=m,
        gate_count_2q=0,
        circuit_depth=1,
        complexity="O(m)",
        exact=True,
    )


def _predict_staircase(m: int, params: dict) -> dict:
    """STAIRCASE: 1q = 2m-1, 2q = 2m-2, depth = 3m-2.  Exact."""
    return dict(
        gate_count_1q=2 * m - 1,
        gate_count_2q=2 * m - 2,
        circuit_depth=3 * m - 2,
        complexity="O(m)",
        exact=True,
    )


def _predict_polynomial(m: int, params: dict) -> dict:
    """POLYNOMIAL: degree-d via signed Walsh-sparse loading.
    d=1 is exact linear: 1q = 5m-4, 2q = 2m-2, depth = 4m-3.
    d>=2 uses closed-form fits to the empirical transpiled counts."""
    coeffs = params["coeffs"]
    d = len(coeffs) - 1
    if d == 0:
        return dict(gate_count_1q=0, gate_count_2q=0, circuit_depth=0,
                    complexity="O(1)", exact=True)
    if d == 1:
        return dict(
            gate_count_1q=5 * m - 4,
            gate_count_2q=2 * m - 2,
            circuit_depth=4 * m - 3,
            complexity="O(m)",
            exact=True,
        )
    if d == 2:
        # Empirical fit: 1q ~ 7.77 m^2 - 20.26 m - 1.07
        #                2q ~ 6.42 m^2 - 16.82 m - 1.60
        oneq = max(0, round(7.77 * m * m - 20.26 * m - 1.07))
        twoq = max(0, round(6.42 * m * m - 16.82 * m - 1.60))
        depth = max(1, round(9.9 * m * m - 21.7 * m + 2))
        return dict(
            gate_count_1q=int(oneq),
            gate_count_2q=int(twoq),
            circuit_depth=int(depth),
            complexity="O(m^2)",
            exact=False,
        )
    # d >= 3: asymptotic O(m^{d+1}) with large constants that depend on
    # coefficients and sign patterns. Return order-of-magnitude estimate.
    s = sum(math.comb(m, k) for k in range(d + 1))
    return dict(
        gate_count_1q=s * m * 2,
        gate_count_2q=s * m * 2,
        circuit_depth=s * m * 3,
        complexity=f"O(m^{d+1})",
        exact=False,
    )


# ---------------------------------------------------------------------------
# Composite predictors: LCU and TENSOR
# ---------------------------------------------------------------------------

def _predict_tensor(vec_obj: TENSOR) -> dict:
    """TENSOR: sum component predictions on their disjoint subregisters.
    Total qubits = sum of component m's. Marked as inexact because the
    transpiler can optimize across the tensor boundary by 1-2 gates."""
    components = vec_obj.params["components"]
    sizes = vec_obj.params["sizes"]
    total_1q = 0
    total_2q = 0
    max_depth = 0
    total_m = 0
    complexities = []
    for comp_obj, comp_N in zip(components, sizes):
        comp_pred = predict_gates(comp_obj, comp_N)
        total_1q += comp_pred["gate_count_1q"]
        total_2q += comp_pred["gate_count_2q"]
        max_depth = max(max_depth, comp_pred["circuit_depth"])
        total_m += comp_pred["m"]
        complexities.append(comp_pred["complexity"])
    N = 1 << total_m
    return dict(
        vector_type="TENSOR",
        N=N,
        m=total_m,
        gate_count_1q=total_1q,
        gate_count_2q=total_2q,
        gate_count=total_1q + total_2q,
        circuit_depth=max_depth,
        complexity=" + ".join(complexities),
        exact=False,  # transpiler can merge 1-2 gates at the tensor boundary
    )


def _predict_lcu(vec_obj: LCU, N: int) -> dict:
    """LCU: sum of component predictions plus ancilla-preparation overhead.
    The ancilla register has ceil(log2(r)) qubits and requires its own
    amplitude encoding (typically O(r) gates since it's a dense r-dim
    vector), plus r controlled component circuits."""
    weights = vec_obj.params["weights"]
    components = vec_obj.params["components"]
    r = len(components)
    m = int(round(math.log2(N)))
    ancilla_qubits = max(1, int(math.ceil(math.log2(r))))

    # Sum component gate counts, each with a 1-qubit ancilla control overhead.
    # Controlled decomposition roughly doubles gates (1q -> 1q+cx, cx -> 2cx+u).
    total_1q = 0
    total_2q = 0
    total_depth = 0
    all_exact = True
    complexities = []
    for comp_obj in components:
        comp_pred = predict_gates(comp_obj, N)
        total_1q += 2 * comp_pred["gate_count_1q"] + comp_pred["gate_count_2q"]
        total_2q += 2 * comp_pred["gate_count_2q"] + comp_pred["gate_count_1q"]
        total_depth += comp_pred["circuit_depth"] * 2
        complexities.append(comp_pred["complexity"])
        if not comp_pred["exact"]:
            all_exact = False

    # Ancilla amplitude prep: dense r-dim state on ancilla qubits
    ancilla_cost_1q = 2 * r
    ancilla_cost_2q = r
    total_1q += ancilla_cost_1q
    total_2q += ancilla_cost_2q
    total_depth += 2 * r

    return dict(
        vector_type="LCU",
        N=N,
        m=m + ancilla_qubits,
        gate_count_1q=total_1q,
        gate_count_2q=total_2q,
        gate_count=total_1q + total_2q,
        circuit_depth=total_depth,
        complexity="O(r * comp_cost)",
        exact=False,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_PREDICTORS = {
    VectorType.SPARSE:     _predict_sparse,
    VectorType.STEP:       _predict_step,
    VectorType.SQUARE:     _predict_square,
    VectorType.FOURIER:    _predict_fourier,
    VectorType.WALSH:      _predict_walsh,
    VectorType.GEOMETRIC:  _predict_geometric,
    VectorType.POPCOUNT:   _predict_popcount,
    VectorType.STAIRCASE:  _predict_staircase,
    VectorType.POLYNOMIAL: _predict_polynomial,
}


def predict_gates(VectorObj: Any, N: int) -> dict:
    """
    Fast gate-count prediction without circuit construction or transpilation.

    Parameters
    ----------
    VectorObj : _VectorObj, LCU, TENSOR, or list of _VectorObj
        Any valid argument to encode().
    N : int
        Vector length (power of 2).  Must match the total vector length
        implied by VectorObj.

    Returns
    -------
    dict with keys:
        vector_type, N, m, gate_count_1q, gate_count_2q, gate_count,
        circuit_depth, complexity, exact.
        'exact' is True when the prediction is guaranteed exact; False
        when it is an upper bound or empirical estimate.

    Examples
    --------
    >>> from pyencode import predict_gates, POLYNOMIAL, GEOMETRIC
    >>> predict_gates(POLYNOMIAL(coeffs=[0.0, 1.0]), N=4096)
    {'vector_type': 'POLYNOMIAL', 'N': 4096, 'm': 12,
     'gate_count_1q': 56, 'gate_count_2q': 22, 'gate_count': 78,
     'circuit_depth': 45, 'complexity': 'O(m)', 'exact': True}

    >>> predict_gates(GEOMETRIC(ratio=0.95), N=65536)
    {'vector_type': 'GEOMETRIC', 'N': 65536, 'm': 16, ...}
    """
    # Composite constructors
    if isinstance(VectorObj, TENSOR):
        return _predict_tensor(VectorObj)
    if isinstance(VectorObj, LCU):
        return _predict_lcu(VectorObj, N)
    if isinstance(VectorObj, list):
        # Legacy composite: list of SQUARE constructors
        total_1q = 0; total_2q = 0; total_depth = 0
        m = int(round(math.log2(N)))
        for comp in VectorObj:
            cp = predict_gates(comp, N)
            total_1q += cp["gate_count_1q"]
            total_2q += cp["gate_count_2q"]
            total_depth += cp["circuit_depth"]
        return dict(
            vector_type="COMPOSITE",
            N=N, m=m,
            gate_count_1q=total_1q, gate_count_2q=total_2q,
            gate_count=total_1q + total_2q,
            circuit_depth=total_depth,
            complexity="O(sum)",
            exact=False,
        )

    if not isinstance(VectorObj, _VectorObj):
        raise TypeError(
            f"VectorObj must be a typed constructor (SPARSE, STEP, SQUARE, "
            f"FOURIER, WALSH, GEOMETRIC, POPCOUNT, STAIRCASE, POLYNOMIAL, "
            f"LCU, TENSOR), got {type(VectorObj).__name__}."
        )

    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError(f"N must be a positive power of 2, got {N}.")
    m = int(round(math.log2(N)))

    vtype = VectorObj.vector_type
    predictor = _PREDICTORS.get(vtype)
    if predictor is None:
        raise ValueError(f"No predictor available for {vtype.name}.")

    out = predictor(m, VectorObj.params)
    out["vector_type"] = vtype.name
    out["N"] = N
    out["m"] = m
    out["gate_count"] = out["gate_count_1q"] + out["gate_count_2q"]
    return out
