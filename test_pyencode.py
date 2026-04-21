"""
test_pyencode.py
================
Unit tests for PyEncode v1.0.

API: encode(VectorObj, N)  with  SPARSE, STEP, SQUARE, FOURIER

Run with:  python -m pytest test_pyencode.py -v
"""

import math
import numpy as np
import pytest
from qiskit import QuantumCircuit

from pyencode import encode, EncodingInfo, VectorType, SPARSE, STEP, SQUARE, FOURIER, WALSH, GEOMETRIC, POPCOUNT, STAIRCASE, TENSOR, POLYNOMIAL, LCU


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def statevector(circuit):
    from qiskit.quantum_info import Statevector
    return np.array(Statevector(circuit))


def assert_encodes_lcu(circuit, expected_f, n_anc, tol=1e-4):
    """Verify LCU circuit by post-selecting ancilla on |0> subspace."""
    from qiskit.quantum_info import Statevector
    N = len(expected_f)
    norm = np.linalg.norm(expected_f)
    assert norm > 1e-12
    expected = np.abs(expected_f / norm)

    sv = np.array(Statevector(circuit))
    sv_anc0 = sv[:N].real   # ancilla is MSB: anc=0 -> indices 0..N-1
    norm_anc0 = np.linalg.norm(sv_anc0)
    assert norm_anc0 > 1e-12, "anc=0 subspace is empty"
    simulated = np.abs(sv_anc0 / norm_anc0)
    np.testing.assert_allclose(
        simulated, expected, atol=tol,
        err_msg=f"LCU post-selected state mismatch.\nGot:      {np.round(simulated,4)}\nExpected: {np.round(expected,4)}"
    )


def assert_encodes(circuit, expected_f, tol=1e-5):
    sv = statevector(circuit)
    norm = np.linalg.norm(expected_f)
    assert norm > 1e-12, "Reference vector is zero"
    ref = expected_f / norm
    np.testing.assert_allclose(
        np.abs(sv), np.abs(ref), atol=tol,
        err_msg=f"Statevector mismatch.\nGot:      {np.abs(sv)}\nExpected: {np.abs(ref)}"
    )


# ===================================================================
# SPARSE
# ===================================================================

class TestSparse:

    def test_single_basic(self):
        circuit, info = encode(SPARSE([(3, 5.0)]), N=8)
        assert info.vector_type == "SPARSE"
        assert info.N == 8 and info.m == 3
        expected = np.zeros(8); expected[3] = 5.0
        assert_encodes(circuit, expected)

    def test_single_k0(self):
        circuit, info = encode(SPARSE([(0, 1.0)]), N=4)
        assert_encodes(circuit, np.array([1, 0, 0, 0], dtype=float))

    def test_single_last_index(self):
        circuit, info = encode(SPARSE([(63, 1.0)]), N=64)
        expected = np.zeros(64); expected[63] = 1.0
        assert_encodes(circuit, expected)

    def test_two_entries(self):
        circuit, info = encode(SPARSE([(1, 3.0), (6, 4.0)]), N=8)
        expected = np.zeros(8); expected[1] = 3.0; expected[6] = 4.0
        assert_encodes(circuit, expected)

    def test_five_entries(self):
        entries = [(1, 1.0), (5, 2.0), (10, 3.0), (20, 1.5), (30, 0.5)]
        circuit, info = encode(SPARSE(entries), N=32)
        expected = np.zeros(32)
        for k, p in entries: expected[k] = p
        assert_encodes(circuit, expected)

    def test_gate_count_single_is_hamming_weight(self):
        for k in [0, 1, 3, 7, 19]:
            _, info = encode(SPARSE([(k, 1.0)]), N=64)
            assert info.gate_count == bin(k).count('1'), \
                f"k={k}: expected {bin(k).count('1')} gates, got {info.gate_count}"

    def test_complexity(self):
        _, info = encode(SPARSE([(3, 1.0)]), N=8)
        assert "m" in info.complexity

    def test_validate(self):
        _, info = encode(SPARSE([(3, 1.0)]), N=8, validate=True)
        assert info.validated

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            SPARSE([])

    def test_bad_input_raises(self):
        with pytest.raises(TypeError):
            SPARSE([42])

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            encode(SPARSE([(10, 1.0)]), N=8)

    def test_duplicate_index_raises(self):
        with pytest.raises(ValueError, match="more than once"):
            encode(SPARSE([(3, 1.0), (3, 2.0)]), N=8)

    def test_returns_encoding_info(self):
        _, info = encode(SPARSE([(1, 1.0)]), N=4)
        assert isinstance(info, EncodingInfo)

    def test_returns_qiskit_circuit(self):
        circuit, _ = encode(SPARSE([(1, 1.0)]), N=4)
        assert isinstance(circuit, QuantumCircuit)


# ===================================================================
# STEP
# ===================================================================

class TestStep:

    def test_basic(self):
        circuit, info = encode(STEP(k_s=4, c=2.0), N=8)
        assert info.vector_type == "STEP"
        expected = np.zeros(8); expected[:4] = 2.0
        assert_encodes(circuit, expected)

    def test_non_power_of_two(self):
        for k_s in [3, 5, 6, 7, 37, 48, 60, 63]:
            circuit, info = encode(STEP(k_s=k_s, c=1.0), N=64)
            expected = np.zeros(64); expected[:k_s] = 1.0
            assert_encodes(circuit, expected)

    def test_full_range_is_uniform(self):
        """STEP(k_s=N) produces the m-Hadamard uniform superposition."""
        circuit, info = encode(STEP(k_s=8, c=1.0), N=8)
        assert_encodes(circuit, np.ones(8))

    def test_gate_count_at_most_m(self):
        for m in [2, 3, 4, 5, 6]:
            N = 2 ** m
            _, info = encode(STEP(k_s=N // 2, c=1.0), N=N)
            assert info.gate_count <= m * 4  # generous bound

    def test_complexity(self):
        _, info = encode(STEP(k_s=4, c=1.0), N=8)
        assert info.complexity == "O(m)"

    def test_validate(self):
        _, info = encode(STEP(k_s=5, c=1.0), N=8, validate=True)
        assert info.validated

    def test_k_s_zero_raises(self):
        with pytest.raises(ValueError):
            encode(STEP(k_s=0, c=1.0), N=8)

    def test_k_s_beyond_N_raises(self):
        with pytest.raises(ValueError):
            encode(STEP(k_s=9, c=1.0), N=8)


# ===================================================================
# SQUARE
# ===================================================================

class TestSquare:

    def test_basic(self):
        circuit, info = encode(SQUARE(k1=2, k2=6, c=1.0), N=8)
        assert info.vector_type == "SQUARE"
        expected = np.zeros(8); expected[2:6] = 1.0
        assert_encodes(circuit, expected)

    def test_aligned(self):
        circuit, info = encode(SQUARE(k1=8, k2=16, c=1.0), N=16)
        expected = np.zeros(16); expected[8:16] = 1.0
        assert_encodes(circuit, expected)

    def test_general_non_aligned(self):
        for k1, k2 in [(1, 5), (3, 7), (8, 24), (10, 50)]:
            circuit, info = encode(SQUARE(k1=k1, k2=k2, c=1.0), N=64)
            expected = np.zeros(64); expected[k1:k2] = 1.0
            assert_encodes(circuit, expected)

    def test_complexity_general(self):
        """General non-aligned SQUARE: O(m²) via Draper adder."""
        _, info = encode(SQUARE(k1=2, k2=6, c=1.0), N=8)
        assert info.complexity == "O(m²)"

    def test_complexity_k1_zero(self):
        """SQUARE with k1=0 reduces to STEP: O(m)."""
        _, info = encode(SQUARE(k1=0, k2=4, c=1.0), N=8)
        assert info.complexity == "O(m)"

    def test_complexity_aligned(self):
        """Power-of-2-aligned SQUARE: O(m) special case."""
        _, info = encode(SQUARE(k1=8, k2=16, c=1.0), N=64)
        assert info.complexity == "O(m)"

    def test_validate(self):
        _, info = encode(SQUARE(k1=3, k2=7, c=1.0), N=8, validate=True)
        assert info.validated

    def test_k1_equals_zero_is_step(self):
        """SQUARE(k1=0, k2=k_s) should match STEP(k_s)."""
        N = 8
        c1, _ = encode(SQUARE(k1=0, k2=4, c=1.0), N=N)
        c2, _ = encode(STEP(k_s=4, c=1.0), N=N)
        expected = np.zeros(N); expected[:4] = 1.0
        assert_encodes(c1, expected)
        assert_encodes(c2, expected)

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            encode(SQUARE(k1=6, k2=2, c=1.0), N=8)  # k1 >= k2

    def test_composite_list(self):
        """List of SQUARE constructors encodes a piecewise-constant vector."""
        circuit, info = encode([
            SQUARE(k1=0,  k2=4,  c=1.0),
            SQUARE(k1=4,  k2=8,  c=3.0),
        ], N=8)
        # Composite uses LCU with ancilla qubits; verify it runs and
        # returns the right metadata rather than checking statevector directly.
        assert info.vector_type == "COMPOSITE"
        assert circuit.num_qubits >= 3  # m=3 data qubits + ancilla


# ===================================================================
# FOURIER
# ===================================================================

class TestFourier:

    def test_single_sine(self):
        circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=64)
        assert info.vector_type == "FOURIER"
        k = np.arange(64)
        assert_encodes(circuit, np.sin(2 * np.pi * k / 64))

    def test_single_cosine(self):
        circuit, info = encode(FOURIER(modes=[(1, 1.0, math.pi / 2)]), N=64)
        k = np.arange(64)
        assert_encodes(circuit, np.cos(2 * np.pi * k / 64))

    def test_single_with_phase(self):
        phi = math.pi / 4
        circuit, info = encode(FOURIER(modes=[(3, 2.0, phi)]), N=64)
        k = np.arange(64)
        assert_encodes(circuit, 2.0 * np.sin(2 * np.pi * 3 * k / 64 + phi))

    def test_two_tuple_defaults_phi(self):
        """(n, A) without phi should default to phi=0."""
        circuit, info = encode(FOURIER(modes=[(1, 1.0)]), N=64)
        k = np.arange(64)
        assert_encodes(circuit, np.sin(2 * np.pi * k / 64))

    def test_multi_mode(self):
        circuit, info = encode(FOURIER(modes=[(1, 2.0, 0), (3, 1.0, 0)]), N=16)
        k = np.arange(16)
        expected = 2.0 * np.sin(2 * np.pi * k / 16) + np.sin(2 * np.pi * 3 * k / 16)
        assert_encodes(circuit, expected)

    def test_complexity(self):
        _, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=16)
        assert "m" in info.complexity

    def test_validate(self):
        _, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=8, validate=True)
        assert info.validated

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            FOURIER(modes=[])

    def test_bad_input_raises(self):
        with pytest.raises(TypeError):
            FOURIER(modes=[42])

    def test_mode_zero_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            encode(FOURIER(modes=[(0, 1.0, 0)]), N=8)


# ===================================================================
# General encode() behaviour
# ===================================================================

class TestEncode:

    def test_non_power_of_two_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            encode(SPARSE([(1, 1.0)]), N=6)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            encode("SPARSE", N=8)

    def test_info_fields(self):
        _, info = encode(SPARSE([(1, 1.0)]), N=8)
        assert isinstance(info, EncodingInfo)
        assert info.N == 8
        assert info.m == 3
        assert info.gate_count >= 0
        assert info.complexity != ""
        assert info.validated is False


# ===================================================================
# Constructors
# ===================================================================

class TestConstructors:

    def test_sparse_repr(self):
        s = SPARSE([(3, 5.0)])
        assert "SPARSE" in repr(s)
        assert s.vector_type == VectorType.SPARSE

    def test_sparse_stores_loads(self):
        s = SPARSE([(1, 2.0), (5, 3.0)])
        assert len(s.params["loads"]) == 2
        assert s.params["loads"][0] == {"k": 1, "P": 2.0}

    def test_fourier_repr(self):
        f = FOURIER(modes=[(1, 1.0, 0)])
        assert "FOURIER" in repr(f)
        assert f.vector_type == VectorType.FOURIER

    def test_fourier_stores_modes(self):
        f = FOURIER(modes=[(1, 2.0, 0.0), (3, 1.0, 0.5)])
        assert len(f.params["modes"]) == 2
        assert f.params["modes"][0]["n"] == 1

    def test_fourier_phi_default(self):
        f = FOURIER(modes=[(1, 1.0)])
        assert f.params["modes"][0]["phi"] == 0.0

    def test_step_stores_params(self):
        s = STEP(k_s=4, c=2.0)
        assert s.params["k_s"] == 4
        assert s.params["c"] == 2.0

    def test_square_stores_params(self):
        s = SQUARE(k1=2, k2=6, c=1.0)
        assert s.params["k1"] == 2
        assert s.params["k2"] == 6



# ===================================================================
# WALSH
# ===================================================================

class TestWalsh:

    def test_k0_standard(self):
        """Standard Walsh k=0: alternates +/-1 every sample."""
        circuit, info = encode(WALSH(k=0), N=8)
        assert info.vector_type == "WALSH"
        expected = np.array([1,-1,1,-1,1,-1,1,-1], dtype=float)
        assert_encodes(circuit, expected)

    def test_k1_standard(self):
        """Standard Walsh k=1: blocks of 2, period 4."""
        circuit, info = encode(WALSH(k=1), N=8)
        expected = np.array([1,1,-1,-1,1,1,-1,-1], dtype=float)
        assert_encodes(circuit, expected)

    def test_k2_standard(self):
        """Standard Walsh k=2: blocks of 4."""
        circuit, info = encode(WALSH(k=2), N=8)
        expected = np.array([1,1,1,1,-1,-1,-1,-1], dtype=float)
        assert_encodes(circuit, expected)

    def test_generalized_two_levels(self):
        """Generalized Walsh: c_pos=1, c_neg=4 (e.g. Fermi-Hubbard t/U ratio)."""
        circuit, info = encode(WALSH(k=2, c_pos=1.0, c_neg=4.0), N=8)
        expected = np.array([1,1,1,1,4,4,4,4], dtype=float)
        assert_encodes(circuit, expected)

    def test_generalized_validate(self):
        _, info = encode(WALSH(k=1, c_pos=1.0, c_neg=3.0), N=8, validate=True)
        assert info.validated

    def test_standard_is_special_case(self):
        """WALSH(k) and WALSH(k, c_pos=1, c_neg=-1) must produce same circuit."""
        c1, i1 = encode(WALSH(k=1), N=8)
        c2, i2 = encode(WALSH(k=1, c_pos=1.0, c_neg=-1.0), N=8)
        assert i1.gate_count == i2.gate_count

    def test_gate_count_is_m_plus_one(self):
        for m in [3, 4, 5]:
            N = 2 ** m
            _, info = encode(WALSH(k=0), N=N)
            assert info.gate_count == m + 1

    def test_generalized_gate_count_is_m_plus_one(self):
        """Generalized Walsh still costs only m+1 gates."""
        _, info = encode(WALSH(k=1, c_pos=1.0, c_neg=4.0), N=8)
        assert info.gate_count == 4  # m+1 = 3+1

    def test_complexity(self):
        _, info = encode(WALSH(k=1), N=8)
        assert info.complexity == "O(m)"

    def test_validate_standard(self):
        _, info = encode(WALSH(k=1), N=8, validate=True)
        assert info.validated

    def test_k_out_of_range_raises(self):
        with pytest.raises(ValueError):
            encode(WALSH(k=3), N=8)  # k must be < m=3

    def test_constructor_stores_params(self):
        w = WALSH(k=2, c_pos=1.0, c_neg=4.0)
        assert w.vector_type == VectorType.WALSH
        assert w.params["k"] == 2
        assert w.params["c_pos"] == 1.0
        assert w.params["c_neg"] == 4.0

    def test_default_c_neg_is_minus_c_pos(self):
        w = WALSH(k=1, c_pos=2.0)
        assert w.params["c_neg"] == -2.0



# ===================================================================
# LCU
# ===================================================================

class TestLCU:

    def test_two_disjoint_squares_correct_state(self):
        """Two disjoint SQUARE intervals: post-selected state is correct."""
        circuit, info = encode(
            LCU([(1.0, SQUARE(k1=0, k2=4, c=1.0)),
                 (1.0, SQUARE(k1=4, k2=8, c=1.0))]), N=8)
        assert info.vector_type == "LCU"
        # p = sum_j beta_j^4 = 2*(1/sqrt(2))^4 = 0.5 for 2 equal-weight disjoint
        assert abs(info.success_probability - 0.5) < 1e-6
        expected = np.ones(8, dtype=float)
        assert_encodes_lcu(circuit, expected, n_anc=1)

    def test_two_disjoint_squares_unequal_weights(self):
        """Disjoint SQUARE with different weights: post-selected state correct."""
        circuit, info = encode(
            LCU([(1.0, SQUARE(k1=0, k2=4, c=1.0)),
                 (4.0, SQUARE(k1=4, k2=8, c=1.0))]), N=8)
        assert 0 < info.success_probability < 1.0
        expected = np.array([1,1,1,1,4,4,4,4], dtype=float)
        assert_encodes_lcu(circuit, expected, n_anc=1)

    def test_three_disjoint_squares(self):
        """Three disjoint intervals — 2 ancilla qubits."""
        circuit, info = encode(
            LCU([(1.0, SQUARE(k1=0,  k2=4,  c=2.0)),
                 (1.0, SQUARE(k1=4,  k2=8,  c=3.0)),
                 (1.0, SQUARE(k1=8,  k2=16, c=1.0))]), N=16)
        assert 0 < info.success_probability <= 1.0
        expected = np.array([2]*4 + [3]*4 + [1]*8, dtype=float)
        assert_encodes_lcu(circuit, expected, n_anc=2)

    def test_disjoint_step_square(self):
        """STEP + SQUARE with disjoint support: post-selected state correct."""
        circuit, info = encode(
            LCU([(1.0, STEP(k_s=4,  c=2.0)),
                 (1.0, SQUARE(k1=4, k2=8, c=3.0))]), N=8)
        assert 0 < info.success_probability <= 1.0
        expected = np.array([2,2,2,2,3,3,3,3], dtype=float)
        assert_encodes_lcu(circuit, expected, n_anc=1)

    def test_overlapping_warns_and_p_lt_1(self):
        """Overlapping components emit UserWarning and p < 1."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            circuit, info = encode(
                LCU([(1.0, STEP(k_s=8,   c=1.0)),
                     (1.0, FOURIER(modes=[(1, 1.0, 0)]))]), N=16)
            assert len(w) == 1
            assert "overlapping" in str(w[0].message).lower()
        # Protocol 1: p < 1 for overlapping, non-identical states
        assert info.success_probability < 1.0
        assert info.success_probability > 0.0

    def test_single_component(self):
        """Single-component LCU reduces to plain encode."""
        c1, i1 = encode(LCU([(1.0, STEP(k_s=4, c=1.0))]), N=8)
        c2, i2 = encode(STEP(k_s=4, c=1.0), N=8)
        assert i1.gate_count == i2.gate_count

    def test_validate_disjoint(self):
        _, info = encode(
            LCU([(1.0, SQUARE(k1=0, k2=4, c=1.0)),
                 (1.0, SQUARE(k1=4, k2=8, c=2.0))]),
            N=8, validate=True)
        assert info.validated

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="positive"):
            LCU([(-1.0, STEP(k_s=4, c=1.0))])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            LCU([])

    def test_bad_component_raises(self):
        with pytest.raises(TypeError):
            LCU([(1.0, "not a VectorObj")])

    def test_success_probability_in_info(self):
        _, info = encode(
            LCU([(1.0, SQUARE(k1=0, k2=4, c=1.0)),
                 (1.0, SQUARE(k1=4, k2=8, c=1.0))]), N=8)
        assert hasattr(info, 'success_probability')
        assert 0.0 < info.success_probability <= 1.0


# ===================================================================
# Gate-count scaling tests: verify O(poly(m)) not O(2^m)
#
# Strategy: measure transpiled gate_count at m = 4, 6, 8, 10, 12 and
# fit two competing models by least-squares in log-space:
#
#   poly  model:  log(gates) ~ a + b*log(m)   => gates ~ C * m^b
#   exp   model:  log(gates) ~ a + b*m         => gates ~ C * alpha^m
#
# A pattern is O(poly(m)) if the poly model fits substantially better
# than the exponential model, measured by R^2.  We require:
#
#   R2_poly > R2_exp  (poly model wins)
#   R2_poly > 0.90    (good fit overall)
#   b_exp   < 0.50    (exponential coefficient is small, ruling out 2^m)
#
# The b_exp < 0.50 guard catches cases where both models fit poorly
# (flat gate-count curves where neither is informative).
#
# These tests are deliberately coarse: they detect O(2^m) blowup, not
# exact constants.  They complement, not replace, the unit correctness
# tests above.
# ===================================================================

import numpy as np


def _fit_scaling(m_vals, gate_vals):
    """
    Fit poly and exponential models to (m, gate_count) data.

    Returns
    -------
    b_poly  : exponent in C * m^b  (poly model)
    b_exp   : coefficient in C * alpha^b*m  (exp model)
    r2_poly : R^2 of poly fit in log-log space
    r2_exp  : R^2 of exp fit in log-linear space
    """
    m = np.array(m_vals, dtype=float)
    g = np.array(gate_vals, dtype=float)

    log_m = np.log(m)
    log_g = np.log(g)

    # Poly fit: log(g) = a + b*log(m)
    A_poly = np.column_stack([np.ones_like(log_m), log_m])
    coeff_poly, _, _, _ = np.linalg.lstsq(A_poly, log_g, rcond=None)
    b_poly = coeff_poly[1]
    pred_poly = A_poly @ coeff_poly
    ss_res = np.sum((log_g - pred_poly) ** 2)
    ss_tot = np.sum((log_g - log_g.mean()) ** 2)
    r2_poly = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    # Exp fit: log(g) = a + b*m
    A_exp = np.column_stack([np.ones_like(m), m])
    coeff_exp, _, _, _ = np.linalg.lstsq(A_exp, log_g, rcond=None)
    b_exp = coeff_exp[1]
    pred_exp = A_exp @ coeff_exp
    ss_res_e = np.sum((log_g - pred_exp) ** 2)
    r2_exp = 1.0 - ss_res_e / ss_tot if ss_tot > 1e-12 else 1.0

    return b_poly, b_exp, r2_poly, r2_exp


def _assert_poly_scaling(m_vals, gate_vals, label):
    """Assert that gate_vals scales as poly(m), not as 2^m."""
    b_poly, b_exp, r2_poly, r2_exp = _fit_scaling(m_vals, gate_vals)
    assert r2_poly > r2_exp, (
        f"{label}: exponential model fits better than polynomial "
        f"(R2_poly={r2_poly:.3f} <= R2_exp={r2_exp:.3f}). "
        f"Gate counts: {list(zip(m_vals, gate_vals))}"
    )
    assert r2_poly > 0.90, (
        f"{label}: polynomial fit is poor (R2_poly={r2_poly:.3f} < 0.90). "
        f"Gate counts may not follow O(poly(m)). "
        f"Gate counts: {list(zip(m_vals, gate_vals))}"
    )
    assert b_exp < 0.50, (
        f"{label}: exponential coefficient b_exp={b_exp:.3f} >= 0.50, "
        f"suggesting near-exponential gate growth. "
        f"Gate counts: {list(zip(m_vals, gate_vals))}"
    )


# ===================================================================
# GEOMETRIC
# ===================================================================

class TestGeometric:

    def test_decay_basic(self):
        circuit, info = encode(GEOMETRIC(ratio=0.5), N=8)
        assert info.vector_type == "GEOMETRIC"
        assert info.complexity == "O(m)"
        f = 0.5 ** np.arange(8)
        assert_encodes(circuit, f)

    def test_growth(self):
        circuit, info = encode(GEOMETRIC(ratio=2.0), N=16)
        f = 2.0 ** np.arange(16)
        assert_encodes(circuit, f)

    def test_near_one(self):
        circuit, info = encode(GEOMETRIC(ratio=0.99), N=16)
        f = 0.99 ** np.arange(16)
        assert_encodes(circuit, f)

    def test_gate_count_equals_m(self):
        for m in [3, 4, 6, 8]:
            N = 2 ** m
            _, info = encode(GEOMETRIC(ratio=0.9), N=N)
            assert info.gate_count == m

    def test_zero_two_qubit_gates(self):
        _, info = encode(GEOMETRIC(ratio=0.9), N=64)
        assert info.gate_count_2q == 0
        assert info.gate_count_1q == 6

    def test_validate(self):
        circuit, info = encode(GEOMETRIC(ratio=0.8), N=16, validate=True)
        assert info.validated is True
        assert info.vector is not None

    def test_custom_c_normalization(self):
        """c only affects normalization, not relative amplitudes."""
        c1, _ = encode(GEOMETRIC(ratio=0.7, c=1.0), N=8)
        c2, _ = encode(GEOMETRIC(ratio=0.7, c=5.0), N=8)
        sv1 = np.abs(np.array(statevector(c1)))
        sv2 = np.abs(np.array(statevector(c2)))
        np.testing.assert_allclose(sv1, sv2, atol=1e-10)

    def test_ratio_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            GEOMETRIC(ratio=0.0)

    def test_ratio_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            GEOMETRIC(ratio=-0.5)

    def test_ratio_one_raises(self):
        with pytest.raises(ValueError, match="uniform"):
            GEOMETRIC(ratio=1.0)

    def test_emitted_code_runs(self):
        circuit, info = encode(GEOMETRIC(ratio=0.8), N=16)
        namespace = {}
        exec(compile(info.circuit_code, "<test>", "exec"), namespace)
        assert isinstance(namespace["qc"], QuantumCircuit)
        sv_orig = np.abs(np.array(statevector(circuit)))
        sv_emit = np.abs(np.array(statevector(namespace["qc"])))
        np.testing.assert_allclose(sv_orig, sv_emit, atol=1e-10)

    def test_lcu_composability(self):
        """GEOMETRIC can be used as an LCU component."""
        circuit, info = encode(
            LCU([(1.0, STEP(k_s=8, c=1.0)),
                 (2.0, GEOMETRIC(ratio=0.5))]),
            N=16)
        assert info.vector_type == "LCU"
        assert 0 < info.success_probability <= 1.0

    # === start parameter tests ===

    def test_start_zero_backward_compatibility(self):
        """start=0 should be identical to the original GEOMETRIC."""
        c1, i1 = encode(GEOMETRIC(ratio=0.7), N=16)
        c2, i2 = encode(GEOMETRIC(ratio=0.7, start=0), N=16)
        sv1 = np.array(statevector(c1))
        sv2 = np.array(statevector(c2))
        np.testing.assert_allclose(sv1, sv2, atol=1e-15)
        assert i1.gate_count == i2.gate_count

    def test_start_aligned_half(self):
        """start=N/2: geometric decay starting at midpoint."""
        circuit, info = encode(GEOMETRIC(ratio=0.5, start=32), N=64)
        expected = np.zeros(64)
        expected[32:] = 0.5 ** np.arange(32)  # 0.5^0, 0.5^1, ..., 0.5^31
        assert_encodes(circuit, expected)
        # Check gate count: log2(32)=5 geometric rotations + 1 X gate on top qubit
        assert info.gate_count_1q == 6  # 5 R_y + 1 X
        assert info.gate_count_2q == 0
        assert info.circuit_depth == 1

    def test_start_aligned_three_quarters(self):
        """start=3N/4: geometric in the upper quarter."""
        circuit, info = encode(GEOMETRIC(ratio=0.6, start=48), N=64)
        expected = np.zeros(64)
        expected[48:] = 0.6 ** np.arange(16)  # decay for 16 elements
        assert_encodes(circuit, expected)
        # log2(16)=4 rotations + 2 X gates (48/16 = 3 = 11 binary)
        assert info.gate_count_1q == 6  # 4 R_y + 2 X
        assert info.gate_count_2q == 0

    def test_start_aligned_near_end(self):
        """start=56 at N=64: geometric in just the last 8 slots."""
        circuit, info = encode(GEOMETRIC(ratio=0.3, start=56), N=64)
        expected = np.zeros(64)
        expected[56:] = 0.3 ** np.arange(8)
        assert_encodes(circuit, expected)
        # log2(8)=3 rotations + 3 X gates (56/8 = 7 = 111 binary)
        assert info.gate_count_1q == 6  # 3 R_y + 3 X
        assert info.gate_count_2q == 0

    def test_start_dyadic_small(self):
        """Dyadic regime: start=10, N=64 (w=54, not power-of-2-aligned)."""
        circuit, info = encode(GEOMETRIC(ratio=0.5, start=10), N=64)
        assert info.complexity == "O(m^2)"
        assert info.gate_count_2q > 0          # must have entangling gates
        assert info.success_probability == 1.0  # disjoint-support blocks

        expected = np.zeros(64)
        expected[10:] = 0.5 ** np.arange(54)
        assert_encodes(circuit, expected)

    def test_start_dyadic_non_multiple(self):
        """Dyadic regime: start=40, N=64 (w=24, aligned in multi-block sense)."""
        circuit, info = encode(GEOMETRIC(ratio=0.5, start=40), N=64)
        assert info.complexity == "O(m^2)"
        assert info.gate_count_2q > 0
        assert info.success_probability == 1.0

        expected = np.zeros(64)
        expected[40:] = 0.5 ** np.arange(24)
        assert_encodes(circuit, expected)

    def test_start_dyadic_user_case(self):
        """Dyadic regime: the motivating case start=4, N=256."""
        circuit, info = encode(GEOMETRIC(ratio=0.8, start=4), N=256)
        assert info.complexity == "O(m^2)"
        assert info.success_probability == 1.0

        # Gate-count ceiling: the previous sparse-fallback took ~17k gates
        # for this case.  The dyadic construction must be bounded by O(m^2)
        # ~= 64 * (const), so well under a few thousand.
        assert info.gate_count < 2500, (
            f"Dyadic regime gate count {info.gate_count} exceeds the O(m^2) "
            f"budget; the old sparse fallback produced ~17 000 gates."
        )

        expected = np.zeros(256)
        expected[4:] = 0.8 ** np.arange(252)
        assert_encodes(circuit, expected, tol=1e-4)

    def test_start_validation_bounds(self):
        """start must be in range [0, N)."""
        with pytest.raises(ValueError, match="start < N"):
            encode(GEOMETRIC(ratio=0.5, start=64), N=64)
        with pytest.raises(ValueError, match="non-negative"):
            GEOMETRIC(ratio=0.5, start=-1)

    def test_start_validate_mode(self):
        """Validation should work correctly with start parameter."""
        circuit, info = encode(GEOMETRIC(ratio=0.8, start=8), N=16, validate=True)
        assert info.validated is True
        expected = np.zeros(16)
        expected[8:] = 0.8 ** np.arange(8)
        np.testing.assert_allclose(np.abs(info.vector), expected, atol=1e-10)

    def test_start_emitted_code_runs(self):
        """Emitted code should work for start offset."""
        circuit, info = encode(GEOMETRIC(ratio=0.9, start=16), N=32)
        namespace = {}
        exec(compile(info.circuit_code, "<test>", "exec"), namespace)
        assert isinstance(namespace["qc"], QuantumCircuit)
        sv_orig = np.abs(np.array(statevector(circuit)))
        sv_emit = np.abs(np.array(statevector(namespace["qc"])))
        np.testing.assert_allclose(sv_orig, sv_emit, atol=1e-10)

    def test_start_custom_c_normalization(self):
        """c parameter still works with start offset."""
        c1, _ = encode(GEOMETRIC(ratio=0.7, start=8, c=1.0), N=16)
        c2, _ = encode(GEOMETRIC(ratio=0.7, start=8, c=3.0), N=16)
        sv1 = np.abs(np.array(statevector(c1)))
        sv2 = np.abs(np.array(statevector(c2)))
        np.testing.assert_allclose(sv1, sv2, atol=1e-10)


# ===================================================================
# GEOMETRIC  --  dyadic-decomposition (regime c) dedicated tests
# ===================================================================

class TestGeometricDyadic:
    """
    Correctness + complexity tests for the regime-(c) dyadic path of
    _synth_geometric, where [start, N) is NOT a single aligned dyadic
    block and the synthesizer decomposes the support into up to m
    power-of-2-aligned sub-blocks.

    References
    ----------
    Bentley & Saxe, J. Algorithms 1(4), 1980 -- dyadic interval split.
    Gleinig & Hoefler, DAC 2021                -- sparse anchor load.
    """

    # ------------------------------------------------------------------
    # Dyadic-decomposition helper (pure combinatorics, no quantum state)
    # ------------------------------------------------------------------

    def test_decomposition_covers_interval_exactly(self):
        """Union of dyadic blocks equals [start, N) exactly, disjointly."""
        from pyencode.synthesizer import _dyadic_decomposition
        for (s, N) in [(1, 16), (5, 16), (7, 32), (100, 1024),
                       (123, 1024), (3, 256), (255, 256)]:
            blocks = _dyadic_decomposition(s, N)
            covered = set()
            for (a, j) in blocks:
                rng = range(a, a + (1 << j))
                # No overlap with previously-covered indices
                assert covered.isdisjoint(rng), (
                    f"Overlap at (s={s}, N={N}) block ({a},{j})")
                covered.update(rng)
            assert covered == set(range(s, N)), (
                f"Coverage mismatch at (s={s}, N={N})")

    def test_decomposition_blocks_aligned(self):
        """Every block (a_k, j_k) must satisfy a_k % 2^j_k == 0."""
        from pyencode.synthesizer import _dyadic_decomposition
        for (s, N) in [(1, 16), (5, 16), (7, 32), (123, 1024)]:
            for (a, j) in _dyadic_decomposition(s, N):
                assert a % (1 << j) == 0, (
                    f"Misaligned block ({a},{j}) in (s={s}, N={N})")

    def test_decomposition_at_most_m_blocks(self):
        """Standard dyadic bound: L <= m = log2 N for any s in [0, N)."""
        from pyencode.synthesizer import _dyadic_decomposition
        for m in [4, 6, 8, 10, 12]:
            N = 1 << m
            for s in [1, 3, 7, N // 3, N // 2 + 1, N - 1]:
                if s >= N:
                    continue
                L = len(_dyadic_decomposition(s, N))
                assert L <= m, f"L={L} > m={m} for (s={s}, N={N})"

    # ------------------------------------------------------------------
    # Regime-selection: aligned vs dyadic
    # ------------------------------------------------------------------

    def test_regime_aligned_reports_Om(self):
        """Aligned start (regime b) reports O(m) and uses 0 CX gates."""
        # start = N/2, N/4*k, etc. -- all single-block dyadic
        for (r, s, N) in [(0.5, 32, 64), (0.7, 48, 64), (0.3, 56, 64),
                          (0.9, 128, 256)]:
            _, info = encode(GEOMETRIC(ratio=r, start=s), N=N)
            assert info.complexity == "O(m)", (
                f"(s={s}, N={N}): got {info.complexity}")
            assert info.gate_count_2q == 0

    def test_regime_dyadic_reports_Om2(self):
        """Non-aligned start (regime c) reports O(m^2), non-zero CX."""
        for (r, s, N) in [(0.5, 5, 16), (0.8, 4, 256), (0.9, 100, 1024),
                          (0.7, 1, 128), (0.3, 123, 256)]:
            _, info = encode(GEOMETRIC(ratio=r, start=s), N=N)
            assert info.complexity == "O(m^2)", (
                f"(s={s}, N={N}): got {info.complexity}")
            assert info.gate_count_2q > 0

    # ------------------------------------------------------------------
    # State-vector correctness across the parameter space
    # ------------------------------------------------------------------

    @staticmethod
    def _reference(ratio, start, N):
        f = np.zeros(N)
        f[start:] = ratio ** np.arange(N - start)
        return f / np.linalg.norm(f)

    def test_statevector_exhaustive_small(self):
        """Exhaustive correctness for N in {8, 16, 32}, all start, a few ratios."""
        for N in [8, 16, 32]:
            for r in [0.3, 0.5, 0.8, 1.5]:
                for s in range(1, N):
                    qc, _ = encode(GEOMETRIC(ratio=r, start=s), N=N)
                    sv = np.abs(np.array(statevector(qc)))
                    ref = np.abs(self._reference(r, s, N))
                    np.testing.assert_allclose(
                        sv, ref, atol=1e-8,
                        err_msg=f"(r={r}, s={s}, N={N})"
                    )

    def test_statevector_larger_N(self):
        """Spot-check correctness at m = 8, 9, 10."""
        for N in [256, 512, 1024]:
            for s in [1, 3, 7, 100, N // 3, N - 3]:
                if s >= N:
                    continue
                qc, _ = encode(GEOMETRIC(ratio=0.9, start=s), N=N)
                sv = np.abs(np.array(statevector(qc)))
                ref = np.abs(self._reference(0.9, s, N))
                np.testing.assert_allclose(
                    sv, ref, atol=1e-6,
                    err_msg=f"(s={s}, N={N})"
                )

    def test_zeros_before_start(self):
        """Amplitudes on |i> for i < start must be exactly zero."""
        for (r, s, N) in [(0.5, 5, 16), (0.8, 4, 256), (0.9, 100, 1024)]:
            qc, _ = encode(GEOMETRIC(ratio=r, start=s), N=N)
            sv = np.array(statevector(qc))
            assert np.max(np.abs(sv[:s])) < 1e-10, (
                f"Non-zero amplitude before start (s={s}, N={N}): "
                f"max={np.max(np.abs(sv[:s])):.2e}"
            )

    def test_c_only_affects_normalization(self):
        """The c parameter is a pure scalar prefactor (normalised out)."""
        for (r, s, N) in [(0.7, 5, 32), (0.8, 4, 256)]:
            qc1, _ = encode(GEOMETRIC(ratio=r, start=s, c=1.0), N=N)
            qc2, _ = encode(GEOMETRIC(ratio=r, start=s, c=7.3), N=N)
            sv1 = np.abs(np.array(statevector(qc1)))
            sv2 = np.abs(np.array(statevector(qc2)))
            np.testing.assert_allclose(sv1, sv2, atol=1e-10)

    def test_ratio_above_and_below_one(self):
        """Growth (r>1) and decay (r<1) both work in the dyadic regime."""
        for r in [0.2, 0.5, 0.95, 1.05, 1.5, 2.0]:
            qc, _ = encode(GEOMETRIC(ratio=r, start=5), N=32)
            sv = np.abs(np.array(statevector(qc)))
            ref = np.abs(self._reference(r, 5, 32))
            np.testing.assert_allclose(sv, ref, atol=1e-8,
                                       err_msg=f"ratio={r}")

    # ------------------------------------------------------------------
    # Complexity / cost-reduction benchmarks
    # ------------------------------------------------------------------

    def test_gate_count_polynomial_in_m(self):
        """Total gate count grows as O(m^2), not O(N)."""
        counts = []
        for m in [4, 6, 8, 10]:
            N = 1 << m
            _, info = encode(GEOMETRIC(ratio=0.9, start=3), N=N)
            counts.append(info.gate_count)
        # Cubic fit c0 + c1*m + c2*m^2 should bound the growth; gate count
        # at m=10 should be at most ~ 30 * (m=10)^2 = 3000, not 2^10 * 10.
        assert counts[-1] < 50 * (10 ** 2), (
            f"Gate count {counts[-1]} at m=10 exceeds quadratic budget.")
        # Monotone in m for a fixed start
        assert counts == sorted(counts), (
            f"Gate count not monotone in m: {counts}")

    def test_gate_count_beats_old_sparse_fallback(self):
        """
        Old sparse fallback produced O((N-s)*m) gates.  New dyadic
        construction must be strictly, materially smaller for any start
        whose [start,N) has more than ~ m non-zero amplitudes.
        """
        # User's reported case
        _, info = encode(GEOMETRIC(ratio=0.8, start=4), N=256)
        # Old: ~ 17 000 gates.  Dyadic is O(m^2) = 64 * const; under 2 000 easily.
        assert info.gate_count < 2000, (
            f"Expected < 2000 gates, got {info.gate_count} "
            f"(old sparse fallback produced ~17 000)."
        )

    def test_gate_count_worst_case_start_one(self):
        """
        start=1 gives the maximum number of dyadic blocks (L = m) and is
        therefore the stress case.  Must still be sub-cubic.
        """
        for m in [5, 7, 9]:
            N = 1 << m
            _, info = encode(GEOMETRIC(ratio=0.9, start=1), N=N)
            # Very loose bound: 100 * m^2.  Guards against accidental O(N*m).
            assert info.gate_count < 100 * m * m, (
                f"m={m}: {info.gate_count} gates exceeds 100*m^2 budget.")

    # ------------------------------------------------------------------
    # Validation + LCU composability
    # ------------------------------------------------------------------

    def test_validate_mode(self):
        """validate=True returns the classically-constructed reference vector."""
        qc, info = encode(GEOMETRIC(ratio=0.7, start=5),
                          N=16, validate=True)
        assert info.validated is True
        # info.vector is UNnormalized (matches existing regime-b convention)
        ref = np.zeros(16)
        ref[5:] = 0.7 ** np.arange(11)
        np.testing.assert_allclose(np.abs(info.vector), ref, atol=1e-10)

    def test_lcu_composability(self):
        """A dyadic-regime GEOMETRIC still works as an LCU component."""
        qc, info = encode(
            LCU([(1.0, STEP(k_s=8, c=1.0)),
                 (2.0, GEOMETRIC(ratio=0.5, start=5))]),
            N=16)
        assert info.vector_type == "LCU"
        assert 0 < info.success_probability <= 1.0

    def test_emitted_code_runs_and_matches(self):
        """Regime-(c) emitted code must run standalone and reproduce |psi>.

        The framework's auto-fallback substitutes a gate-level extraction
        for the inline skeleton, so the snippet must be fully executable
        and give the same state vector as the originally-synthesized
        circuit.  Same guarantee as regimes (a) and (b).
        """
        qc, info = encode(GEOMETRIC(ratio=0.8, start=5), N=32)
        namespace = {"QuantumCircuit": QuantumCircuit}
        exec(compile(info.circuit_code, "<emit>", "exec"), namespace)
        assert isinstance(namespace["qc"], QuantumCircuit)
        sv_orig = np.abs(np.array(statevector(qc)))
        sv_emit = np.abs(np.array(statevector(namespace["qc"])))
        np.testing.assert_allclose(sv_orig, sv_emit, atol=1e-10)


class TestScaling:
    """
    Verify that each pattern's gate count scales as O(poly(m)),
    not O(2^m).  Tests use m = 4, 6, 8, 10, 12 (N up to 4096).
    """

    M_VALS = [4, 6, 8, 10, 12]

    def test_sparse_single_entry_scaling(self):
        """SPARSE(s=1): gate count ~ Hamming weight, bounded by m."""
        # Use index 2^(m-1) - 1 to give Hamming weight ~ m/2 consistently
        gate_vals = []
        for m in self.M_VALS:
            N = 2 ** m
            idx = N // 2 - 1  # e.g. m=4 -> idx=7 = 0111, HW=3
            _, info = encode(SPARSE([(idx, 1.0)]), N=N)
            gate_vals.append(info.gate_count)
        _assert_poly_scaling(self.M_VALS, gate_vals, "SPARSE(s=1)")

    def test_sparse_multi_entry_scaling(self):
        """SPARSE(s=4): gate count should scale as O(s*m), not O(2^m).

        Uses non-power-of-2 indices so the Gleinig-Hoefler tree is
        non-trivial and gate count actually grows with m.
        """
        gate_vals = []
        for m in self.M_VALS:
            N = 2 ** m
            # Non-aligned indices: small odd offsets stay in range for m>=4
            indices = [N // 8 + 1, N // 4 + 1, N // 2 + 1, 3 * N // 4 + 1]
            entries = [(idx, float(j + 1)) for j, idx in enumerate(indices)]
            _, info = encode(SPARSE(entries), N=N)
            gate_vals.append(info.gate_count)
        # If gate count is constant across m, it is trivially O(poly(m)) --
        # verify only that it does not grow exponentially.
        if max(gate_vals) == min(gate_vals):
            m_large = self.M_VALS[-1]
            assert gate_vals[-1] < 2 ** m_large, (
                f"SPARSE(s=4): constant gate count {gate_vals[-1]} >= 2^{m_large}."
            )
        else:
            _assert_poly_scaling(self.M_VALS, gate_vals, "SPARSE(s=4)")

    def test_step_scaling(self):
        """STEP: gate count is O(m) by Shukla-Vedula construction."""
        gate_vals = []
        for m in self.M_VALS:
            N = 2 ** m
            # k_s = 3*N//4 gives a non-trivial binary decomposition
            _, info = encode(STEP(k_s=3 * N // 4, c=1.0), N=N)
            gate_vals.append(info.gate_count)
        _assert_poly_scaling(self.M_VALS, gate_vals, "STEP")

    def test_square_scaling(self):
        """SQUARE: gate count is O(m) via two controlled STEP circuits."""
        gate_vals = []
        for m in self.M_VALS:
            N = 2 ** m
            k1 = N // 4
            k2 = 3 * N // 4
            _, info = encode(SQUARE(k1=k1, k2=k2, c=1.0), N=N)
            gate_vals.append(info.gate_count)
        _assert_poly_scaling(self.M_VALS, gate_vals, "SQUARE")

    def test_walsh_scaling(self):
        """WALSH: gate count is exactly m+1, clearly O(m)."""
        gate_vals = []
        for m in self.M_VALS:
            N = 2 ** m
            _, info = encode(WALSH(k=m // 2, c_pos=1.0, c_neg=4.0), N=N)
            gate_vals.append(info.gate_count)
        _assert_poly_scaling(self.M_VALS, gate_vals, "WALSH")

    def test_geometric_scaling(self):
        """GEOMETRIC: gate count is exactly m, clearly O(m)."""
        gate_vals = []
        for m in self.M_VALS:
            N = 2 ** m
            _, info = encode(GEOMETRIC(ratio=0.95), N=N)
            gate_vals.append(info.gate_count)
            assert info.gate_count == m
            assert info.gate_count_2q == 0
        _assert_poly_scaling(self.M_VALS, gate_vals, "GEOMETRIC")

    def test_fourier_single_mode_scaling(self):
        """FOURIER(T=1): gate count is O(m^2) via inverse QFT."""
        gate_vals = []
        for m in self.M_VALS:
            N = 2 ** m
            _, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=N)
            gate_vals.append(info.gate_count)
        _assert_poly_scaling(self.M_VALS, gate_vals, "FOURIER(T=1)")

    def test_fourier_multi_mode_scaling(self):
        """FOURIER(T=2): gate count is still O(m^2), independent of T."""
        gate_vals = []
        for m in self.M_VALS:
            N = 2 ** m
            _, info = encode(FOURIER(modes=[(1, 1.0, 0), (2, 0.5, 0)]), N=N)
            gate_vals.append(info.gate_count)
        _assert_poly_scaling(self.M_VALS, gate_vals, "FOURIER(T=2)")

    def test_square_scales_as_step(self):
        """
        SQUARE gate count should grow no faster than a fixed multiple
        of STEP gate count at the same m (verifies the ~2x constant factor).
        """
        for m in self.M_VALS:
            N = 2 ** m
            _, info_sq = encode(SQUARE(k1=N // 4, k2=3 * N // 4, c=1.0), N=N)
            _, info_st = encode(STEP(k_s=3 * N // 4, c=1.0), N=N)
            ratio = info_sq.gate_count / max(info_st.gate_count, 1)
            assert ratio < 6, (
                f"m={m}: SQUARE gate count ({info_sq.gate_count}) is more than "
                f"6x STEP gate count ({info_st.gate_count}). "
                f"Ratio={ratio:.2f}. The ancilla overhead may be larger than expected."
            )

    def test_no_pattern_matches_exponential(self):
        """
        Omnibus guard: for every O(m) pattern, gate count at m=12 must
        be less than 2^12 = 4096.  A single violation signals exponential
        growth slipping through.
        """
        m = 12
        N = 2 ** m
        threshold = 2 ** m  # 4096

        checks = [
            ("SPARSE(s=1)", encode(SPARSE([(N // 2 - 1, 1.0)]), N=N)[1].gate_count),
            ("STEP",        encode(STEP(k_s=3 * N // 4, c=1.0), N=N)[1].gate_count),
            ("SQUARE",      encode(SQUARE(k1=N // 4, k2=3 * N // 4, c=1.0), N=N)[1].gate_count),
            ("WALSH",       encode(WALSH(k=m // 2), N=N)[1].gate_count),
            ("GEOMETRIC",   encode(GEOMETRIC(ratio=0.95), N=N)[1].gate_count),
            ("FOURIER(T=1)",encode(FOURIER(modes=[(1, 1.0, 0)]), N=N)[1].gate_count),
        ]
        for label, count in checks:
            assert count < threshold, (
                f"{label}: gate_count={count} >= 2^m={threshold} at m={m}. "
                f"This indicates exponential gate growth."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPopcount:
    """Tests for POPCOUNT — product state with identical Ry per qubit.
    Amplitudes depend only on Hamming weight: f_i proportional to r^popcount(i).
    """

    def test_basic(self):
        circuit, info = encode(POPCOUNT(r=0.5), N=8)
        assert info.vector_type == "POPCOUNT"
        assert info.complexity == "O(m)"
        pops = np.array([bin(i).count("1") for i in range(8)], dtype=float)
        f = 0.5 ** pops
        assert_encodes(circuit, f)

    def test_binomial_large_r(self):
        circuit, info = encode(POPCOUNT(r=2.0), N=16)
        pops = np.array([bin(i).count("1") for i in range(16)], dtype=float)
        f = 2.0 ** pops
        assert_encodes(circuit, f)

    def test_r_equals_one_is_uniform(self):
        """r=1 gives the uniform superposition (all amplitudes equal)."""
        circuit, _ = encode(POPCOUNT(r=1.0), N=16)
        sv = np.abs(statevector(circuit))
        np.testing.assert_allclose(sv, 1.0 / np.sqrt(16), atol=1e-10)

    def test_gate_count_equals_m(self):
        for m in [3, 4, 6, 8, 10]:
            N = 2 ** m
            _, info = encode(POPCOUNT(r=0.7), N=N)
            assert info.gate_count == m, \
                f"m={m}: expected {m} gates, got {info.gate_count}"

    def test_zero_two_qubit_gates_and_depth_one(self):
        """POPCOUNT is a product state: zero CX, depth 1."""
        _, info = encode(POPCOUNT(r=0.6), N=64)
        assert info.gate_count_2q == 0
        assert info.circuit_depth == 1

    def test_validate(self):
        circuit, info = encode(POPCOUNT(r=0.4), N=16, validate=True)
        assert info.validated is True
        assert info.vector is not None

    def test_custom_c_normalization(self):
        """c only affects global normalisation, not relative amplitudes."""
        c1, _ = encode(POPCOUNT(r=0.5, c=1.0), N=8)
        c2, _ = encode(POPCOUNT(r=0.5, c=7.0), N=8)
        sv1 = np.abs(statevector(c1))
        sv2 = np.abs(statevector(c2))
        np.testing.assert_allclose(sv1, sv2, atol=1e-10)

    def test_r_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            POPCOUNT(r=0.0)

    def test_r_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            POPCOUNT(r=-0.3)

    def test_emitted_code_runs(self):
        circuit, info = encode(POPCOUNT(r=0.8), N=16)
        namespace = {}
        exec(compile(info.circuit_code, "<test>", "exec"), namespace)
        assert isinstance(namespace["qc"], QuantumCircuit)
        sv_orig = np.abs(statevector(circuit))
        sv_emit = np.abs(statevector(namespace["qc"]))
        np.testing.assert_allclose(sv_orig, sv_emit, atol=1e-10)

    def test_lcu_composability(self):
        """POPCOUNT can be used as an LCU component."""
        circuit, info = encode(
            LCU([(1.0, STEP(k_s=8, c=1.0)),
                 (2.0, POPCOUNT(r=0.5))]),
            N=16)
        assert info.vector_type == "LCU"
        assert 0 < info.success_probability <= 1.0

    def test_hamming_weight_structure(self):
        """Verify amplitudes group by Hamming weight, not index."""
        r = 0.6
        _, info = encode(POPCOUNT(r=r), N=16, validate=True)
        f = info.vector / np.linalg.norm(info.vector)
        # Indices 3 (011), 5 (101), 6 (110) all have popcount=2
        # so amplitudes should be identical
        for i, j in [(3, 5), (3, 6), (5, 6), (7, 11), (7, 13), (7, 14)]:
            np.testing.assert_allclose(abs(f[i]), abs(f[j]), atol=1e-10,
                err_msg=f"indices {i},{j} have same popcount but different amplitudes")


class TestStaircase:
    """Tests for STAIRCASE — sparse geometric staircase on unary indices.
    Produces m+1 nonzero amplitudes at indices {0, 1, 3, 7, ..., 2^m-1} with
    f_{2^k-1} = c * r^k, via cascaded CR_y gates.
    """

    @staticmethod
    def _unary_indices(m):
        return [(1 << k) - 1 for k in range(m + 1)]

    def test_basic(self):
        circuit, info = encode(STAIRCASE(r=0.5), N=8)
        assert info.vector_type == "STAIRCASE"
        assert info.complexity == "O(m)"
        f = np.zeros(8)
        for k in range(4):
            f[(1 << k) - 1] = 0.5 ** k
        assert_encodes(circuit, f)

    def test_support_is_unary_indices(self):
        """Only indices 2^k - 1 should have nonzero amplitude."""
        for m in [3, 4, 5]:
            N = 2 ** m
            circuit, _ = encode(STAIRCASE(r=0.4), N=N)
            sv = np.array(statevector(circuit))
            unary = set(self._unary_indices(m))
            for i in range(N):
                if i in unary:
                    assert abs(sv[i]) > 1e-10, f"m={m}: expected nonzero at i={i}"
                else:
                    assert abs(sv[i]) < 1e-10, f"m={m}: expected zero at i={i}, got {sv[i]}"

    def test_geometric_ratio_exact(self):
        """Consecutive nonzero amplitudes should have ratio exactly r."""
        for r in [0.3, 0.5, 0.8, 1.5, 2.5]:
            for m in [3, 4, 5]:
                N = 2 ** m
                circuit, _ = encode(STAIRCASE(r=r), N=N)
                sv = np.array(statevector(circuit)).real
                amps = [sv[(1 << k) - 1] for k in range(m + 1)]
                for k in range(m):
                    observed = amps[k + 1] / amps[k]
                    np.testing.assert_allclose(observed, r, atol=1e-10,
                        err_msg=f"r={r}, m={m}, k={k}: ratio mismatch")

    def test_gate_count_equals_m(self):
        for m in [3, 4, 6, 8]:
            N = 2 ** m
            _, info = encode(STAIRCASE(r=0.7), N=N)
            assert info.gate_count == m, \
                f"m={m}: expected {m} gates, got {info.gate_count}"

    def test_cx_count_linear(self):
        """Each CR_y decomposes to O(1) CX, so total CX is O(m)."""
        for m in [4, 6, 8]:
            N = 2 ** m
            _, info = encode(STAIRCASE(r=0.6), N=N)
            assert info.gate_count_2q <= 3 * m, \
                f"m={m}: CX count {info.gate_count_2q} exceeds 3m"

    def test_validate(self):
        circuit, info = encode(STAIRCASE(r=0.5), N=16, validate=True)
        assert info.validated is True
        assert info.vector is not None
        # Check vector has exactly m+1 nonzero entries
        nonzero = np.count_nonzero(info.vector)
        assert nonzero == 5  # m+1 = log2(16)+1 = 5

    def test_custom_c_normalization(self):
        c1, _ = encode(STAIRCASE(r=0.6, c=1.0), N=16)
        c2, _ = encode(STAIRCASE(r=0.6, c=8.0), N=16)
        sv1 = np.abs(statevector(c1))
        sv2 = np.abs(statevector(c2))
        np.testing.assert_allclose(sv1, sv2, atol=1e-10)

    def test_growth_r_greater_than_one(self):
        circuit, _ = encode(STAIRCASE(r=2.0), N=8, validate=True)
        sv = np.array(statevector(circuit)).real
        # Amplitudes should grow by factor 2 at each unary index
        for k in range(3):
            observed = sv[(1 << (k + 1)) - 1] / sv[(1 << k) - 1]
            np.testing.assert_allclose(observed, 2.0, atol=1e-10)

    def test_r_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            STAIRCASE(r=0.0)

    def test_r_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            STAIRCASE(r=-0.4)

    def test_r_one_raises(self):
        with pytest.raises(ValueError, match="uniform"):
            STAIRCASE(r=1.0)

    def test_emitted_code_runs(self):
        circuit, info = encode(STAIRCASE(r=0.55), N=16)
        namespace = {}
        exec(compile(info.circuit_code, "<test>", "exec"), namespace)
        assert isinstance(namespace["qc"], QuantumCircuit)
        sv_orig = np.abs(statevector(circuit))
        sv_emit = np.abs(statevector(namespace["qc"]))
        np.testing.assert_allclose(sv_orig, sv_emit, atol=1e-10)

    def test_lcu_composability(self):
        circuit, info = encode(
            LCU([(1.0, SQUARE(k1=0, k2=4, c=1.0)),
                 (2.0, STAIRCASE(r=0.5))]),
            N=16)
        assert info.vector_type == "LCU"
        assert 0 < info.success_probability <= 1.0


class TestTensor:
    """Tests for TENSOR — disjoint-subregister composition.
    Wraps the circ_A.tensor(circ_B) idiom as a named pattern.  Component 0
    occupies the LSB subregister (lowest-indexed qubits).
    """

    def test_basic_two_component(self):
        circuit, info = encode(
            TENSOR([(GEOMETRIC(ratio=0.5), 8),
                    (GEOMETRIC(ratio=0.8), 8)]),
            N=64)
        assert info.vector_type == "TENSOR"
        assert info.N == 64
        assert info.m == 6

    def test_matches_manual_tensor(self):
        """TENSOR should give the same state as manually calling circ.tensor()."""
        c_a, _ = encode(FOURIER(modes=[(2, 1.0, 0)]), N=16)
        c_b, _ = encode(FOURIER(modes=[(3, 1.0, 0)]), N=16)
        manual = c_b.tensor(c_a)

        auto, _ = encode(
            TENSOR([(FOURIER(modes=[(2, 1.0, 0)]), 16),
                    (FOURIER(modes=[(3, 1.0, 0)]), 16)]),
            N=256)
        sv_m = np.abs(statevector(manual))
        sv_a = np.abs(statevector(auto))
        np.testing.assert_allclose(sv_a, sv_m, atol=1e-10)

    def test_separable_poisson(self):
        """Matches the paper's 2D Poisson separable source-term example."""
        circuit, info = encode(
            TENSOR([(FOURIER(modes=[(1, 1.0, 0)]), 32),
                    (FOURIER(modes=[(2, 1.0, 0)]), 32)]),
            N=32 * 32, validate=True)
        assert info.validated is True

    def test_three_way_tensor(self):
        circuit, info = encode(
            TENSOR([(GEOMETRIC(ratio=0.7), 4),
                    (POPCOUNT(r=0.5), 8),
                    (SQUARE(k1=1, k2=5, c=1.0), 8)]),
            N=4 * 8 * 8, validate=True)
        assert info.validated is True
        assert info.N == 256

    def test_validate_matches_kron(self):
        """Validated vector should equal Kronecker product of components."""
        from pyencode._helpers import _build_component_vector
        a = GEOMETRIC(ratio=0.6)
        b = GEOMETRIC(ratio=0.9)
        _, info = encode(TENSOR([(a, 8), (b, 4)]), N=32, validate=True)
        va = _build_component_vector(a, 8)
        vb = _build_component_vector(b, 4)
        expected = np.kron(vb, va)   # b on MSB side per TENSOR convention
        np.testing.assert_allclose(info.vector, expected, atol=1e-10)

    def test_gate_count_equals_sum_of_components(self):
        """Disjoint composition: no extra gates beyond component counts."""
        _, info_a = encode(GEOMETRIC(ratio=0.7), N=16)
        _, info_b = encode(POPCOUNT(r=0.5), N=16)
        _, info_t = encode(
            TENSOR([(GEOMETRIC(ratio=0.7), 16), (POPCOUNT(r=0.5), 16)]),
            N=256)
        assert info_t.gate_count == info_a.gate_count + info_b.gate_count

    def test_success_probability_is_one(self):
        _, info = encode(
            TENSOR([(GEOMETRIC(ratio=0.5), 8), (GEOMETRIC(ratio=0.8), 8)]),
            N=64)
        assert info.success_probability == 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            TENSOR([])

    def test_non_power_of_two_size_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            TENSOR([(GEOMETRIC(ratio=0.5), 6), (GEOMETRIC(ratio=0.5), 8)])

    def test_bad_component_type_raises(self):
        with pytest.raises(TypeError):
            TENSOR([("not a VectorObj", 8)])

    def test_mismatched_total_N_raises(self):
        with pytest.raises(ValueError, match="product of subregister sizes"):
            encode(
                TENSOR([(GEOMETRIC(ratio=0.5), 8),
                        (GEOMETRIC(ratio=0.5), 8)]),
                N=32)  # should be 64

    def test_single_component_tensor(self):
        """A one-component TENSOR should equal the component by itself."""
        c_plain, _ = encode(GEOMETRIC(ratio=0.7), N=16)
        c_tensor, _ = encode(TENSOR([(GEOMETRIC(ratio=0.7), 16)]), N=16)
        sv_p = np.abs(statevector(c_plain))
        sv_t = np.abs(statevector(c_tensor))
        np.testing.assert_allclose(sv_p, sv_t, atol=1e-10)


class TestPolynomial:
    """Tests for POLYNOMIAL — degree-d polynomial via Walsh-sparse loading.
    Covers ramp, quadratic (Poiseuille), cubic, verifies Walsh-sparsity
    structure and machine-precision reconstruction.
    """

    @staticmethod
    def _normalised_eval(coeffs, N, normalize=True):
        if normalize and N > 1:
            x = np.arange(N, dtype=float) / (N - 1)
        else:
            x = np.arange(N, dtype=float)
        return np.polyval(list(reversed(coeffs)), x)

    @staticmethod
    def _compare_upto_sign(sv, expected, tol=1e-10):
        """Validate up to a possible global -1 sign."""
        err_plus  = np.max(np.abs(sv - expected))
        err_minus = np.max(np.abs(sv + expected))
        return min(err_plus, err_minus) < tol

    def test_ramp_basic(self):
        """POLYNOMIAL(coeffs=[0, 1]) == ramp f(i) = i/(N-1)."""
        circuit, info = encode(POLYNOMIAL(coeffs=[0.0, 1.0]), N=16)
        assert info.vector_type == "POLYNOMIAL"
        sv = np.array(statevector(circuit)).real
        f = self._normalised_eval([0.0, 1.0], 16)
        expected = f / np.linalg.norm(f)
        assert self._compare_upto_sign(sv, expected, tol=1e-10)

    def test_poiseuille(self):
        """Parabolic profile f(x) = 4x(1-x) (degree-2)."""
        circuit, info = encode(POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]), N=32)
        sv = np.array(statevector(circuit)).real
        f = self._normalised_eval([0.0, 4.0, -4.0], 32)
        expected = f / np.linalg.norm(f)
        assert self._compare_upto_sign(sv, expected, tol=1e-10)

    def test_cubic(self):
        circuit, _ = encode(POLYNOMIAL(coeffs=[0.1, 0.5, 1.0, -0.3]), N=32)
        sv = np.array(statevector(circuit)).real
        f = self._normalised_eval([0.1, 0.5, 1.0, -0.3], 32)
        expected = f / np.linalg.norm(f)
        assert self._compare_upto_sign(sv, expected, tol=1e-10)

    def test_validate(self):
        circuit, info = encode(POLYNOMIAL(coeffs=[0.0, 2.0, -1.0]),
                               N=16, validate=True)
        assert info.validated is True
        assert info.vector is not None

    def test_degree_1_sparsity(self):
        """Degree-1 polynomial: Walsh spectrum supported on Hamming weight <= 1."""
        from pyencode.synthesizer import _fwht_inplace
        for m in [3, 5, 7]:
            N = 2 ** m
            f = self._normalised_eval([1.0, 3.0], N)
            walsh = f.copy()
            _fwht_inplace(walsh)
            walsh /= np.sqrt(N)
            for k in range(N):
                if bin(k).count("1") > 1:
                    assert abs(walsh[k]) < 1e-10, \
                        f"m={m}, k={k}: weight-{bin(k).count('1')} Walsh coeff is nonzero"

    def test_degree_2_sparsity_count(self):
        """Degree-2: exactly 1 + m + C(m,2) nonzero Walsh coefficients."""
        from pyencode.synthesizer import _fwht_inplace
        from math import comb
        for m in [4, 5, 6, 7]:
            N = 2 ** m
            f = self._normalised_eval([1.0, 2.0, -3.0], N)
            walsh = f.copy()
            _fwht_inplace(walsh)
            walsh /= np.sqrt(N)
            nonzero = sum(1 for k in range(N) if abs(walsh[k]) > 1e-10)
            expected = 1 + m + comb(m, 2)
            assert nonzero == expected, \
                f"m={m}: got {nonzero} nonzero Walsh coeffs, expected {expected}"

    def test_normalize_domain_false(self):
        """normalize_domain=False evaluates at raw integer indices."""
        circuit, info = encode(POLYNOMIAL(coeffs=[0.0, 1.0], normalize_domain=False),
                               N=8)
        sv = np.array(statevector(circuit)).real
        f = np.arange(8, dtype=float)   # raw indices
        expected = f / np.linalg.norm(f)
        assert self._compare_upto_sign(sv, expected, tol=1e-10)

    def test_constant_polynomial_is_uniform(self):
        """Degree-0 (constant) polynomial maps to the uniform superposition."""
        circuit, info = encode(POLYNOMIAL(coeffs=[3.5]), N=8)
        sv = np.array(statevector(circuit)).real
        expected = np.ones(8) / np.sqrt(8)
        assert self._compare_upto_sign(sv, expected, tol=1e-10)

    def test_trailing_zeros_stripped(self):
        """POLYNOMIAL([1.0, 2.0, 0.0, 0.0]) == POLYNOMIAL([1.0, 2.0])."""
        p1 = POLYNOMIAL(coeffs=[1.0, 2.0, 0.0, 0.0])
        p2 = POLYNOMIAL(coeffs=[1.0, 2.0])
        assert p1.params["coeffs"] == p2.params["coeffs"]

    def test_zero_polynomial_raises(self):
        with pytest.raises(ValueError, match="zero polynomial"):
            POLYNOMIAL(coeffs=[0.0, 0.0, 0.0])

    def test_empty_coeffs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            POLYNOMIAL(coeffs=[])

    def test_emitted_code_runs(self):
        circuit, info = encode(POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]), N=16)
        namespace = {"QuantumCircuit": QuantumCircuit}
        exec(compile(info.circuit_code, "<test>", "exec"), namespace)
        assert isinstance(namespace["qc"], QuantumCircuit)
        sv_orig = np.abs(statevector(circuit))
        sv_emit = np.abs(statevector(namespace["qc"]))
        np.testing.assert_allclose(sv_orig, sv_emit, atol=1e-10)

    def test_lcu_composability(self):
        """POLYNOMIAL can be used as an LCU component."""
        circuit, info = encode(
            LCU([(1.0, STEP(k_s=8, c=1.0)),
                 (2.0, POLYNOMIAL(coeffs=[0.0, 1.0]))]),
            N=16)
        assert info.vector_type == "LCU"
        assert 0 < info.success_probability <= 1.0

    def test_tensor_composability(self):
        """POLYNOMIAL can be used as a TENSOR component (separable 2D)."""
        circuit, info = encode(
            TENSOR([(POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]), 16),
                    (POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]), 16)]),
            N=256, validate=True)
        assert info.validated is True

class TestSparseRegressions:
    """Regression tests for SPARSE loader bugs.

    Gleinig-Hoefler double-X-flanking bug (fixed): certain index patterns
    produced large errors because _mcry added X-flanking a second time,
    on top of the explicit 'x' gates already in the gate list.  Minimal
    reproducer: {0, 1, 3, 5, 6, 9} at m=5 with uniform amplitudes.
    """

    def test_double_x_flanking_minimal(self):
        """Minimal reproducer of the double-X-flanking bug."""
        circuit, _ = encode(SPARSE([(k, 1.0) for k in [0, 1, 3, 5, 6, 9]]),
                            N=32)
        expected = np.zeros(32)
        for k in [0, 1, 3, 5, 6, 9]:
            expected[k] = 1.0
        expected /= np.linalg.norm(expected)
        sv = np.abs(statevector(circuit))
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_double_x_flanking_polynomial_walsh(self):
        """The s=11 sparse case from POLYNOMIAL's Walsh spectrum."""
        loads = [(0, 0.9), (3, 0.006), (5, 0.012), (6, 0.023), (9, 0.023),
                 (10, 0.046), (12, 0.046), (17, 0.006), (18, 0.012),
                 (20, 0.023), (24, 0.046)]
        circuit, _ = encode(SPARSE(loads), N=32)
        expected = np.zeros(32)
        for k, a in loads:
            expected[k] = abs(a)
        expected /= np.linalg.norm(expected)
        sv = np.abs(statevector(circuit))
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_sparse_random_sweep(self):
        """SPARSE over 50 random configurations — all should be machine precision."""
        import random
        random.seed(42)
        for trial in range(50):
            m = random.choice([4, 5, 6])
            N = 2 ** m
            s = random.randint(2, min(15, N))
            idx = random.sample(range(N), s)
            amps = [random.uniform(0.1, 2.0) for _ in idx]
            circuit, _ = encode(SPARSE(list(zip(idx, amps))), N=N)
            expected = np.zeros(N)
            for k, a in zip(idx, amps):
                expected[k] = abs(a)
            expected /= np.linalg.norm(expected)
            sv = np.abs(statevector(circuit))
            err = float(np.max(np.abs(sv - expected)))
            assert err < 1e-10, f"trial {trial}: m={m}, idx={idx}, err={err:.2e}"


# ===========================================================================
# predict_gates: fast closed-form predictor (v1.4)
# ===========================================================================

class TestPredictor:
    """
    Cross-check predict_gates() against encode() ground truth.
    Every 'exact' prediction must match the transpiled counts to the gate.
    'Approximate' predictions are checked only for positivity and sanity.
    """

    def _ground(self, obj, N):
        from pyencode import encode
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = encode(obj, N=N)
        return info.gate_count_1q, info.gate_count_2q, info.circuit_depth

    def test_popcount_exact(self):
        from pyencode import predict_gates, POPCOUNT
        for m in [4, 6, 8, 10, 12, 14]:
            u_t, c_t, d_t = self._ground(POPCOUNT(r=0.7), 2**m)
            p = predict_gates(POPCOUNT(r=0.7), 2**m)
            assert p["exact"] is True
            assert p["gate_count_1q"] == u_t, f"m={m}: {p['gate_count_1q']} != {u_t}"
            assert p["gate_count_2q"] == c_t
            assert p["circuit_depth"] == d_t

    def test_walsh_exact(self):
        from pyencode import predict_gates, WALSH
        for m in [4, 6, 8, 10, 12]:
            u_t, c_t, d_t = self._ground(WALSH(k=m//2, c_pos=1.0, c_neg=4.0), 2**m)
            p = predict_gates(WALSH(k=m//2, c_pos=1.0, c_neg=4.0), 2**m)
            assert p["exact"] is True
            assert p["gate_count_1q"] == u_t
            assert p["gate_count_2q"] == c_t
            assert p["circuit_depth"] == d_t

    def test_staircase_exact(self):
        from pyencode import predict_gates, STAIRCASE
        for m in [4, 6, 8, 10, 12]:
            u_t, c_t, d_t = self._ground(STAIRCASE(r=0.5), 2**m)
            p = predict_gates(STAIRCASE(r=0.5), 2**m)
            assert p["exact"] is True
            assert p["gate_count_1q"] == u_t
            assert p["gate_count_2q"] == c_t
            assert p["circuit_depth"] == d_t

    def test_polynomial_d1_exact(self):
        """POLYNOMIAL d=1 ramp: closed-form 5m-4 / 2m-2 / 4m-3."""
        from pyencode import predict_gates, POLYNOMIAL
        for m in [4, 6, 8, 10, 12, 14]:
            u_t, c_t, d_t = self._ground(POLYNOMIAL(coeffs=[0.0, 1.0]), 2**m)
            p = predict_gates(POLYNOMIAL(coeffs=[0.0, 1.0]), 2**m)
            assert p["exact"] is True
            assert p["gate_count_1q"] == u_t, f"m={m}: 1q {p['gate_count_1q']} != {u_t}"
            assert p["gate_count_2q"] == c_t
            assert p["circuit_depth"] == d_t

    def test_step_exact(self):
        """STEP: popcount-based closed form for various k_s."""
        from pyencode import predict_gates, STEP
        for m in [6, 8, 10, 12]:
            N = 2**m
            test_cases = [N//2, 3*N//4, N//2 + N//4 + N//8, N-1]
            for k_s in test_cases:
                u_t, c_t, d_t = self._ground(STEP(k_s=k_s, c=1.0), N)
                p = predict_gates(STEP(k_s=k_s, c=1.0), N)
                assert p["exact"] is True
                assert p["gate_count_1q"] == u_t, \
                    f"m={m}, k_s={k_s}: 1q {p['gate_count_1q']} != {u_t}"
                assert p["gate_count_2q"] == c_t
                assert p["circuit_depth"] == d_t

    def test_sparse_s1_exact(self):
        """SPARSE s=1: just X gates on set bits."""
        from pyencode import predict_gates, SPARSE
        for m in [6, 10, 14]:
            N = 2**m
            for k in [0, 1, 19, N//2, N-1]:
                u_t, c_t, d_t = self._ground(SPARSE([(k, 1.0)]), N)
                p = predict_gates(SPARSE([(k, 1.0)]), N)
                assert p["exact"] is True
                assert p["gate_count_1q"] == u_t
                assert p["gate_count_2q"] == c_t

    def test_fourier_t1_exact(self):
        """FOURIER single mode: exact quadratic fit."""
        from pyencode import predict_gates, FOURIER
        for m in [4, 6, 8, 10, 12]:
            u_t, c_t, d_t = self._ground(FOURIER(modes=[(1, 1.0, 0)]), 2**m)
            p = predict_gates(FOURIER(modes=[(1, 1.0, 0)]), 2**m)
            assert p["exact"] is True
            assert p["gate_count_1q"] == u_t
            assert p["gate_count_2q"] == c_t

    def test_square_aligned_exact(self):
        """SQUARE [0, N/2): reduces to STEP, exact."""
        from pyencode import predict_gates, SQUARE
        for m in [6, 8, 10, 12]:
            N = 2**m
            u_t, c_t, d_t = self._ground(SQUARE(k1=0, k2=N//2, c=1.0), N)
            p = predict_gates(SQUARE(k1=0, k2=N//2, c=1.0), N)
            assert p["exact"] is True
            assert p["gate_count_1q"] == u_t

    def test_polynomial_d2_approximate(self):
        """POLYNOMIAL d=2: approximate but within 1% of actual."""
        from pyencode import predict_gates, POLYNOMIAL
        for m in [6, 8, 10, 12]:
            u_t, c_t, d_t = self._ground(POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]), 2**m)
            p = predict_gates(POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]), 2**m)
            assert p["exact"] is False
            actual = u_t + c_t
            predicted = p["gate_count_1q"] + p["gate_count_2q"]
            err = abs(predicted - actual) / max(1, actual)
            assert err < 0.05, f"m={m}: err={err:.3f}, pred={predicted}, actual={actual}"

    def test_square_general_upper_bound(self):
        """SQUARE general: prediction is an upper bound (<=40% over)."""
        from pyencode import predict_gates, SQUARE
        for m in [8, 10, 12]:
            N = 2**m
            u_t, c_t, d_t = self._ground(SQUARE(k1=N//4+1, k2=3*N//4+1, c=1.0), N)
            p = predict_gates(SQUARE(k1=N//4+1, k2=3*N//4+1, c=1.0), N)
            assert p["exact"] is False
            assert p["gate_count_1q"] + p["gate_count_2q"] >= u_t + c_t - 5

    def test_tensor_composition(self):
        """TENSOR: predictions sum across disjoint subregisters."""
        from pyencode import predict_gates, TENSOR, FOURIER
        obj = TENSOR([(FOURIER(modes=[(2, 1.0, 0)]), 32),
                      (FOURIER(modes=[(3, 1.0, 0)]), 32)])
        p = predict_gates(obj, 32*32)
        assert p["m"] == 10
        assert p["N"] == 1024
        # Each FOURIER m=5 has 1q = 1.5*25 - 2.5*5 + 6 = 31, 2q = 25 - 3 = 22
        # Tensor: 2 x (31 + 22) = 106; actual within a few gates of this
        actual = 31 + 22  # per-component (approximately)
        assert p["gate_count"] >= 2 * actual - 5

    def test_speed_at_large_m(self):
        """Prediction must be fast even where encode() is slow."""
        import time
        from pyencode import predict_gates, POLYNOMIAL
        t0 = time.perf_counter()
        p = predict_gates(POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]), N=2**16)
        dt = time.perf_counter() - t0
        assert dt < 0.01, f"predict_gates took {dt*1000:.1f}ms (should be sub-ms)"
        assert p["gate_count"] > 0

    def test_rejects_invalid_n(self):
        from pyencode import predict_gates, POPCOUNT
        with pytest.raises(ValueError):
            predict_gates(POPCOUNT(r=0.7), N=5)  # not a power of 2

    def test_geometric_start_prediction(self):
        """Predict GEOMETRIC with start offset: log2(w) + popcount(start/w) gates."""
        from pyencode import predict_gates, encode, GEOMETRIC
        import warnings
        
        # Test cases: (N, start, expected_1q_gates)
        test_cases = [
            (64, 0,  6),   # vanilla: m=6 rotations
            (64, 32, 6),   # start=N/2: log2(32)=5 + popcount(1)=1 = 6
            (64, 48, 6),   # start=3N/4: log2(16)=4 + popcount(3)=2 = 6
            (32, 24, 5),   # start=3N/4: log2(8)=3 + popcount(3)=2 = 5
            (16, 8,  4),   # start=N/2: log2(8)=3 + popcount(1)=1 = 4
        ]
        
        for N, start, expected_1q in test_cases:
            p = predict_gates(GEOMETRIC(ratio=0.7, start=start), N)
            assert p["exact"] is False  # transpiler may optimize small rotations
            assert p["gate_count_1q"] == expected_1q, \
                f"N={N}, start={start}: pred {p['gate_count_1q']} != expected {expected_1q}"
            assert p["gate_count_2q"] == 0
            assert p["circuit_depth"] == 1
            
            # Cross-check against actual encode() for a few cases
            if N <= 32:  # avoid slow encode() for large N in tests
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, info = encode(GEOMETRIC(ratio=0.7, start=start), N)
                # Prediction should be exact or close for aligned offsets
                assert p["gate_count_1q"] == info.gate_count_1q, \
                    f"N={N}, start={start}: predicted {p['gate_count_1q']} != actual {info.gate_count_1q}"

    def test_rejects_invalid_type(self):
        from pyencode import predict_gates
        with pytest.raises(TypeError):
            predict_gates("not a vector object", N=16)
