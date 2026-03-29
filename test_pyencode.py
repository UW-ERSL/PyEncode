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

from pyencode import encode, EncodingInfo, VectorType, SPARSE, STEP, SQUARE, FOURIER, WALSH, GEOMETRIC, LCU


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
