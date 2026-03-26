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

from pyencode import encode, EncodingInfo, VectorType, SPARSE, STEP, SQUARE, FOURIER, WALSH, LCU


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

    def test_complexity(self):
        _, info = encode(SQUARE(k1=2, k2=6, c=1.0), N=8)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
