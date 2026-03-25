"""
test_pyencode.py
========================
Unit tests for PyEncode.

Primary API (paper):
  encode()   — single entry point, typed constructors
  SPARSE     — Gleinig-Hoefler sparse state (replaces DISCRETE + MULTI_DISCRETE)
  FOURIER    — inverse-QFT sinusoidal modes (replaces SINE + COSINE + MULTI_SINE)

Legacy API (still supported):
  encode_params() — identical to encode()
  DISCRETE, UNIFORM, STEP, SQUARE, SINE, COSINE, MULTI_DISCRETE, MULTI_SINE

Run with:  python -m pytest test_pyencode.py -v
"""

import math
import numpy as np
import pytest

from qiskit import QuantumCircuit

from pyencode import (
    encode,
    encode_params,
    EncodingInfo,
    VectorType,
    SPARSE,
    FOURIER,
    DISCRETE,
    UNIFORM,
    STEP,
    SQUARE,
    SINE,
    COSINE,
    MULTI_DISCRETE,
    MULTI_SINE,
)
from pyencode.recognizer import recognise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def statevector(circuit):
    from qiskit.quantum_info import Statevector
    return np.array(Statevector(circuit))


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
# Primary API: encode() with SPARSE and FOURIER
# ===================================================================

class TestEncode:

    # ── SPARSE ────────────────────────────────────────────────────

    def test_sparse_single(self):
        circuit, info = encode(SPARSE([(3, 5.0)]), N=8)
        assert info.vector_type == "SPARSE"
        assert info.N == 8 and info.m == 3
        expected = np.zeros(8); expected[3] = 5.0
        assert_encodes(circuit, expected)

    def test_sparse_single_k0(self):
        circuit, info = encode(SPARSE([(0, 1.0)]), N=4)
        assert_encodes(circuit, np.array([1, 0, 0, 0], dtype=float))

    def test_sparse_single_last(self):
        circuit, info = encode(SPARSE([(63, 1.0)]), N=64)
        expected = np.zeros(64); expected[63] = 1.0
        assert_encodes(circuit, expected)

    def test_sparse_two_entries(self):
        circuit, info = encode(SPARSE([(1, 3.0), (6, 4.0)]), N=8)
        assert info.vector_type == "SPARSE"
        expected = np.zeros(8); expected[1] = 3.0; expected[6] = 4.0
        assert_encodes(circuit, expected)

    def test_sparse_five_entries(self):
        entries = [(1, 1.0), (5, 2.0), (10, 3.0), (20, 1.5), (30, 0.5)]
        circuit, info = encode(SPARSE(entries), N=32)
        expected = np.zeros(32)
        for k, p in entries: expected[k] = p
        assert_encodes(circuit, expected)

    def test_sparse_gate_count_single_is_hamming_weight(self):
        for k in [0, 1, 3, 7, 19]:
            _, info = encode(SPARSE([(k, 1.0)]), N=64)
            assert info.gate_count == bin(k).count('1'), \
                f"k={k}: expected {bin(k).count('1')} gates, got {info.gate_count}"

    def test_sparse_validate(self):
        _, info = encode(SPARSE([(3, 1.0)]), N=8, validate=True)
        assert info.validated

    def test_sparse_constructor_rejects_empty(self):
        with pytest.raises(ValueError):
            SPARSE([])

    def test_sparse_constructor_rejects_bad_input(self):
        with pytest.raises(TypeError):
            SPARSE([42])

    def test_sparse_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            encode(SPARSE([(10, 1.0)]), N=8)

    def test_sparse_duplicate_index_raises(self):
        with pytest.raises(ValueError, match="more than once"):
            encode(SPARSE([(3, 1.0), (3, 2.0)]), N=8)

    # ── FOURIER ───────────────────────────────────────────────────

    def test_fourier_single_sine(self):
        circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=64)
        assert info.vector_type == "FOURIER"
        k = np.arange(64)
        assert_encodes(circuit, np.sin(2 * np.pi * k / 64))

    def test_fourier_single_cosine(self):
        circuit, info = encode(FOURIER(modes=[(1, 1.0, math.pi / 2)]), N=64)
        k = np.arange(64)
        assert_encodes(circuit, np.cos(2 * np.pi * k / 64))

    def test_fourier_single_phase(self):
        phi = math.pi / 4
        circuit, info = encode(FOURIER(modes=[(3, 2.0, phi)]), N=64)
        k = np.arange(64)
        assert_encodes(circuit, 2.0 * np.sin(2 * np.pi * 3 * k / 64 + phi))

    def test_fourier_two_tuple_defaults_phi(self):
        circuit, info = encode(FOURIER(modes=[(1, 1.0)]), N=64)
        k = np.arange(64)
        assert_encodes(circuit, np.sin(2 * np.pi * k / 64))

    def test_fourier_multi_mode(self):
        circuit, info = encode(FOURIER(modes=[(1, 2.0, 0), (3, 1.0, 0)]), N=16)
        assert info.vector_type == "FOURIER"
        k = np.arange(16)
        expected = 2.0 * np.sin(2 * np.pi * k / 16) + np.sin(2 * np.pi * 3 * k / 16)
        assert_encodes(circuit, expected)

    def test_fourier_validate(self):
        _, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=8, validate=True)
        assert info.validated

    def test_fourier_constructor_rejects_empty(self):
        with pytest.raises(ValueError):
            FOURIER(modes=[])

    def test_fourier_constructor_rejects_bad_input(self):
        with pytest.raises(TypeError):
            FOURIER(modes=[42])

    def test_fourier_mode_zero_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            encode(FOURIER(modes=[(0, 1.0, 0)]), N=8)

    # ── STEP ─────────────────────────────────────────────────────

    def test_step(self):
        circuit, info = encode(STEP(k_s=4, c=2.0), N=8)
        assert info.vector_type == "STEP"
        expected = np.zeros(8); expected[:4] = 2.0
        assert_encodes(circuit, expected)

    def test_step_non_power_of_two(self):
        for k_s in [3, 5, 6, 7, 37, 48, 60, 63]:
            circuit, info = encode(STEP(k_s=k_s, c=1.0), N=64)
            expected = np.zeros(64); expected[:k_s] = 1.0
            assert_encodes(circuit, expected)

    def test_step_full_range_is_uniform(self):
        circuit, info = encode(STEP(k_s=8, c=1.0), N=8)
        assert_encodes(circuit, np.ones(8))

    # ── SQUARE ───────────────────────────────────────────────────

    def test_square(self):
        circuit, info = encode(SQUARE(k1=2, k2=6, c=1.0), N=8)
        assert info.vector_type == "SQUARE"
        expected = np.zeros(8); expected[2:6] = 1.0
        assert_encodes(circuit, expected)

    def test_square_general(self):
        for k1, k2 in [(1, 5), (3, 7), (8, 24), (10, 50)]:
            circuit, info = encode(SQUARE(k1=k1, k2=k2, c=1.0), N=64)
            expected = np.zeros(64); expected[k1:k2] = 1.0
            assert_encodes(circuit, expected)

    # ── error cases ───────────────────────────────────────────────

    def test_non_power_of_two_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            encode(SPARSE([(1, 1.0)]), N=6)

    def test_returns_encoding_info(self):
        _, info = encode(SPARSE([(1, 1.0)]), N=4)
        assert isinstance(info, EncodingInfo)

    def test_returns_qiskit_circuit(self):
        circuit, _ = encode(SPARSE([(1, 1.0)]), N=4)
        assert isinstance(circuit, QuantumCircuit)


# ===================================================================
# Legacy API: encode_params() unchanged
# ===================================================================

class TestEncodeParamsLegacy:

    def test_discrete(self):
        circuit, info = encode_params(DISCRETE(k=3, P=5.0), N=8)
        assert info.vector_type == "DISCRETE"
        expected = np.zeros(8); expected[3] = 5.0
        assert_encodes(circuit, expected)

    def test_uniform(self):
        circuit, info = encode_params(UNIFORM(c=3.0), N=8)
        assert info.vector_type == "UNIFORM"
        assert_encodes(circuit, np.full(8, 3.0))

    def test_step(self):
        circuit, info = encode_params(STEP(k_s=4, c=2.0), N=8)
        assert info.vector_type == "STEP"

    def test_square(self):
        circuit, info = encode_params(SQUARE(k1=8, k2=16, c=1.0), N=16)
        assert info.vector_type == "SQUARE"

    def test_sine(self):
        circuit, info = encode_params(SINE(n=1, A=1.0), N=64)
        assert info.vector_type == "SINE"
        k = np.arange(64)
        assert_encodes(circuit, np.sin(2 * np.pi * k / 64))

    def test_cosine(self):
        circuit, info = encode_params(COSINE(n=1, A=1.0), N=64)
        assert info.vector_type == "COSINE"
        k = np.arange(64)
        assert_encodes(circuit, np.cos(2 * np.pi * k / 64))

    def test_multi_discrete(self):
        circuit, info = encode_params(
            MULTI_DISCRETE(vectors=[DISCRETE(k=1, P=3.0), DISCRETE(k=6, P=4.0)]), N=8)
        expected = np.zeros(8); expected[1] = 3.0; expected[6] = 4.0
        assert_encodes(circuit, expected)

    def test_multi_sine(self):
        circuit, info = encode_params(
            MULTI_SINE(modes=[SINE(n=1, A=2.0), SINE(n=3, A=1.0)]), N=16)
        assert info.vector_type == "MULTI_SINE"

    def test_string_api(self):
        circuit, info = encode_params("DISCRETE", N=8, k=3, P=5.0)
        assert info.vector_type == "DISCRETE"

    def test_encode_equals_encode_params(self):
        c1, i1 = encode(STEP(k_s=4, c=1.0), N=8)
        c2, i2 = encode_params(STEP(k_s=4, c=1.0), N=8)
        assert i1.gate_count == i2.gate_count
        assert i1.complexity == i2.complexity

    def test_missing_required_param_raises(self):
        with pytest.raises(TypeError, match="Missing required"):
            encode_params("DISCRETE", N=8)

    def test_unexpected_param_raises(self):
        with pytest.raises(TypeError, match="Unexpected"):
            encode_params("DISCRETE", N=8, k=3, P=5.0, banana=42)

    def test_multi_discrete_wrong_type_raises(self):
        with pytest.raises(TypeError, match="DISCRETE"):
            MULTI_DISCRETE(vectors=[SINE(n=1)])

    def test_multi_sine_wrong_type_raises(self):
        with pytest.raises(TypeError, match="SINE"):
            MULTI_SINE(modes=[DISCRETE(k=1)])


# ===================================================================
# Recognizer (internal)
# ===================================================================

class TestRecognizer:

    def test_discrete_recognised(self):
        code = "import numpy as np\nN = 8\nf = np.zeros(N)\nf[3] = 5.0"
        pattern = recognise(code)
        assert pattern.load_type == VectorType.DISCRETE
        assert pattern.params["k"] == 3

    def test_uniform_recognised(self):
        pattern = recognise("N = 16\nf = np.ones(N)")
        assert pattern.load_type == VectorType.UNIFORM

    def test_step_recognised(self):
        pattern = recognise("N = 8\nf = np.zeros(N)\nf[:4] = 2.0")
        assert pattern.load_type == VectorType.STEP

    def test_sine_mode3(self):
        code = "N = 16\nx = np.linspace(0, 1.0, N)\nf = np.sin(3 * np.pi * x / 1.0)"
        pattern = recognise(code)
        assert pattern.load_type == VectorType.SINE
        assert pattern.params["n"] == 3

    def test_unknown_pattern(self):
        pattern = recognise("N = 8\nf = np.random.rand(N)")
        assert pattern.load_type == VectorType.UNKNOWN


# ===================================================================
# Display / string
# ===================================================================

class TestDisplay:

    def test_info_str_contains_vector_type(self):
        _, info = encode(SPARSE([(0, 1.0)]), N=4)
        s = str(info)
        assert "SPARSE" in s
        assert "PyEncode" in s

    def test_info_str_legacy(self):
        _, info = encode_params(DISCRETE(k=0, P=1.0), N=4)
        s = str(info)
        assert "DISCRETE" in s

    def test_circuit_is_qiskit(self):
        circuit, _ = encode(SPARSE([(1, 1.0)]), N=4)
        assert isinstance(circuit, QuantumCircuit)


# ===================================================================
# Constructors
# ===================================================================

class TestConstructors:

    def test_sparse_repr(self):
        s = SPARSE([(3, 5.0)])
        assert "SPARSE" in repr(s)
        assert s.vector_type == VectorType.SPARSE

    def test_fourier_repr(self):
        f = FOURIER(modes=[(1, 1.0, 0)])
        assert "FOURIER" in repr(f)
        assert f.vector_type == VectorType.FOURIER

    def test_sparse_stores_loads(self):
        s = SPARSE([(1, 2.0), (5, 3.0)])
        assert len(s.params["loads"]) == 2
        assert s.params["loads"][0] == {"k": 1, "P": 2.0}

    def test_fourier_stores_modes(self):
        f = FOURIER(modes=[(1, 2.0, 0.0), (3, 1.0, 0.5)])
        assert len(f.params["modes"]) == 2
        assert f.params["modes"][0]["n"] == 1

    def test_fourier_phi_default(self):
        f = FOURIER(modes=[(1, 1.0)])
        assert f.params["modes"][0]["phi"] == 0.0

    def test_discrete_repr(self):
        d = DISCRETE(k=3, P=5.0)
        assert "DISCRETE" in repr(d)
        assert d.vector_type == VectorType.DISCRETE

    def test_sine_defaults(self):
        s = SINE(n=3)
        assert s.params["A"] == 1.0
        assert s.params["phi"] == 0.0

    def test_multi_discrete_validates(self):
        md = MULTI_DISCRETE(vectors=[DISCRETE(k=1, P=2.0), DISCRETE(k=5, P=3.0)])
        assert len(md.params["loads"]) == 2

    def test_multi_sine_validates(self):
        ms = MULTI_SINE(modes=[SINE(n=1, A=2.0), SINE(n=3, A=1.0)])
        assert len(ms.params["modes"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
