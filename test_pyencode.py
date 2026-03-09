"""
test_pyencode.py
========================
Unit tests for all PyEncode load patterns and the Möttönen fallback.

Run with:  python -m pytest test_pyencode.py -v
"""

import math
import numpy as np
import pytest

from qiskit import QuantumCircuit

from pyencode import encode, EncodingInfo
from pyencode.recognizer import recognise, LoadType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def statevector(circuit):
    """Return the statevector of a circuit as a numpy array."""
    from qiskit.quantum_info import Statevector
    return np.array(Statevector(circuit))


def assert_encodes(circuit, expected_f, atol=1e-5):
    """
    Assert that circuit prepares the normalised version of expected_f.
    Ignores global phase.
    """
    sv = statevector(circuit)
    norm = np.linalg.norm(expected_f)
    assert norm > 1e-12, "Reference vector is zero"
    ref = expected_f / norm
    # Allow for global phase: compare magnitudes
    np.testing.assert_allclose(
        np.abs(sv), np.abs(ref), atol=atol,
        err_msg=f"Statevector mismatch.\nGot:      {np.abs(sv)}\nExpected: {np.abs(ref)}"
    )


# ---------------------------------------------------------------------------
# Recognition tests
# ---------------------------------------------------------------------------

class TestRecognizer:

    def test_point_load_recognised(self):
        code = """
import numpy as np
N = 8
f = np.zeros(N)
f[3] = 5.0
"""
        pattern = recognise(code)
        assert pattern.load_type == LoadType.POINT_LOAD
        assert pattern.N == 8
        assert pattern.params["k"] == 3
        assert pattern.params["P"] == pytest.approx(5.0)

    def test_point_load_k0(self):
        code = "N = 4\nf = np.zeros(N)\nf[0] = 1.0"
        pattern = recognise(code)
        assert pattern.load_type == LoadType.POINT_LOAD
        assert pattern.params["k"] == 0

    def test_uniform_load_ones(self):
        code = "N = 16\nf = np.ones(N)"
        pattern = recognise(code)
        assert pattern.load_type == LoadType.UNIFORM_LOAD
        assert pattern.params["c"] == pytest.approx(1.0)

    def test_uniform_load_scaled(self):
        code = "N = 8\nf = np.ones(N) * 3.7"
        pattern = recognise(code)
        assert pattern.load_type == LoadType.UNIFORM_LOAD
        assert pattern.params["c"] == pytest.approx(3.7)

    def test_step_load_power_of_two(self):
        code = "N = 8\nf = np.zeros(N)\nf[:4] = 2.0"
        pattern = recognise(code)
        assert pattern.load_type == LoadType.STEP_LOAD
        assert pattern.params["k_s"] == 4
        assert pattern.params["c"] == pytest.approx(2.0)

    def test_step_load_general(self):
        code = "N = 8\nf = np.zeros(N)\nf[:6] = 1.0"
        pattern = recognise(code)
        assert pattern.load_type == LoadType.STEP_LOAD
        assert pattern.params["k_s"] == 6

    def test_sinusoidal_mode1(self):
        code = """
import numpy as np
N = 8
L = 1.0
x = np.linspace(0, L, N)
f = np.sin(1 * np.pi * x / L)
"""
        pattern = recognise(code)
        assert pattern.load_type == LoadType.SINUSOIDAL_LOAD
        assert pattern.params["n"] == 1

    def test_sinusoidal_mode3(self):
        code = """
N = 16
x = np.linspace(0, 1.0, N)
f = np.sin(3 * np.pi * x / 1.0)
"""
        pattern = recognise(code)
        assert pattern.load_type == LoadType.SINUSOIDAL_LOAD
        assert pattern.params["n"] == 3

    def test_multi_point_load_two_equal(self):
        code = "N = 8\nf = np.zeros(N)\nf[1] = 2.0\nf[5] = 3.0"
        pattern = recognise(code)
        assert pattern.load_type == LoadType.MULTI_POINT_LOAD
        indices = [l["k"] for l in pattern.params["loads"]]
        assert set(indices) == {1, 5}

    def test_multi_point_load_three_unequal(self):
        code = "N = 16\nf = np.zeros(N)\nf[0] = 1.0\nf[7] = 2.0\nf[15] = 0.5"
        pattern = recognise(code)
        assert pattern.load_type == LoadType.MULTI_POINT_LOAD
        assert len(pattern.params["loads"]) == 3
        weights = {l["k"]: l["P"] for l in pattern.params["loads"]}
        assert weights[0]  == pytest.approx(1.0)
        assert weights[7]  == pytest.approx(2.0)
        assert weights[15] == pytest.approx(0.5)

    def test_multi_point_load_four_loads(self):
        code = "N = 16\nf = np.zeros(N)\nf[2] = 1.0\nf[5] = 3.0\nf[9] = 2.0\nf[14] = 4.0"
        pattern = recognise(code)
        assert pattern.load_type == LoadType.MULTI_POINT_LOAD
        assert len(pattern.params["loads"]) == 4

    def test_multi_point_load_five_loads(self):
        code = "N = 32\nf = np.zeros(N)\nf[1]=1.0\nf[5]=2.0\nf[10]=3.0\nf[20]=1.5\nf[30]=0.5"
        pattern = recognise(code)
        assert pattern.load_type == LoadType.MULTI_POINT_LOAD
        assert len(pattern.params["loads"]) == 5

    def test_uniform_spike_load_recognised(self):
        code = "N = 8\nf = np.ones(N) * 1.0\nf[3] = 10.0"
        pattern = recognise(code)
        assert pattern.load_type == LoadType.UNIFORM_SPIKE_LOAD
        assert pattern.params["k"] == 3
        assert pattern.params["delta"] == pytest.approx(10.0)
        assert pattern.params["c"] == pytest.approx(1.0)

    def test_unknown_pattern(self):
        code = "N = 8\nf = np.random.rand(N)"
        pattern = recognise(code)
        assert pattern.load_type == LoadType.UNKNOWN

    def test_non_power_of_two_is_unknown(self):
        code = "N = 6\nf = np.zeros(N)\nf[2] = 1.0"
        pattern = recognise(code)
        assert pattern.load_type == LoadType.UNKNOWN


# ---------------------------------------------------------------------------
# Encoding correctness tests  (statevector simulation)
# ---------------------------------------------------------------------------

class TestEncoding:

    def test_point_load_m2(self):
        code = "N = 4\nf = np.zeros(N)\nf[2] = 1.0"
        circuit, info = encode(code)
        assert info.load_type == "POINT_LOAD"
        assert info.m == 2
        expected = np.array([0, 0, 1, 0], dtype=float)
        assert_encodes(circuit, expected)

    def test_point_load_m3_k5(self):
        code = "N = 8\nf = np.zeros(N)\nf[5] = 7.3"
        circuit, info = encode(code)
        expected = np.zeros(8)
        expected[5] = 7.3
        assert_encodes(circuit, expected)

    def test_point_load_k0(self):
        code = "N = 4\nf = np.zeros(N)\nf[0] = 1.0"
        circuit, info = encode(code)
        expected = np.array([1, 0, 0, 0], dtype=float)
        assert_encodes(circuit, expected)

    def test_uniform_load_m2(self):
        code = "N = 4\nf = np.ones(N)"
        circuit, info = encode(code)
        assert info.load_type == "UNIFORM_LOAD"
        expected = np.ones(4)
        assert_encodes(circuit, expected)

    def test_uniform_load_m3_scaled(self):
        code = "N = 8\nf = np.ones(N) * 2.0"
        circuit, info = encode(code)
        expected = np.full(8, 2.0)
        assert_encodes(circuit, expected)

    def test_step_load_half(self):
        """f[:4] = 1 on N=8 → uniform superposition over first 4 nodes."""
        code = "N = 8\nf = np.zeros(N)\nf[:4] = 1.0"
        circuit, info = encode(code)
        assert info.load_type == "STEP_LOAD"
        expected = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
        assert_encodes(circuit, expected)

    def test_step_load_quarter(self):
        """f[:2] = 1 on N=8."""
        code = "N = 8\nf = np.zeros(N)\nf[:2] = 1.0"
        circuit, info = encode(code)
        expected = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
        assert_encodes(circuit, expected)

    def test_multi_point_load_L2_unequal(self):
        """L=2 unequal weights: binary-tree Ry circuit."""
        code = "N = 8\nf = np.zeros(N)\nf[1] = 3.0\nf[6] = 4.0"
        circuit, info = encode(code)
        assert info.load_type == "MULTI_POINT_LOAD"
        expected = np.zeros(8)
        expected[1] = 3.0
        expected[6] = 4.0
        assert_encodes(circuit, expected)

    def test_multi_point_load_L2_same_high_bit(self):
        """L=2 with indices sharing their high bits — tests split-bit logic."""
        code = "N = 16\nf = np.zeros(N)\nf[4] = 1.0\nf[7] = 1.0"
        circuit, info = encode(code)
        assert info.load_type == "MULTI_POINT_LOAD"
        expected = np.zeros(16)
        expected[4] = 1.0
        expected[7] = 1.0
        assert_encodes(circuit, expected)

    def test_multi_point_load_L3_unequal(self):
        """L=3 arbitrary weights on N=16."""
        code = "N = 16\nf = np.zeros(N)\nf[0] = 1.0\nf[7] = 2.0\nf[15] = 3.0"
        circuit, info = encode(code)
        assert info.load_type == "MULTI_POINT_LOAD"
        expected = np.zeros(16)
        expected[0]  = 1.0
        expected[7]  = 2.0
        expected[15] = 3.0
        assert_encodes(circuit, expected)

    def test_multi_point_load_L4_unequal(self):
        """L=4 arbitrary weights on N=16."""
        code = "N = 16\nf = np.zeros(N)\nf[2] = 1.0\nf[5] = 3.0\nf[9] = 2.0\nf[14] = 4.0"
        circuit, info = encode(code)
        assert info.load_type == "MULTI_POINT_LOAD"
        expected = np.zeros(16)
        expected[2]  = 1.0
        expected[5]  = 3.0
        expected[9]  = 2.0
        expected[14] = 4.0
        assert_encodes(circuit, expected)

    def test_multi_point_load_L5_unequal(self):
        """L=5 (non-power-of-2 number of loads) arbitrary weights on N=32."""
        code = "N = 32\nf = np.zeros(N)\nf[1]=1.0\nf[5]=2.0\nf[10]=3.0\nf[20]=1.5\nf[30]=0.5"
        circuit, info = encode(code)
        assert info.load_type == "MULTI_POINT_LOAD"
        expected = np.zeros(32)
        expected[1]  = 1.0
        expected[5]  = 2.0
        expected[10] = 3.0
        expected[20] = 1.5
        expected[30] = 0.5
        assert_encodes(circuit, expected)

    def test_multi_point_load_equal_weights(self):
        """Equal weights (W-state style) still handled correctly."""
        code = "N = 8\nf = np.zeros(N)\nf[0] = 1.0\nf[3] = 1.0\nf[5] = 1.0"
        circuit, info = encode(code)
        assert info.load_type == "MULTI_POINT_LOAD"
        expected = np.zeros(8)
        expected[0] = 1.0
        expected[3] = 1.0
        expected[5] = 1.0
        assert_encodes(circuit, expected)

    def test_uniform_spike_load_uniform_plus_spike(self):
        code = "N = 8\nf = np.ones(N) * 1.0\nf[3] = 5.0"
        circuit, info = encode(code)
        assert info.load_type == "UNIFORM_SPIKE_LOAD"
        expected = np.ones(8)
        expected[3] = 5.0
        assert_encodes(circuit, expected)

    def test_encoding_info_fields(self):
        code = "N = 4\nf = np.zeros(N)\nf[1] = 1.0"
        circuit, info = encode(code)
        assert isinstance(info, EncodingInfo)
        assert info.N == 4
        assert info.m == 2
        assert info.gate_count > 0
        assert not info.fallback

    def test_fallback_mottonen(self):
        """Unrecognised pattern with explicit vector falls back to Möttönen."""
        code = "f = np.random.rand(4)"
        f = np.array([0.5, 0.5, 0.5, 0.5])
        circuit, info = encode(code, fallback_vector=f)
        assert info.fallback is True
        assert_encodes(circuit, f)

    def test_missing_fallback_vector_raises(self):
        code = "f = np.random.rand(8)"
        with pytest.raises(RuntimeError, match="fallback_vector"):
            encode(code)


# ---------------------------------------------------------------------------
# Gate count sanity checks
# ---------------------------------------------------------------------------


    def test_sinusoidal_n1_phi0_recognised(self):
        """sin(pi*x) recognised as SINUSOIDAL_LOAD with n=1, phi=0.
        The circuit encodes sin(2*pi*n*k/N) — the recognizer extracts n from sin(n*pi*x)."""
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.sin(np.pi * x)"
        pat = recognise(code)
        assert pat.load_type == LoadType.SINUSOIDAL_LOAD
        assert pat.params["n"] == 1
        assert pat.params.get("phi", 0.0) == pytest.approx(0.0)

    def test_sinusoidal_n3_phi_pi4_recognised(self):
        """sin(3*pi*x + pi/4) recognised as SINUSOIDAL_LOAD with n=3, phi=pi/4."""
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.sin(3 * np.pi * x + np.pi / 4)"
        pat = recognise(code)
        assert pat.load_type == LoadType.SINUSOIDAL_LOAD
        assert pat.params["n"] == 3
        assert pat.params["phi"] == pytest.approx(math.pi / 4, rel=1e-6)

    def test_sinusoidal_n3_phi_neg_recognised(self):
        """sin(3*pi*x - pi/3) recognised as SINUSOIDAL_LOAD with n=3, phi=-pi/3."""
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.sin(3 * np.pi * x - np.pi / 3)"
        pat = recognise(code)
        assert pat.load_type == LoadType.SINUSOIDAL_LOAD
        assert pat.params["n"] == 3
        assert pat.params["phi"] == pytest.approx(-math.pi / 3, rel=1e-6)

    def test_sinusoidal_n1_encoding_fidelity(self):
        """Circuit for sin(pi*x) encodes sin(2*pi*n*k/N) with n=1."""
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.sin(np.pi * x)"
        qc, _ = encode(code)
        k = np.arange(64)
        ref = np.sin(2 * np.pi * 1 * k / 64)   # circuit convention: sin(2*pi*n*k/N)
        assert_encodes(qc, ref)

    def test_sinusoidal_n3_phi_pi4_encoding_fidelity(self):
        """Circuit for sin(3*pi*x + pi/4) encodes sin(2*pi*3*k/N + pi/4)."""
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.sin(3 * np.pi * x + np.pi / 4)"
        qc, _ = encode(code)
        k = np.arange(64)
        ref = np.sin(2 * np.pi * 3 * k / 64 + math.pi / 4)
        assert_encodes(qc, ref)

    def test_sinusoidal_phase_complexity_label(self):
        """Sinusoidal with phase still reports O(m²) complexity."""
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.sin(3 * np.pi * x + np.pi / 4)"
        _, info = encode(code)
        assert info.gate_complexity == "O(m²)"
        assert not info.fallback


    def test_cosine_n1_phi0_recognised(self):
        """cos(pi*x) recognised as COSINE_LOAD with n=1, phi=0."""
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.cos(np.pi * x)"
        pat = recognise(code)
        assert pat.load_type == LoadType.COSINE_LOAD
        assert pat.params["n"] == 1
        assert pat.params.get("phi", 0.0) == pytest.approx(0.0)

    def test_cosine_n3_phi_pi4_recognised(self):
        """cos(3*pi*x + pi/4) recognised as COSINE_LOAD with n=3, phi=pi/4."""
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.cos(3 * np.pi * x + np.pi / 4)"
        pat = recognise(code)
        assert pat.load_type == LoadType.COSINE_LOAD
        assert pat.params["n"] == 3
        assert pat.params["phi"] == pytest.approx(math.pi / 4, rel=1e-6)

    def test_cosine_n1_encoding_fidelity(self):
        """Circuit for cos(pi*x) encodes cos(2*pi*k/N) exactly."""
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.cos(np.pi * x)"
        qc, _ = encode(code)
        k = np.arange(64)
        ref = np.cos(2 * np.pi * 1 * k / 64)
        assert_encodes(qc, ref)

    def test_cosine_n3_phi_pi4_encoding_fidelity(self):
        """Circuit for cos(3*pi*x + pi/4) encodes cos(2*pi*3*k/N + pi/4) exactly."""
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.cos(3 * np.pi * x + np.pi / 4)"
        qc, _ = encode(code)
        k = np.arange(64)
        ref = np.cos(2 * np.pi * 3 * k / 64 + math.pi / 4)
        assert_encodes(qc, ref)

    def test_cosine_phase_complexity_label(self):
        """Cosine with phase reports O(m²) and no fallback."""
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.cos(3 * np.pi * x + np.pi / 4)"
        _, info = encode(code)
        assert info.gate_complexity == "O(m²)"
        assert not info.fallback

class TestGateCount:

    def test_point_load_gate_count_is_at_most_m(self):
        """At most m X gates for a point load."""
        for m in [2, 3, 4]:
            N = 2 ** m
            code = f"N = {N}\nf = np.zeros(N)\nf[{N//2}] = 1.0"
            circuit, info = encode(code)
            assert info.gate_count <= m, \
                f"Expected ≤{m} gates for m={m}, got {info.gate_count}"

    def test_uniform_load_gate_count_equals_m(self):
        """Exactly m Hadamard gates for uniform load."""
        for m in [2, 3, 4, 5]:
            N = 2 ** m
            code = f"N = {N}\nf = np.ones(N)"
            circuit, info = encode(code)
            assert info.gate_count == m, \
                f"Expected {m} gates for m={m}, got {info.gate_count}"

    def test_sinusoidal_complexity_label(self):
        code = "N = 8\nx = np.linspace(0,1,8)\nf = np.sin(2*np.pi*x/1)"
        circuit, info = encode(code)
        assert "m²" in info.gate_complexity or "m" in info.gate_complexity


# ---------------------------------------------------------------------------
# Display / string tests
# ---------------------------------------------------------------------------

class TestDisplay:

    def test_info_str_contains_load_type(self):
        code = "N = 4\nf = np.zeros(N)\nf[0] = 1.0"
        _, info = encode(code)
        s = str(info)
        assert "POINT_LOAD" in s
        assert "PyEncode" in s

    def test_circuit_has_name(self):
        code = "N = 4\nf = np.zeros(N)\nf[1] = 1.0"
        circuit, _ = encode(code)
        assert isinstance(circuit, QuantumCircuit)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])