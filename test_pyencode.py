"""
test_pyencode.py
========================
Unit tests for PyEncode's three entry points:

  encode_params  — direct parameter specification (typed constructors)
  encode_vector  — vector input with optional type hint / auto-detect
  encode_python  — compile Python source code (AST or code execution)

Run with:  python -m pytest test_pyencode.py -v
"""

import math
import warnings
import numpy as np
import pytest

from qiskit import QuantumCircuit

from pyencode import (
    encode_params,
    encode_vector,
    encode_python,
    EncodingInfo,
    VectorType,
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
    """Return the statevector of a circuit as a numpy array."""
    from qiskit.quantum_info import Statevector
    return np.array(Statevector(circuit))


def assert_encodes(circuit, expected_f, tol=1e-5):
    """Assert that circuit prepares the normalised version of expected_f."""
    sv = statevector(circuit)
    norm = np.linalg.norm(expected_f)
    assert norm > 1e-12, "Reference vector is zero"
    ref = expected_f / norm
    np.testing.assert_allclose(
        np.abs(sv), np.abs(ref), atol=tol,
        err_msg=f"Statevector mismatch.\nGot:      {np.abs(sv)}\nExpected: {np.abs(ref)}"
    )


def _exec_circuit_code(code_str: str) -> QuantumCircuit:
    """Execute emitted code and return the 'qc' circuit object."""
    ns = {}
    exec(compile(code_str, "<emitted>", "exec"), ns)
    assert "qc" in ns, "Emitted code must define variable 'qc'"
    return ns["qc"]


# ===================================================================
# encode_params tests — typed constructors
# ===================================================================

class TestEncodeParams:

    def test_discrete(self):
        circuit, info = encode_params(DISCRETE(k=3, P=5.0), N=8)
        assert info.vector_type == "DISCRETE"
        assert info.N == 8
        assert info.m == 3
        expected = np.zeros(8); expected[3] = 5.0
        assert_encodes(circuit, expected)

    def test_discrete_k0(self):
        circuit, info = encode_params(DISCRETE(k=0, P=1.0), N=4)
        assert_encodes(circuit, np.array([1, 0, 0, 0], dtype=float))

    def test_discrete_enum(self):
        circuit, info = encode_params(DISCRETE(k=5, P=7.3), N=8)
        assert info.vector_type == "DISCRETE"
        expected = np.zeros(8); expected[5] = 7.3
        assert_encodes(circuit, expected)

    def test_uniform(self):
        circuit, info = encode_params(UNIFORM(c=3.0), N=8)
        assert info.vector_type == "UNIFORM"
        assert_encodes(circuit, np.full(8, 3.0))

    def test_uniform_unit(self):
        circuit, info = encode_params(UNIFORM(c=1.0), N=16)
        assert_encodes(circuit, np.ones(16))

    def test_step(self):
        circuit, info = encode_params(STEP(k_s=4, c=2.0), N=8)
        assert info.vector_type == "STEP"
        expected = np.zeros(8); expected[:4] = 2.0
        assert_encodes(circuit, expected)

    def test_square(self):
        circuit, info = encode_params(SQUARE(k1=8, k2=16, c=1.0), N=16)
        assert info.vector_type == "SQUARE"
        expected = np.zeros(16); expected[8:16] = 1.0
        assert_encodes(circuit, expected)

    def test_sine(self):
        circuit, info = encode_params(SINE(n=1, A=1.0), N=64)
        assert info.vector_type == "SINE"
        assert not info.validated
        assert "m\u00b2" in info.complexity
        k = np.arange(64)
        assert_encodes(circuit, np.sin(2 * np.pi * 1 * k / 64))

    def test_sine_mode3_phase(self):
        phi = math.pi / 4
        circuit, info = encode_params(SINE(n=3, A=2.0, phi=phi), N=64)
        k = np.arange(64)
        assert_encodes(circuit, 2.0 * np.sin(2 * np.pi * 3 * k / 64 + phi))

    def test_sine_phi_defaults_to_zero(self):
        _, info = encode_params(SINE(n=1, A=1.0), N=64)
        assert info.params.get("phi", 0.0) == pytest.approx(0.0)

    def test_cosine(self):
        circuit, info = encode_params(COSINE(n=1, A=1.0), N=64)
        assert info.vector_type == "COSINE"
        k = np.arange(64)
        assert_encodes(circuit, np.cos(2 * np.pi * 1 * k / 64))

    def test_cosine_mode3_phase(self):
        phi = math.pi / 4
        circuit, info = encode_params(COSINE(n=3, A=1.0, phi=phi), N=64)
        k = np.arange(64)
        assert_encodes(circuit, np.cos(2 * np.pi * 3 * k / 64 + phi))

    def test_multi_discrete(self):
        circuit, info = encode_params(
            MULTI_DISCRETE(vectors=[DISCRETE(k=1, P=3.0), DISCRETE(k=6, P=4.0)]),
            N=8,
        )
        expected = np.zeros(8); expected[1] = 3.0; expected[6] = 4.0
        assert_encodes(circuit, expected)

    def test_multi_discrete_L5(self):
        circuit, info = encode_params(
            MULTI_DISCRETE(vectors=[
                DISCRETE(k=1, P=1.0), DISCRETE(k=5, P=2.0),
                DISCRETE(k=10, P=3.0), DISCRETE(k=20, P=1.5),
                DISCRETE(k=30, P=0.5),
            ]),
            N=32,
        )
        expected = np.zeros(32)
        for k, p in [(1, 1.0), (5, 2.0), (10, 3.0), (20, 1.5), (30, 0.5)]:
            expected[k] = p
        assert_encodes(circuit, expected)

    def test_multi_sine(self):
        circuit, info = encode_params(
            MULTI_SINE(modes=[SINE(n=1, A=2.0), SINE(n=3, A=1.0)]),
            N=16,
        )
        assert info.vector_type == "MULTI_SINE"
        assert not info.validated

    def test_composite_list_discrete(self):
        """List of DISCRETE constructors creates MULTI_DISCRETE."""
        circuit, info = encode_params(
            [DISCRETE(k=1, P=3.0), DISCRETE(k=6, P=4.0)],
            N=8,
        )
        assert info.vector_type == "MULTI_DISCRETE"

    def test_composite_list_sine(self):
        """List of SINE constructors creates MULTI_SINE."""
        circuit, info = encode_params(
            [SINE(n=1, A=2.0), SINE(n=3, A=1.0)],
            N=16,
        )
        assert info.vector_type == "MULTI_SINE"

    def test_circuit_code_produced(self):
        _, info = encode_params(DISCRETE(k=5, P=2.0), N=8)
        assert info.circuit_code != ""
        qc = _exec_circuit_code(info.circuit_code)
        assert isinstance(qc, QuantumCircuit)

    def test_gate_count_discrete_at_most_m(self):
        for m in [2, 3, 4]:
            N = 2 ** m
            _, info = encode_params(DISCRETE(k=N // 2, P=1.0), N=N)
            assert info.gate_count <= m

    def test_gate_count_uniform_equals_m(self):
        for m in [2, 3, 4, 5]:
            N = 2 ** m
            _, info = encode_params(UNIFORM(c=1.0), N=N)
            assert info.gate_count == m

    # ── backward compat: string-based API still works ────────────

    def test_string_discrete(self):
        circuit, info = encode_params("DISCRETE", N=8, k=3, P=5.0)
        assert info.vector_type == "DISCRETE"
        expected = np.zeros(8); expected[3] = 5.0
        assert_encodes(circuit, expected)

    def test_old_name_point_load(self):
        """Old name POINT_LOAD maps to DISCRETE."""
        circuit, info = encode_params("POINT_LOAD", N=8, k=3, P=5.0)
        assert info.vector_type == "DISCRETE"

    def test_old_name_sinusoidal_load(self):
        """Old name SINUSOIDAL_LOAD maps to SINE."""
        circuit, info = encode_params("SINUSOIDAL_LOAD", N=64, n=1, A=1.0)
        assert info.vector_type == "SINE"

    def test_case_insensitive_string(self):
        _, info = encode_params("discrete", N=4, k=2, P=1.0)
        assert info.vector_type == "DISCRETE"

    # ── error cases ───────────────────────────────────────────────

    def test_non_power_of_two_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            encode_params(DISCRETE(k=3, P=5.0), N=6)

    def test_missing_required_param_raises(self):
        with pytest.raises(TypeError, match="Missing required"):
            encode_params("DISCRETE", N=8)  # missing k

    def test_amplitude_params_optional(self):
        """P, c, A are absorbed by normalisation and can be omitted."""
        circuit, info = encode_params(DISCRETE(k=3), N=8)
        assert info.vector_type == "DISCRETE"
        circuit, info = encode_params(UNIFORM(), N=8)
        assert info.vector_type == "UNIFORM"
        circuit, info = encode_params(STEP(k_s=4), N=8)
        assert info.vector_type == "STEP"
        circuit, info = encode_params(SQUARE(k1=8, k2=16), N=16)
        assert info.vector_type == "SQUARE"
        circuit, info = encode_params(SINE(n=1), N=64)
        assert info.vector_type == "SINE"
        circuit, info = encode_params(COSINE(n=1), N=64)
        assert info.vector_type == "COSINE"

    def test_unexpected_param_raises(self):
        with pytest.raises(TypeError, match="Unexpected"):
            encode_params("DISCRETE", N=8, k=3, P=5.0, banana=42)

    def test_invalid_vector_type_string_raises(self):
        with pytest.raises(ValueError, match="Unknown vector_type"):
            encode_params("BANANA", N=8, k=3, P=5.0)

    def test_unknown_vector_type_raises(self):
        with pytest.raises(ValueError, match="UNKNOWN is not valid"):
            encode_params("UNKNOWN", N=8, k=3, P=5.0)

    def test_k_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            encode_params(DISCRETE(k=10, P=5.0), N=8)

    def test_multi_discrete_wrong_type_raises(self):
        with pytest.raises(TypeError, match="DISCRETE"):
            MULTI_DISCRETE(vectors=[SINE(n=1)])

    def test_multi_sine_wrong_type_raises(self):
        with pytest.raises(TypeError, match="SINE"):
            MULTI_SINE(modes=[DISCRETE(k=1)])


# ===================================================================
# encode_vector tests — with declared vector_type
# ===================================================================

class TestEncodeVectorWithType:

    def test_discrete(self):
        f = np.zeros(8); f[3] = 5.0
        circuit, info = encode_vector(f, vector_type="DISCRETE")
        assert info.vector_type == "DISCRETE"
        assert info.N == 8
        assert not info.validated
        assert_encodes(circuit, f)

    def test_uniform(self):
        f = np.full(8, 3.0)
        circuit, info = encode_vector(f, vector_type="UNIFORM")
        assert_encodes(circuit, f)

    def test_step(self):
        f = np.zeros(8); f[:4] = 2.0
        circuit, info = encode_vector(f, vector_type="STEP")
        assert_encodes(circuit, f)

    def test_square(self):
        f = np.zeros(16); f[8:16] = 1.0
        circuit, info = encode_vector(f, vector_type="SQUARE")
        assert_encodes(circuit, f)

    def test_sine(self):
        N = 64; k = np.arange(N)
        f = np.sin(1 * np.pi * k / N)
        circuit, info = encode_vector(f, vector_type="SINE")
        assert info.vector_type == "SINE"
        assert not info.validated

    def test_cosine(self):
        N = 64; k = np.arange(N)
        f = np.cos(1 * np.pi * k / N)
        circuit, info = encode_vector(f, vector_type="COSINE")
        assert info.vector_type == "COSINE"

    def test_multi_discrete(self):
        f = np.zeros(8); f[1] = 2.0; f[5] = 3.0; f[7] = 1.0
        circuit, info = encode_vector(f, vector_type="MULTI_DISCRETE")
        assert_encodes(circuit, f)

    def test_multi_sine(self):
        N = 16; k = np.arange(N)
        f = 2.0 * np.sin(1 * np.pi * k / N) + np.sin(3 * np.pi * k / N)
        circuit, info = encode_vector(f, vector_type="MULTI_SINE")
        assert info.vector_type == "MULTI_SINE"

    def test_enum_value(self):
        f = np.zeros(4); f[2] = 1.0
        _, info = encode_vector(f, vector_type=VectorType.DISCRETE)
        assert info.vector_type == "DISCRETE"

    def test_old_name_compat(self):
        """Old name POINT_LOAD still works."""
        f = np.zeros(8); f[3] = 5.0
        _, info = encode_vector(f, vector_type="POINT_LOAD")
        assert info.vector_type == "DISCRETE"

    def test_n_inferred_from_vector(self):
        f = np.zeros(16); f[7] = 3.0
        _, info = encode_vector(f, vector_type="DISCRETE")
        assert info.N == 16
        assert info.m == 4

    # ── error cases ───────────────────────────────────────────────

    def test_wrong_type_raises(self):
        f = np.full(8, 3.0)
        with pytest.raises(ValueError, match="DISCRETE"):
            encode_vector(f, vector_type="DISCRETE")

    def test_non_power_of_two_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            encode_vector(np.zeros(6), vector_type="DISCRETE")

    def test_sine_mismatch_raises(self):
        f = np.zeros(8); f[1] = 1.0; f[5] = 1.0
        with pytest.raises(ValueError):
            encode_vector(f, vector_type="SINE")


# ===================================================================
# encode_vector tests — auto-detect (no vector_type)
# ===================================================================

class TestEncodeVectorAutoDetect:

    def test_auto_discrete(self):
        f = np.zeros(8); f[3] = 5.0
        circuit, info = encode_vector(f)
        assert info.vector_type == "DISCRETE"
        assert_encodes(circuit, f)

    def test_auto_uniform(self):
        f = np.full(8, 3.0)
        circuit, info = encode_vector(f)
        assert info.vector_type == "UNIFORM"
        assert_encodes(circuit, f)

    def test_auto_step(self):
        f = np.zeros(8); f[:4] = 2.0
        circuit, info = encode_vector(f)
        assert info.vector_type == "STEP"
        assert_encodes(circuit, f)

    def test_auto_square(self):
        f = np.zeros(16); f[4:12] = 1.0
        circuit, info = encode_vector(f)
        assert info.vector_type == "SQUARE"
        assert_encodes(circuit, f)

    def test_auto_sine(self):
        N = 64; k = np.arange(N)
        f = np.sin(1 * np.pi * k / N)
        circuit, info = encode_vector(f)
        assert info.vector_type == "SINE"
        assert not info.validated

    def test_auto_cosine(self):
        N = 64; k = np.arange(N)
        f = np.cos(1 * np.pi * k / N)
        circuit, info = encode_vector(f)
        assert info.vector_type == "COSINE"
        assert not info.validated

    def test_auto_multi_discrete(self):
        f = np.zeros(8); f[1] = 2.0; f[5] = 3.0; f[7] = 1.0
        circuit, info = encode_vector(f)
        assert info.vector_type == "MULTI_DISCRETE"
        assert_encodes(circuit, f)

    def test_auto_multi_sine(self):
        N = 16; k = np.arange(N)
        f = 2.0 * np.sin(1 * np.pi * k / N) + np.sin(3 * np.pi * k / N)
        circuit, info = encode_vector(f)
        assert info.vector_type == "MULTI_SINE"
        assert not info.validated

    def test_auto_prefers_simpler(self):
        """Point load at k=0 should be detected as DISCRETE, not STEP."""
        f = np.zeros(8); f[0] = 1.0
        _, info = encode_vector(f)
        assert info.vector_type == "DISCRETE"

    def test_auto_random_raises(self):
        """Truly random vector should fail auto-detection."""
        rng = np.random.default_rng(42)
        f = rng.standard_normal(8)
        with pytest.raises(ValueError, match="No known"):
            encode_vector(f)


# ===================================================================
# encode_python tests — AST path
# ===================================================================

class TestEncodePythonAST:

    def test_discrete(self):
        code = "N = 8\nf = np.zeros(N)\nf[3] = 5.0"
        circuit, info = encode_python(code)
        assert info.vector_type == "DISCRETE"
        expected = np.zeros(8); expected[3] = 5.0
        assert_encodes(circuit, expected)

    def test_uniform(self):
        circuit, info = encode_python("N = 4\nf = np.ones(N)")
        assert info.vector_type == "UNIFORM"
        assert_encodes(circuit, np.ones(4))

    def test_step(self):
        circuit, info = encode_python("N = 8\nf = np.zeros(N)\nf[:4] = 1.0")
        assert info.vector_type == "STEP"

    def test_sine(self):
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.sin(np.pi * x)"
        circuit, info = encode_python(code)
        assert info.vector_type == "SINE"

    def test_cosine(self):
        code = "N = 64\nx = np.linspace(0, 1, N)\nf = np.cos(np.pi * x)"
        circuit, info = encode_python(code)
        assert info.vector_type == "COSINE"

    def test_multi_discrete(self):
        code = "N = 8\nf = np.zeros(N)\nf[1] = 3.0\nf[6] = 4.0"
        circuit, info = encode_python(code)
        assert info.vector_type == "MULTI_DISCRETE"

    def test_fallback_with_warning(self):
        """Unrecognised pattern falls back to Shende with a warning."""
        code = "import numpy as np\nN = 4\nf = np.array([0.5, 0.3, 0.7, 0.1])"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            circuit, info = encode_python(code)
            assert len(w) >= 1
            assert "Shende" in str(w[0].message)
        assert info.vector_type == "UNKNOWN"

    def test_callable_source(self):
        """Callable code input via inspect.getsource."""
        def my_load():
            N = 8
            f = np.zeros(N)
            f[3] = 5.0

        circuit, info = encode_python(my_load)
        assert info.vector_type == "DISCRETE"

    def test_circuit_code_produced(self):
        cases = [
            "N=8\nf=np.zeros(8)\nf[3]=1.0",
            "N=8\nf=np.ones(8)",
            "N=8\nf=np.zeros(8)\nf[:4]=1.0",
            "N=8\nx=np.linspace(0,1,8)\nf=np.sin(np.pi*x)",
            "N=8\nx=np.linspace(0,1,8)\nf=np.cos(np.pi*x)",
            "N=8\nf=np.zeros(8)\nf[1]=1.0\nf[6]=2.0",
        ]
        for c in cases:
            _, info = encode_python(c)
            assert info.circuit_code != "", f"Empty circuit_code for: {c!r}"
            assert "qc" in info.circuit_code


# ===================================================================
# encode_python tests — with vector_type hint (code execution path)
# ===================================================================

class TestEncodePythonWithHint:

    def test_for_loop_discrete(self):
        """For-loop producing a discrete load — AST can't parse this."""
        code = """
import numpy as np
N = 8
f = np.zeros(N)
for i in range(N):
    if i == 3:
        f[i] = 5.0
"""
        circuit, info = encode_python(code, vector_type="DISCRETE")
        assert info.vector_type == "DISCRETE"
        assert not info.validated
        expected = np.zeros(8); expected[3] = 5.0
        assert_encodes(circuit, expected)

    def test_list_comp_uniform(self):
        code = """
import numpy as np
N = 8
f = np.array([3.0 for _ in range(N)])
"""
        circuit, info = encode_python(code, vector_type="UNIFORM")
        assert info.vector_type == "UNIFORM"
        assert_encodes(circuit, np.full(8, 3.0))

    def test_zip_loop_multi_discrete(self):
        code = """
import numpy as np
N = 8
indices = [1, 5, 7]
values = [2.0, 3.0, 1.0]
f = np.zeros(N)
for idx, val in zip(indices, values):
    f[idx] = val
"""
        circuit, info = encode_python(code, vector_type="MULTI_DISCRETE")
        assert info.vector_type == "MULTI_DISCRETE"
        expected = np.zeros(8)
        expected[1] = 2.0; expected[5] = 3.0; expected[7] = 1.0
        assert_encodes(circuit, expected)

    def test_sine_for_loop(self):
        code = """
import numpy as np
N = 64
f = np.zeros(N)
for k in range(N):
    f[k] = np.sin(1 * np.pi * k / N)
"""
        circuit, info = encode_python(code, vector_type="SINE")
        assert info.vector_type == "SINE"
        assert not info.validated

    def test_wrong_type_raises(self):
        code = "import numpy as np\nN = 8\nf = np.ones(N) * 3.0"
        with pytest.raises(ValueError, match="DISCRETE"):
            encode_python(code, vector_type="DISCRETE")

    def test_no_f_variable_raises(self):
        code = "import numpy as np\nx = np.zeros(8)"
        with pytest.raises(RuntimeError, match="does not define variable 'f'"):
            encode_python(code, vector_type="DISCRETE")

    def test_non_power_of_two_raises(self):
        code = "import numpy as np\nf = np.zeros(6)"
        with pytest.raises(ValueError, match="not a power of 2"):
            encode_python(code, vector_type="DISCRETE")


# ===================================================================
# Recognizer tests (verify AST recognizer still works with new names)
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
# Display / string tests
# ===================================================================

class TestDisplay:

    def test_info_str_contains_vector_type(self):
        _, info = encode_params(DISCRETE(k=0, P=1.0), N=4)
        s = str(info)
        assert "DISCRETE" in s
        assert "PyEncode" in s

    def test_circuit_is_qiskit(self):
        circuit, _ = encode_params(DISCRETE(k=1, P=1.0), N=4)
        assert isinstance(circuit, QuantumCircuit)


# ===================================================================
# Constructor object tests
# ===================================================================

class TestConstructors:

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
