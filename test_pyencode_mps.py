"""Tests for pyencode.mps."""

import numpy as np
import pytest

from pyencode.mps import encode_mps, encode_mps_from_tensors


# ============================================================
# Basic correctness
# ============================================================

def _prepared_state(qc, n_bond, m):
    """Extract the physical statevector after the bond is projected to |0>."""
    from qiskit.quantum_info import Statevector
    sv = Statevector(qc).data.reshape(2 ** m, 2 ** n_bond)
    sv_phys = sv[:, 0]
    p = float(np.linalg.norm(sv_phys) ** 2)
    if p > 0:
        sv_phys = sv_phys / np.linalg.norm(sv_phys)
    return sv_phys, p


def _resolve_phase(target, prepared):
    phase = np.vdot(target, prepared)
    if abs(phase) < 1e-15:
        return prepared
    return prepared * (np.conj(phase) / abs(phase))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("m", [3, 5, 7])
def test_random_vector_chi_full_is_exact(seed, m):
    """At chi >= 2^(m-1), MPS is exact for any vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(2 ** m)
    v /= np.linalg.norm(v)
    chi_full = 2 ** (m - 1)

    qc, info = encode_mps(v, bond_dim=chi_full, validate=True)
    n_bond = info.params["n_bond"]
    sv_phys, p = _prepared_state(qc, n_bond, m)
    assert abs(p - 1.0) < 1e-10, f"p_succ = {p}"
    sv_phys = _resolve_phase(v, sv_phys)
    assert np.linalg.norm(sv_phys - v) < 1e-10


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_low_entanglement_vector_is_exact_at_small_chi(seed):
    """A product state has chi=1 MPS rank; chi=1 should reproduce it exactly."""
    rng = np.random.default_rng(seed)
    m = 6
    # Build a product state
    factors = [rng.standard_normal(2) for _ in range(m)]
    factors = [f / np.linalg.norm(f) for f in factors]
    v = factors[0]
    for f in factors[1:]:
        v = np.kron(v, f)
    v /= np.linalg.norm(v)

    qc, info = encode_mps(v, bond_dim=1, validate=True)
    assert info.params["truncation_error_sq"] < 1e-20


def test_truncation_decreases_with_chi():
    """For a structured non-product vector, truncation error should drop with chi."""
    m = 8
    # Sparse head + geometric tail: low entanglement
    v = np.zeros(2 ** m)
    v[:6] = [1.0, 0.6, 0.4, 0.3, 0.25, 0.2]
    rho = 0.85
    for i in range(6, 2 ** m):
        v[i] = 0.15 * rho ** (i - 6)
    v /= np.linalg.norm(v)

    errs = []
    for chi in [1, 2, 4, 8, 16]:
        _, info = encode_mps(v, bond_dim=chi)
        errs.append(info.params["truncation_error_sq"])
    # Monotonic non-increasing
    for j in range(len(errs) - 1):
        assert errs[j + 1] <= errs[j] + 1e-15, f"chi grew yet error did not drop: {errs}"
    # Saturates well below 1e-10 once chi covers the structure
    assert errs[-1] < 1e-10


def test_p_succ_is_exactly_one():
    """The bond register must end in |0> deterministically."""
    rng = np.random.default_rng(7)
    m = 6
    v = rng.standard_normal(2 ** m)
    v /= np.linalg.norm(v)
    for chi in [2, 4, 8]:
        qc, info = encode_mps(v, bond_dim=chi)
        n_bond = info.params["n_bond"]
        _, p = _prepared_state(qc, n_bond, m)
        assert abs(p - 1.0) < 1e-10, f"chi={chi}: p_succ = {p}"


def test_complex_vector():
    rng = np.random.default_rng(3)
    m = 5
    v = rng.standard_normal(2 ** m) + 1j * rng.standard_normal(2 ** m)
    v /= np.linalg.norm(v)
    qc, info = encode_mps(v, bond_dim=2 ** (m - 1), validate=True)
    assert info.gate_count_2q is not None


def test_zero_padding_for_non_power_of_2_length():
    """Non-power-of-2 input is zero-padded to the next power of 2."""
    rng = np.random.default_rng(11)
    v = rng.standard_normal(100)
    v /= np.linalg.norm(v)
    qc, info = encode_mps(v, bond_dim=8, validate=True)
    assert info.N == 128
    assert info.m == 7


def test_returned_info_has_expected_fields():
    rng = np.random.default_rng(5)
    v = rng.standard_normal(64); v /= np.linalg.norm(v)
    qc, info = encode_mps(v, bond_dim=4)
    assert info.pattern_name == "MPS"
    assert info.N == 64
    assert info.m == 6
    assert info.params["bond_dim"] == 4
    assert info.params["n_bond"] == 2
    assert info.success_probability == 1.0
    # Transpilation should populate
    assert info.gate_count_2q is not None
    assert info.circuit_depth is not None


# ============================================================
# Pre-built tensors path
# ============================================================

def test_encode_mps_from_tensors_roundtrip():
    """vector -> tensors -> tensors -> circuit should match vector -> circuit."""
    from pyencode.mps import _vector_to_right_canonical_mps

    rng = np.random.default_rng(42)
    m = 5
    v = rng.standard_normal(2 ** m); v /= np.linalg.norm(v)

    tensors, _ = _vector_to_right_canonical_mps(v, m, chi_max=2 ** (m - 1))
    # Renormalise leftmost tensor — what encode_mps does internally
    nrm = np.linalg.norm(tensors[0])
    tensors = [tensors[0] / nrm] + tensors[1:]

    qc, info = encode_mps_from_tensors(tensors)
    n_bond = info.params["n_bond"]
    sv_phys, p = _prepared_state(qc, n_bond, m)
    sv_phys = _resolve_phase(v, sv_phys)
    assert abs(p - 1.0) < 1e-10
    assert np.linalg.norm(sv_phys - v) < 1e-10


def test_encode_mps_from_tensors_rejects_bad_bonds():
    A = np.zeros((2, 2, 3))   # left-bond should be 1
    with pytest.raises(ValueError):
        encode_mps_from_tensors([A])


# ============================================================
# Error paths
# ============================================================

def test_zero_vector_raises():
    with pytest.raises(ValueError):
        encode_mps(np.zeros(8), bond_dim=2)


def test_negative_bond_dim_raises():
    with pytest.raises(ValueError):
        encode_mps(np.array([1.0, 0.0]), bond_dim=0)


def test_validation_fails_loudly_when_chi_too_small():
    """A high-entanglement random vector at chi=1 should fail validation."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(64); v /= np.linalg.norm(v)
    with pytest.raises(RuntimeError):
        encode_mps(v, bond_dim=1, validate=True, tol=1e-8)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
