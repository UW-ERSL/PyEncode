"""
pyencode.encode_vector
======================
Vector input with optional type hint / auto-detect.
"""

from typing import Optional, Union

import numpy as np
from qiskit import QuantumCircuit

from .recognizer import VectorType, LoadPattern
from .extractor import extract, auto_detect
from .types import EncodingInfo
from ._helpers import (
    _normalize_vector_type,
    _synthesize_and_build_info,
    _build_expected_vector,
)


def encode_vector(
    f: np.ndarray,
    *,
    vector_type: Optional[Union[VectorType, str]] = None,
    validate: bool = False,
    tol: float = 1e-6,
) -> tuple[QuantumCircuit, EncodingInfo]:
    """
    Synthesise a circuit from an explicit load vector.

    If ``vector_type`` is given, parameters are extracted numerically
    and verified against the declared type.  If omitted, the vector
    type is auto-detected.

    Parameters
    ----------
    f : np.ndarray
        Load vector of length N (must be a power of 2).
    vector_type : VectorType or str, optional
        Declared vector type.  If None, auto-detected.
    validate : bool
        If True, simulate and verify.
    tol : float
        Tolerance for extraction and validation.

    Examples
    --------
    >>> f = np.zeros(8); f[3] = 5.0
    >>> circuit, info = encode_vector(f)                          # auto-detect
    >>> circuit, info = encode_vector(f, vector_type="DISCRETE")  # with hint
    """
    f = np.asarray(f, dtype=float).ravel()
    N = len(f)

    if N < 2:
        raise ValueError(f"Vector length {N} is too small, need >= 2.")
    if N & (N - 1) != 0:
        raise ValueError(f"Vector length {N} is not a power of 2.")

    if vector_type is not None:
        # Declared type: extract params for that specific type
        vector_type = _normalize_vector_type(vector_type)
        params = extract(f, vector_type, tol=tol)
    else:
        # Auto-detect
        vector_type, params = auto_detect(f, tol=tol)

    pattern = LoadPattern(vector_type, N=N, params=params)

    # Verify reconstruction
    expected = _build_expected_vector(pattern, fallback_vector=None)
    if expected is not None:
        norm_f = np.linalg.norm(f)
        norm_expected = np.linalg.norm(expected)
        if norm_f > 1e-14 and norm_expected > 1e-14:
            diff = np.linalg.norm(f / norm_f - expected / norm_expected)
            if diff > tol:
                raise ValueError(
                    f"Verification failed for vector_type={vector_type.name}: "
                    f"||f_normalized - f_reconstructed|| = {diff:.2e} > "
                    f"tol={tol}.  The vector may not match the declared type."
                )

    return _synthesize_and_build_info(
        pattern, fallback_vector=None,
        validate=validate, tol=tol,
    )
