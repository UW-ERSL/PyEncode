"""
pyencode.encode
===============
Single entry point for structured quantum state preparation.
"""

from typing import Union
from qiskit import QuantumCircuit

from .recognizer import LoadPattern, VectorType
from .types import _VectorObj, EncodingInfo
from ._helpers import (
    _validate_params,
    _synthesize_and_build_info,
    _encode_composite,
    _encode_lcu,
    _normalize_vector_type,
)


def encode(VectorObj, N: int, validate: bool = False, tol: float = 1e-6):
    """
    encode(VectorObj, N, validate=False, tol=1e-6)

    Prepare a structured quantum state from a typed parameter declaration.
    Maps directly to a closed-form Qiskit circuit with no vector
    materialization and no approximation.

    Parameters
    ----------
    VectorObj : _VectorObj instance or list of _VectorObj instances
        A typed constructor:
          SPARSE([(x1,a1), (x2,a2), ...])
          STEP(k_s, c)
          SQUARE(k1, k2, c)
          FOURIER(modes=[(n, A, phi), ...])
        Pass a list of SQUARE constructors for composite piecewise-constant
        vectors (e.g. Fermi-Hubbard PREP).
    N : int
        Vector length. Must be a power of 2.
    validate : bool, optional
        If True, simulate the circuit and verify the output statevector.
        Disabled by default (requires O(2^m) memory).
    tol : float, optional
        Tolerance for statevector validation. Default 1e-6.

    Returns
    -------
    (QuantumCircuit, EncodingInfo)

    Examples
    --------
    >>> from pyencode import encode, SPARSE, STEP, SQUARE, FOURIER
    >>> circuit, info = encode(SPARSE([(19, 1.0)]), N=64)
    >>> circuit, info = encode(STEP(k_s=4, c=1.0), N=8)
    >>> circuit, info = encode(SQUARE(k1=2, k2=6, c=1.0), N=8)
    >>> circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=16)
    >>> circuit, info = encode([SQUARE(k1=0, k2=16, c=1.0),
    ...                         SQUARE(k1=16, k2=24, c=4.0)], N=32)
    """
    if isinstance(VectorObj, list):
        return _encode_composite(VectorObj, N, validate=validate, tol=tol)

    from .types import LCU as _LCU
    if isinstance(VectorObj, _LCU):
        return _encode_lcu(VectorObj, N, validate=validate, tol=tol)

    if not isinstance(VectorObj, _VectorObj):
        raise TypeError(
            f"VectorObj must be a typed constructor (SPARSE, STEP, SQUARE, "
            f"FOURIER), got {type(VectorObj).__name__}."
        )

    vtype = VectorObj.vector_type
    validated_params = _validate_params(vtype, N, VectorObj.params)
    pattern = LoadPattern(vtype, N=N, params=validated_params)
    return _synthesize_and_build_info(
        pattern,
        validate=validate, tol=tol,
    )
