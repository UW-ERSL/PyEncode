"""
pyencode.encode_params
======================
Direct parameter specification via typed constructors.
"""

from typing import Union

from qiskit import QuantumCircuit

from .recognizer import VectorType, LoadPattern
from .types import _VectorObj, EncodingInfo
from ._helpers import (
    _normalise_vector_type,
    _validate_params,
    _synthesise_and_build_info,
    _encode_composite,
)


def encode_params(
    vector_obj: Union[_VectorObj, list, "VectorType", str],
    N: int,
    *,
    validate: bool = False,
    tol: float = 1e-6,
    # Backward-compatible kwargs for string/enum usage
    **params,
) -> tuple[QuantumCircuit, EncodingInfo]:
    """
    Synthesise a circuit from a typed vector constructor and parameters.

    Parameters
    ----------
    vector_obj : VectorObj, list, VectorType, or str
        A typed constructor (e.g. ``SINE(n=3, A=1.0)``), a list of
        constructors for composite vectors, or a VectorType enum / string
        for backward compatibility.
    N : int
        Vector length (must be a power of 2).
    validate : bool
        If True, simulate the circuit and verify the output state.
    tol : float
        Tolerance for validation.
    **params
        Type-specific parameters (only for string/enum backward compat).

    Examples
    --------
    >>> circuit, info = encode_params(DISCRETE(k=3, P=5.0), N=8)
    >>> circuit, info = encode_params(SINE(n=3, A=1.0), N=64)
    >>> circuit, info = encode_params(
    ...     [SINE(n=1, A=2.0), SINE(n=3, A=1.0)], N=64)
    """
    if N < 2 or (N & (N - 1)) != 0:
        raise ValueError(f"N={N} is not a power of 2.")

    # Handle typed constructor objects
    if isinstance(vector_obj, _VectorObj):
        vtype = vector_obj.vector_type
        validated_params = _validate_params(vtype, N, vector_obj.params)
        pattern = LoadPattern(vtype, N=N, params=validated_params)
        return _synthesise_and_build_info(
            pattern, fallback_vector=None,
            validate=validate, tol=tol,
        )

    # Handle list of constructors (composite)
    if isinstance(vector_obj, list):
        return _encode_composite(vector_obj, N, validate=validate, tol=tol)

    # Backward compat: string or VectorType enum
    vtype = _normalise_vector_type(vector_obj)
    validated_params = _validate_params(vtype, N, params)
    pattern = LoadPattern(vtype, N=N, params=validated_params)
    return _synthesise_and_build_info(
        pattern, fallback_vector=None,
        validate=validate, tol=tol,
    )
