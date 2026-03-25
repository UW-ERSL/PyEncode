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
    _normalize_vector_type,
    _validate_params,
    _synthesize_and_build_info,
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
        return _synthesize_and_build_info(
            pattern, fallback_vector=None,
            validate=validate, tol=tol,
        )

    # Handle list of constructors (composite)
    if isinstance(vector_obj, list):
        return _encode_composite(vector_obj, N, validate=validate, tol=tol)

    # Backward compat: string or VectorType enum
    vtype = _normalize_vector_type(vector_obj)
    validated_params = _validate_params(vtype, N, params)
    pattern = LoadPattern(vtype, N=N, params=validated_params)
    return _synthesize_and_build_info(
        pattern, fallback_vector=None,
        validate=validate, tol=tol,
    )


# ---------------------------------------------------------------------------
# Public alias: encode() is the paper API name for encode_params()
# ---------------------------------------------------------------------------

def encode(VectorObj, N, validate=False, tol=1e-6):
    """
    encode(VectorObj, N, validate=False, tol=1e-6)

    Single entry point for structured quantum state preparation.
    Maps a typed constructor directly to a verified Qiskit circuit.

    This is the primary API described in the paper. It is identical to
    encode_params() and is provided as the canonical name going forward.

    Parameters
    ----------
    VectorObj : _VectorObj instance or list of _VectorObj instances
        A typed constructor such as SPARSE([(3, 1.0)]), STEP(k_s=4, c=1.0),
        SQUARE(k1=2, k2=6, c=1.0), or FOURIER(modes=[(1, 1.0, 0)]).
        Pass a list for composite vectors.
    N : int
        Vector length. Must be a power of 2.
    validate : bool, optional
        If True, simulate the circuit and verify the output state.
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
    >>> circuit, info = encode(SPARSE([(1, 3.0), (6, 4.0)]), N=8)
    """
    return encode_params(VectorObj, N, validate=validate, tol=tol)
