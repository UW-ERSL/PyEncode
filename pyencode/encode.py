"""
pyencode.encode
===============
Single entry point for structured quantum state preparation.
"""

from typing import Union
from qiskit import QuantumCircuit

from .recognizer import LoadPattern, PatternKind
from .types import _Pattern, EncodingInfo
from ._helpers import (
    _validate_params,
    _synthesize_and_build_info,
    _encode_composite,
    _encode_sum,
    _normalize_kind,
)


def encode(pattern, N: int, validate: bool = False, tol: float = 1e-6):
    """
    encode(pattern, N, validate=False, tol=1e-6)

    Prepare a structured quantum state from a typed parameter declaration.
    Maps directly to a closed-form Qiskit circuit with no vector
    materialization and no approximation.

    Parameters
    ----------
    pattern : _Pattern instance or list of _Pattern instances
        A typed constructor:
          SPARSE([(x1,a1), (x2,a2), ...])
          STEP(k_e, c)
          SQUARE(k_s, k_e, c)
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
    >>> circuit, info = encode(STEP(k_e=4, c=1.0), N=8)
    >>> circuit, info = encode(SQUARE(k_s=2, k_e=6, c=1.0), N=8)
    >>> circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=16)
    >>> circuit, info = encode([SQUARE(k_s=0, k_e=16, c=1.0),
    ...                         SQUARE(k_s=16, k_e=24, c=4.0)], N=32)
    """
    if isinstance(pattern, list):
        return _encode_composite(pattern, N, validate=validate, tol=tol)

    from .types import SUM as _SUM
    if isinstance(pattern, _SUM):
        return _encode_sum(pattern, N, validate=validate, tol=tol)

    from .types import TENSOR as _TENSOR
    if isinstance(pattern, _TENSOR):
        from ._helpers import _encode_tensor
        return _encode_tensor(pattern, N, validate=validate, tol=tol)

    from .types import PARTITION as _PARTITION
    if isinstance(pattern, _PARTITION):
        from ._helpers import _encode_partition
        return _encode_partition(pattern, N, validate=validate, tol=tol)

    if not isinstance(pattern, _Pattern):
        raise TypeError(
            f"pattern must be a typed constructor (SPARSE, STEP, SQUARE, "
            f"FOURIER), got {type(pattern).__name__}."
        )

    vtype = pattern.kind
    validated_params = _validate_params(vtype, N, pattern.params)
    pattern = LoadPattern(vtype, N=N, params=validated_params)
    return _synthesize_and_build_info(
        pattern,
        validate=validate, tol=tol,
    )
