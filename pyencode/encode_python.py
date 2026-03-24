"""
pyencode.encode_python
======================
Compile Python source code into a Qiskit circuit.
"""

import inspect
import textwrap
import warnings
from typing import Optional, Union

import numpy as np
from qiskit import QuantumCircuit

from .recognizer import recognize, VectorType, LoadPattern
from .extractor import extract
from .types import EncodingInfo
from ._helpers import (
    _execute_code,
    _normalize_vector_type,
    _synthesize_and_build_info,
    _build_expected_vector,
)


def encode_python(
    code,
    *,
    vector_type: Optional[Union[VectorType, str]] = None,
    N: Optional[int] = None,
    validate: bool = False,
    tol: float = 1e-6,
) -> tuple[QuantumCircuit, EncodingInfo]:
    """
    Compile Python source code into a Qiskit circuit.

    ``code`` accepts a string of valid Python source or a callable;
    in the latter case the source is retrieved automatically via
    ``inspect.getsource``.

    If ``vector_type`` is not given, the AST recognizer infers the
    pattern from the code structure.  If ``vector_type`` is given,
    the code is *executed* to produce a vector, then parameters
    are extracted numerically.

    Parameters
    ----------
    code : str or callable
        Python source that constructs a load vector named ``f``,
        or a callable whose source will be inspected.
    vector_type : VectorType or str, optional
        If given, bypass AST recognition: execute the code and
        extract parameters numerically for the declared type.
    N : int, optional
        Vector length hint (used when AST cannot determine N).
    validate : bool
        If True, simulate and verify.
    tol : float
        Tolerance for extraction and validation.

    Examples
    --------
    AST recognition (automatic):

    >>> circuit, info = encode_python('''
    ... import numpy as np
    ... N = 8; f = np.zeros(N); f[3] = 5.0
    ... ''')

    With type hint (executes code, handles loops etc.):

    >>> circuit, info = encode_python('''
    ... import numpy as np
    ... N = 8; f = np.zeros(N)
    ... for i in range(N):
    ...     if i == 3: f[i] = 5.0
    ... ''', vector_type="DISCRETE")
    """
    # Accept callable: retrieve source via inspect.getsource
    if callable(code) and not isinstance(code, str):
        code = textwrap.dedent(inspect.getsource(code))

    if vector_type is not None:
        # Path 2: Execute code -> get vector -> extract params
        vector_type = _normalize_vector_type(vector_type)
        f = _execute_code(code)

        vec_N = len(f)
        if vec_N < 2 or (vec_N & (vec_N - 1)) != 0:
            raise ValueError(
                f"Executed code produced vector of length {vec_N}, "
                f"which is not a power of 2."
            )

        params = extract(f, vector_type, tol=tol)
        pattern = LoadPattern(vector_type, N=vec_N, params=params)

        # Classical validation (mandatory on Path 2)
        expected = _build_expected_vector(pattern, fallback_vector=None)
        if expected is not None:
            norm_f = np.linalg.norm(f)
            norm_expected = np.linalg.norm(expected)
            if norm_f > 1e-14 and norm_expected > 1e-14:
                diff = np.linalg.norm(f / norm_f - expected / norm_expected)
                if diff > tol:
                    raise ValueError(
                        f"Verification failed for vector_type={vector_type.name}: "
                        f"the executed code does not produce a vector "
                        f"matching the declared type "
                        f"(normalized diff = {diff:.2e})."
                    )

        return _synthesize_and_build_info(
            pattern, fallback_vector=None,
            validate=validate, tol=tol,
        )

    # Path 1: AST recognition (no vector_type hint)
    pattern: LoadPattern = recognize(code)

    if pattern.load_type == VectorType.UNKNOWN:
        # Fallback: execute the code to get the vector, use Shende
        warnings.warn(
            "PyEncode: pattern not recognized by the AST recognizer. "
            "Falling back to Shende's O(2^m) general state preparation.",
            stacklevel=2,
        )
        try:
            f = _execute_code(code)
            vec_N = len(f)
            if vec_N < 2 or (vec_N & (vec_N - 1)) != 0:
                raise ValueError(
                    f"Executed code produced vector of length {vec_N}, "
                    f"which is not a power of 2."
                )
            pattern.N = vec_N
            pattern.params = {"amplitudes": f.astype(complex)}
        except Exception as e:
            raise RuntimeError(
                "PyEncode: load pattern not recognized and code execution "
                f"failed: {e}\n"
                "Either restructure the code to match a supported "
                "pattern or use vector_type= to declare the pattern."
            ) from e

    return _synthesize_and_build_info(
        pattern, fallback_vector=None,
        validate=validate, tol=tol,
    )
