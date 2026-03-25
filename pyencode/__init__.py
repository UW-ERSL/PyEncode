"""
pyencode
==========
A toolkit for quantum amplitude encoding of structured load vectors.

PyEncode synthesizes efficient Qiskit quantum circuits for structured
state vectors (point loads, sinusoidal modes, step functions, etc.)
using closed-form circuit templates instead of general O(2^m) routines.

Three entry points
------------------

**encode** — the primary API (paper name for encode_params):

>>> circuit, info = encode(SPARSE([(19, 1.0)]), N=64)
>>> circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=16)

**encode_params** — identical to encode(), legacy name:

>>> circuit, info = encode_params(DISCRETE(k=3, P=5.0), N=8)
>>> circuit, info = encode_params(SINE(n=3, A=1.0), N=64)

**encode_vector** — pass a vector, optionally declare the type:

>>> f = np.zeros(8); f[3] = 5.0
>>> circuit, info = encode_vector(f)                        # auto-detect
>>> circuit, info = encode_vector(f, vector_type="DISCRETE") # with hint

**encode_python** — compile Python source code into a circuit:

>>> circuit, info = encode_python('''
... import numpy as np
... N = 8; f = np.zeros(N); f[3] = 5.0
... ''')

Supported vector patterns
--------------------------
  SPARSE         :  Gleinig-Hoefler s-sparse state  (replaces DISCRETE + MULTI_DISCRETE)
  FOURIER        :  T sinusoidal modes via inverse QFT  (replaces SINE + COSINE + MULTI_SINE)
  DISCRETE       :  f[k] = P  (legacy; use SPARSE)
  UNIFORM        :  f = ones(N) * c
  STEP           :  f[:k_s] = c
  SQUARE         :  f[k1:k2] = c
  SINE           :  f = A * sin(n * pi * k / N + phi)
  COSINE         :  f = A * cos(n * pi * k / N + phi)
  MULTI_DISCRETE :  multiple disjoint point loads (Gleinig-Hoefler O(L·n))
  MULTI_SINE     :  sum of sinusoidal modes

References
----------
  Shende, Markov, Bullock, IEEE TCAD 25(6), 1000-1010, 2006.
  Gleinig & Hoefler, DAC 2021.
  Harrow, Hassidim, Lloyd, Phys. Rev. Lett. 103, 150502, 2009.
"""

from .recognizer import VectorType, LoadPattern, recognize, recognise
from .types import (
    LoadType,
    _VectorObj,
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
    EncodingInfo,
)
from .encode_params import encode_params, encode
from .encode_vector import encode_vector
from .encode_python import encode_python

__all__ = [
    "encode",
    "encode_params",
    "encode_vector",
    "encode_python",
    "EncodingInfo",
    "VectorType",
    "LoadType",
    "recognize",
    "recognise",   # backward-compatible alias
    # Constructor classes (new unified API)
    "SPARSE",
    "FOURIER",
    # Legacy constructors
    "DISCRETE",
    "UNIFORM",
    "STEP",
    "SQUARE",
    "SINE",
    "COSINE",
    "MULTI_DISCRETE",
    "MULTI_SINE",
]
__version__ = "0.6.0"
