"""
pyencode
========
Open-source library for structured quantum state preparation.

PyEncode maps typed parameter declarations directly to exact, closed-form
Qiskit circuits — no vector materialization, no approximation.

Four vector patterns are supported:

  SPARSE   :  Gleinig-Hoefler s-sparse state        O(s·m)
  STEP     :  Prefix uniform superposition [0, k_s)  O(m)
  SQUARE   :  Interval uniform superposition [k1,k2) O(m)
  FOURIER  :  T sinusoidal modes via inverse QFT     O(m²)

Usage
-----
>>> from pyencode import encode, SPARSE, STEP, SQUARE, FOURIER
>>> circuit, info = encode(SPARSE([(19, 1.0)]), N=64)
>>> circuit, info = encode(STEP(k_s=4, c=1.0), N=8)
>>> circuit, info = encode(SQUARE(k1=2, k2=6, c=1.0), N=8)
>>> circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=16)

References
----------
  Gleinig & Hoefler, DAC 2021.
  Shukla & Vedula, 2024.
  Gonzalez-Conde et al., 2024.
  Moosa et al., 2024.
"""

from .types import (
    _VectorObj,
    SPARSE,
    FOURIER,
    WALSH,
    STEP,
    SQUARE,
    LCU,
    EncodingInfo,
)
from .recognizer import VectorType
from .encode_params import encode

__all__ = [
    "encode",
    "EncodingInfo",
    "VectorType",
    # Constructors
    "SPARSE",
    "FOURIER",
    "WALSH",
    "LCU",
    "STEP",
    "SQUARE",
]
__version__ = "1.2.0"
