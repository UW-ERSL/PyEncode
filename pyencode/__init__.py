"""
pyencode
========
Open-source library for structured quantum state preparation.

PyEncode maps typed parameter declarations directly to exact, closed-form
Qiskit circuits — no vector materialization, no approximation.

The exact pattern families are:

  SPARSE     :  Gleinig-Hoefler s-sparse state             O(s·m)
  STEP       :  Prefix uniform superposition [0, k_s)      O(m)
  SQUARE     :  Interval uniform superposition [k1, k2)    O(m²) / O(m) aligned
  WALSH      :  k-th Walsh function                        O(m)
  FOURIER    :  T sinusoidal modes via inverse QFT         O(m²)
  GEOMETRIC  :  Geometric product state c·rⁱ               O(m) / O(m²) offset
  HAMMING    :  c·r^{wt(i)} (Hamming-weight product state) O(m), depth 1
  STAIRCASE  :  r^k on unary indices 2^k - 1               O(m)
  DICKE      :  |D^m_k⟩ — uniform over weight-k indices    O(k·(m-k))
  POLYNOMIAL :  Degree-d polynomial via Walsh-sparse load  O(m^{d+1})

Composition rules:

  SUM        :  Weighted superposition via LCU              O(Σᵢ Cᵢ)
  PARTITION  :  Ancilla-free disjoint-support composition   O(L·m)
  TENSOR     :  Kronecker product over disjoint subregs     O(Σᵢ Cᵢ)

Usage
-----
>>> from pyencode import encode, SPARSE, STEP, SQUARE, FOURIER, DICKE
>>> circuit, info = encode(SPARSE([(19, 1.0)]), N=64)
>>> circuit, info = encode(STEP(k_s=4, c=1.0), N=8)
>>> circuit, info = encode(SQUARE(k1=2, k2=6, c=1.0), N=8)
>>> circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=16)
>>> circuit, info = encode(DICKE(k=2), N=16)

References
----------
  Gleinig & Hoefler, DAC 2021.
  Shukla & Vedula, 2024.
  Gonzalez-Conde et al., 2024.
  Moosa et al., 2024.
  Bärtschi & Eidenbenz, FCT 2019.
"""

from .types import (
    _Pattern,
    SPARSE,
    FOURIER,
    WALSH,
    STEP,
    SQUARE,
    GEOMETRIC,
    HAMMING,
    STAIRCASE,
    DICKE,
    TENSOR,
    POLYNOMIAL,
    SUM,
    LCU,              # deprecated alias for SUM; retained for backward compat
    PARTITION,
    EncodingInfo,
)
from .recognizer import PatternKind
from .encode import encode
from .predictor import predict_gates

__all__ = [
    "encode",
    "predict_gates",
    "EncodingInfo",
    "PatternKind",
    # Constructors
    "SPARSE",
    "FOURIER",
    "WALSH",
    "GEOMETRIC",
    "HAMMING",
    "STAIRCASE",
    "DICKE",
    "TENSOR",
    "POLYNOMIAL",
    "SUM",
    "LCU",            # deprecated alias; emits DeprecationWarning on use
    "PARTITION",
    "STEP",
    "SQUARE",
]
__version__ = "2.0.0"
