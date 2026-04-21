"""
examples/partition_example.py
=============================
Demonstration of PARTITION: ancilla-free composition of bounded-support
patterns with pairwise-disjoint support.

The motivating scenario (from the paper discussion): N = 256 indices,
where the first ten indices are a handful of SPARSE point masses and
everything from index 11 onward follows a GEOMETRIC decay.

    f = [ x0, 0, x2, 0, 0, x5, 0, x7, 0, 0, 0,  |  GEOMETRIC tail  ]
        <------- SPARSE prefix, support = {2,5,7} ------|--- [11, 256) --->

PARTITION handles this in a single ancilla-free circuit with
success probability 1, whereas SUM (the ancilla-based counterpart)
would need an ancilla plus
controlled component circuits.

Run with:  python examples/partition_example.py
"""

import os
import sys

# Allow running from anywhere: add repo root (parent of this file's dir)
# to sys.path so `import pyencode` works without an install step.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from qiskit.quantum_info import Statevector

from pyencode import encode, PARTITION, SPARSE, GEOMETRIC, SUM


# ---------------------------------------------------------------------
# Problem: SPARSE prefix + GEOMETRIC tail on N = 256 indices
# ---------------------------------------------------------------------

N = 256
sparse_entries = [(2, 0.3), (5, 0.5), (7, 0.7)]    # only indices 2, 5, 7
geometric_ratio = 0.8
geometric_start = 11                                # tail begins at 11

components = [
    SPARSE(sparse_entries),
    GEOMETRIC(ratio=geometric_ratio, start=geometric_start),
]

# ---------------------------------------------------------------------
# Solution via PARTITION
# ---------------------------------------------------------------------

qc_p, info_p = encode(PARTITION(components), N=N, validate=True)

print("=" * 64)
print("PARTITION: disjoint-support composition")
print("=" * 64)
print(info_p)
print()
print(f"  atoms (singletons + dyadic blocks): L = {info_p.params['L']}")
print(f"  ancilla qubits:                      0")
print(f"  success probability:                 {info_p.success_probability}")

# ---------------------------------------------------------------------
# Reference: same state constructed via SUM (implemented as LCU:
# ancilla + PREP/SELECT)
# ---------------------------------------------------------------------

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")   # SUM may warn about overlap detection
    qc_l, info_l = encode(
        SUM([(1.0, SPARSE(sparse_entries)),
             (1.0, GEOMETRIC(ratio=geometric_ratio, start=geometric_start))]),
        N=N)

print()
print("=" * 64)
print("SUM baseline -- same disjoint state via LCU (ancilla-based)")
print("=" * 64)
print(info_l)

# ---------------------------------------------------------------------
# Speedup summary
# ---------------------------------------------------------------------

print()
print("=" * 64)
print("PARTITION vs SUM")
print("=" * 64)
print(f"  PARTITION gate count: {info_p.gate_count:>6d}")
print(f"  SUM       gate count: {info_l.gate_count:>6d}")
print(f"  ratio (SUM/PARTITION): {info_l.gate_count / info_p.gate_count:.2f}x")

# ---------------------------------------------------------------------
# Correctness check against analytical reference vector
# ---------------------------------------------------------------------

expected = np.zeros(N)
for k, v in sparse_entries:
    expected[k] = v
expected[geometric_start:] = geometric_ratio ** np.arange(N - geometric_start)
expected /= np.linalg.norm(expected)

sv = np.abs(np.array(Statevector(qc_p)))
max_error = np.max(np.abs(sv - np.abs(expected)))
print(f"\n  max state-vector error: {max_error:.2e}")
assert max_error < 1e-10, "PARTITION output does not match the analytical vector"
print("  ✓ Output matches the analytical target state.")

# ---------------------------------------------------------------------
# Bonus: overlap detection behavior
# ---------------------------------------------------------------------

print()
print("=" * 64)
print("Overlap detection")
print("=" * 64)
try:
    from pyencode import STEP, SQUARE
    encode(PARTITION([STEP(k_s=8, c=1.0),
                      SQUARE(k1=4, k2=12, c=1.0)]), N=16)
except ValueError as exc:
    print(f"  Expected: overlapping supports rejected.")
    print(f"  Error   : {exc}")
