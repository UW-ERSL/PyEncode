# predict_gates — Fast cost estimation for design optimization

## Motivation

In design-optimization workflows (solver tuning, pattern selection), you often need to evaluate *many* candidate encodings before committing to a circuit. A single `encode()` call takes 100+ ms at large m due to Qiskit's transpile pass; in an outer optimization loop with thousands of candidates, this dominates runtime.

`predict_gates(VectorObj, N)` returns the same transpiled gate counts — sub-millisecond, without any circuit construction — using closed-form formulas derived from the patterns' analytical structure and validated against `encode()`.

## Basic use

```python
from pyencode import predict_gates, POLYNOMIAL, GEOMETRIC, FOURIER

# Exact prediction, 500-8000x faster than encode()
p = predict_gates(POLYNOMIAL(coeffs=[0.0, 1.0]), N=4096)
print(p)
# {'vector_type': 'POLYNOMIAL', 'N': 4096, 'm': 12,
#  'gate_count_1q': 56, 'gate_count_2q': 22, 'gate_count': 78,
#  'circuit_depth': 45, 'complexity': 'O(m)', 'exact': True}
```

## Accuracy contract

Every prediction carries an `exact` flag:

- **`exact=True`**: prediction matches `encode()`'s transpiled counts to the gate. Guaranteed for POPCOUNT, WALSH, STAIRCASE, STEP, SPARSE (s=1), FOURIER (T=1), POLYNOMIAL (d=1), and SQUARE (aligned intervals).
- **`exact=False`**: prediction is an empirical fit or asymptotic upper bound. Still useful for coarse selection; within a few percent for POLYNOMIAL d=2 and SQUARE general, more conservative for SPARSE s≥2.

Cross-checked in `test_pyencode.py::TestPredictor` (14 tests against ground-truth `encode()` output).

## Design-optimization example

Evaluating thousands of polynomial source-term candidates in a topology-optimization loop:

```python
import numpy as np
from pyencode import predict_gates, POLYNOMIAL

best_gate_count = float('inf')
best_degree = None

for d in range(1, 6):
    # generate random polynomial coefficients for degree d
    for trial in range(1000):
        coeffs = np.random.randn(d + 1).tolist()
        p = predict_gates(POLYNOMIAL(coeffs=coeffs), N=4096)
        if p["gate_count"] < best_gate_count:
            best_gate_count = p["gate_count"]
            best_degree = d

# Only now, build the actual circuit for the best candidate:
from pyencode import encode
circuit, info = encode(POLYNOMIAL(coeffs=best_coeffs), N=4096)
```

The inner loop runs 5,000 predictions in under a second. Without `predict_gates`, it would take minutes.

## Composite constructors

`predict_gates` handles TENSOR, SUM, and PARTITION by recursing into components:

```python
from pyencode import predict_gates, TENSOR, SUM, PARTITION, FOURIER, SQUARE, SPARSE, GEOMETRIC

# TENSOR sums over disjoint subregisters
obj = TENSOR([(FOURIER(modes=[(2, 1.0, 0)]), 32),
              (FOURIER(modes=[(3, 1.0, 0)]), 32)])
p = predict_gates(obj, N=32*32)    # m=10, gate_count ~119

# SUM: weighted superposition (ancilla-based via the LCU technique)
obj = SUM([(0.5, SQUARE(k1=0, k2=8, c=1.0)),
           (0.5, SQUARE(k1=8, k2=16, c=1.0))])
p = predict_gates(obj, N=16)

# PARTITION: disjoint-support composition (ancilla-free, p = 1)
obj = PARTITION([SPARSE([(2, 0.3), (5, 0.5), (7, 0.7)]),
                 GEOMETRIC(ratio=0.8, start=11)])
p = predict_gates(obj, N=256)
```

## When to use encode() instead

- You need the Qiskit circuit itself (to simulate, transpile to hardware, run experiments)
- You need `info.circuit_code` (the emitted source snippet)
- You need statevector validation (`validate=True`)

`predict_gates` is for the selection phase; `encode` is for the commit phase.
