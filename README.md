# PyEncode

An open-source Python library for structured quantum state preparation.

PyEncode maps typed parameter declarations directly to exact, closed-form
Qiskit circuits — no vector materialization, no approximation.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and Qiskit 2.3+.

## Quick Start

```python
from pyencode import encode, SPARSE, STEP, SQUARE, FOURIER, WALSH, GEOMETRIC, HAMMING, STAIRCASE, TENSOR, POLYNOMIAL, SUM, PARTITION

# Basis vector at index 19 (Hamming weight 3 → 3 gates)
circuit, info = encode(SPARSE([(19, 1.0)]), N=64)

# Prefix uniform superposition [0, 4)
circuit, info = encode(STEP(k_s=4, c=1.0), N=8)

# General interval [2, 6) via Draper adder
circuit, info = encode(SQUARE(k1=2, k2=6, c=1.0), N=8)

# Sinusoidal mode via inverse QFT
circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=16)

# Two-level piecewise-constant state
circuit, info = encode(WALSH(k=1, c_pos=1.0, c_neg=4.0), N=8)

# Exponential decay — product state, zero two-qubit gates
circuit, info = encode(GEOMETRIC(ratio=0.95), N=64)

# Hamming-weight structured — product state, identical Ry per qubit, depth 1
circuit, info = encode(HAMMING(r=0.5), N=64)

# Sparse geometric staircase on unary indices — cascaded CR_y, O(m) gates
circuit, info = encode(STAIRCASE(r=0.5), N=16)

# Degree-d polynomial via Walsh-sparse loading
# Ramp:
circuit, info = encode(POLYNOMIAL(coeffs=[0.0, 1.0]), N=32)
# Poiseuille parabolic profile f(x) = 4x(1-x):
circuit, info = encode(POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]), N=32)

# Separable multi-dimensional state — disjoint subregisters, parallel execution
circuit, info = encode(
    TENSOR([(FOURIER(modes=[(1, 1.0, 0)]), 32),
            (FOURIER(modes=[(3, 1.0, 0)]), 32)]),
    N=32 * 32)

# Weighted superposition of patterns via SUM (implemented using the LCU technique)
circuit, info = encode(
    SUM([(1.0, SQUARE(k1=0, k2=8, c=1.0)),
         (3.0, SQUARE(k1=8, k2=16, c=1.0))]),
    N=16)

# Ancilla-free disjoint-support composition via PARTITION
circuit, info = encode(
    PARTITION([SPARSE([(2, 0.3), (5, 0.5), (7, 0.7)]),
               GEOMETRIC(ratio=0.8, start=11)]),
    N=256)
```

Every call returns `(circuit, info)` where `circuit` is a Qiskit
`QuantumCircuit` and `info` is an `EncodingInfo` dataclass with gate
counts, circuit depth, complexity class, and optionally the
statevector-validated amplitude vector.

## Supported Patterns

| Pattern    | Constructor                    | Complexity         | Source                       |
|------------|--------------------------------|--------------------|------------------------------|
| Sparse     | `SPARSE([(x,a), ...])`        | O(s·m)             | Gleinig & Hoefler (2021)     |
| Step       | `STEP(k_s, c)`                | O(m)               | Shukla & Vedula (2024)       |
| Square     | `SQUARE(k1, k2, c)`           | O(m²) / O(m)       | Shukla & Vedula + Draper     |
| Walsh      | `WALSH(k, c_pos, c_neg)`      | O(m)               | Welch et al. (2014)          |
| Geometric  | `GEOMETRIC(ratio, c)`         | O(m), 0 CX, depth 1| Xie & Ben-Ami (2025)        |
| Hamming    | `HAMMING(r, c)`              | O(m), 0 CX, depth 1| Product state (this work)    |
| Staircase  | `STAIRCASE(r, c)`             | O(m), O(m) CX      | Hackbusch (1999)             |
| Polynomial | `POLYNOMIAL(coeffs)`          | O(m^(d+1))         | Welch (2014), Gonzalez-Conde (2024) |
| Fourier    | `FOURIER(modes=[...])`        | O(m²)              | Gonzalez-Conde / Moosa       |
| Sum        | `SUM([(w, VectorObj), ...])`  | Σ component costs  | Childs & Wiebe (LCU, 2012)   |
| Partition  | `PARTITION([VectorObj, ...])` | O(L·m), ancilla-free | Bentley & Saxe (1980) + Gleinig & Hoefler (2021) |
| Tensor     | `TENSOR([(VectorObj, N_i), ...])` | Σ component costs, depth = max | Composition rule (this work) |

Here m = log₂(N) is the number of qubits and N is the vector length.

## Validation

```python
circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=16,
                       validate=True, tol=1e-6)
# info.validated  → True
# info.vector     → numpy array of length N
```

Setting `validate=True` runs the circuit on Qiskit's statevector simulator
and checks against the classically constructed vector.
Requires O(2^m) memory; disabled by default.

## EncodingInfo Fields

Each `encode()` call returns an `EncodingInfo` with:

- `vector_type` — pattern name (e.g. `"SPARSE"`, `"GEOMETRIC"`)
- `N`, `m` — vector length and number of qubits
- `params` — supplied vector parameters (e.g. `{"ratio": 0.95, "c": 1.0}`)
- `gate_count` — total gates (pre-transpilation)
- `gate_count_1q`, `gate_count_2q` — U and CX counts after transpilation to {cx, u}
- `circuit_depth` — circuit depth after transpilation to {cx, u} (determines minimum execution time when gates on disjoint qubits run in parallel; may differ from the raw circuit depth visible via `print(circuit)`)
- `complexity` — asymptotic class (e.g. `"O(m)"`, `"O(m²)"`)
- `success_probability` — 1.0 for single patterns and PARTITION; p ∈ (0,1] for SUM with overlapping supports
- `circuit_code` — standalone Qiskit snippet reproducing the circuit
- `validated` — True if statevector validation was performed
- `vector` — amplitude vector (only when `validate=True`; requires O(2^m) memory)

## Standalone Circuit Code

Every circuit can be exported as a standalone Qiskit snippet:

```python
_, info = encode(SPARSE([(3, 1.0)]), N=8)
print(info.circuit_code)
# Prints runnable Python code that builds the same circuit
# without requiring PyEncode.
```

## Cost Prediction Without Synthesis

For design-optimization workflows, `predict_gates(VectorObj, N)` returns
transpiled gate counts analytically, without ever building a circuit.
It's typically 500–8000× faster than `encode()`, which makes it practical
to evaluate thousands of candidate encodings inside an outer optimization
loop before committing to synthesis.

```python
from pyencode import predict_gates, POLYNOMIAL

p = predict_gates(POLYNOMIAL(coeffs=[0.0, 1.0]), N=4096)
# {'vector_type': 'POLYNOMIAL', 'N': 4096, 'm': 12,
#  'gate_count_1q': 56, 'gate_count_2q': 22, 'gate_count': 78,
#  'circuit_depth': 45, 'complexity': 'O(m)', 'exact': True}
```

### Accuracy contract

Every prediction carries an `exact` flag:

- **`exact=True`** — prediction matches `encode()`'s transpiled counts to
  the gate. Guaranteed for HAMMING, WALSH, STAIRCASE, STEP, SPARSE (s=1),
  FOURIER (T=1), POLYNOMIAL (d=1), and SQUARE (aligned intervals).
- **`exact=False`** — prediction is an empirical fit or asymptotic upper
  bound. Still useful for coarse selection (within a few percent for
  POLYNOMIAL d=2 and SQUARE general; more conservative for SPARSE s≥2).

Cross-checked against `encode()` ground-truth in `test_pyencode.py::TestPredictor`.

### Design-optimization example

Evaluating thousands of polynomial source-term candidates:

```python
import numpy as np
from pyencode import predict_gates, encode, POLYNOMIAL

best_gate_count = float('inf')
best_coeffs = None

for d in range(1, 6):
    for _ in range(1000):
        coeffs = np.random.randn(d + 1).tolist()
        p = predict_gates(POLYNOMIAL(coeffs=coeffs), N=4096)
        if p["gate_count"] < best_gate_count:
            best_gate_count = p["gate_count"]
            best_coeffs = coeffs

# Build the actual circuit only for the best candidate
circuit, info = encode(POLYNOMIAL(coeffs=best_coeffs), N=4096)
```

The inner loop runs 5,000 predictions in under a second. Without
`predict_gates`, the same selection step would take minutes.

### Composite constructors

`predict_gates` handles `TENSOR`, `SUM`, and `PARTITION` by recursing
into components:

```python
from pyencode import predict_gates, TENSOR, SUM, PARTITION, FOURIER, SQUARE, SPARSE, GEOMETRIC

# TENSOR sums over disjoint subregisters
p = predict_gates(TENSOR([(FOURIER(modes=[(2, 1.0, 0)]), 32),
                          (FOURIER(modes=[(3, 1.0, 0)]), 32)]),
                  N=32*32)

# SUM: weighted superposition (ancilla-based via the LCU technique)
p = predict_gates(SUM([(0.5, SQUARE(k1=0, k2=8, c=1.0)),
                       (0.5, SQUARE(k1=8, k2=16, c=1.0))]),
                  N=16)

# PARTITION: disjoint-support composition (ancilla-free, p = 1)
p = predict_gates(PARTITION([SPARSE([(2, 0.3), (5, 0.5), (7, 0.7)]),
                             GEOMETRIC(ratio=0.8, start=11)]),
                  N=256)
```

### When to use `encode()` instead

`predict_gates` is for the selection phase; `encode` is for the commit
phase. Use `encode()` when you need:

- the Qiskit circuit itself, to simulate, transpile to hardware, or run
  experiments;
- the emitted source snippet in `info.circuit_code`;
- statevector validation via `validate=True`.

## Testing

```bash
python -m pytest test_pyencode.py -v
```

## Paper

The paper and experiment notebooks are in `notebooks/paperExperiments.ipynb`.
Figures can be regenerated with:

```bash
python generate_figures.py
```

## Citation

If you use PyEncode in your work, please cite:

```
K. Suresh and S. Suresh, "PyEncode: An Open-Source Library for
Structured Quantum State Preparation," 2026.
```

## License

University of Wisconsin–Madison.