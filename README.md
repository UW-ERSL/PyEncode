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
from pyencode import encode, SPARSE, STEP, SQUARE, FOURIER, WALSH, GEOMETRIC, LCU

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

# Weighted superposition of patterns via LCU
circuit, info = encode(
    LCU([(1.0, SQUARE(k1=0, k2=8, c=1.0)),
         (3.0, SQUARE(k1=8, k2=16, c=1.0))]),
    N=16)
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
| Fourier    | `FOURIER(modes=[...])`        | O(m²)              | Gonzalez-Conde / Moosa       |
| LCU        | `LCU([(w, VectorObj), ...])`  | Σ component costs  | Childs / Babbush             |

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
- `success_probability` — 1.0 for single patterns; p ∈ (0,1] for LCU
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