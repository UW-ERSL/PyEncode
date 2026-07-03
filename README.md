# PyEncode

An open-source Python library for structured quantum state preparation.

PyEncode maps typed parameter declarations directly to exact, closed-form
Qiskit circuits — no vector materialization, no approximation. For smooth
amplitude vectors that fall outside the exact pattern families, a
standalone matrix product state loader provides bounded-error approximate
encoding.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and Qiskit 2.3+.

## Quick Start

```python
from pyencode import (encode, SPARSE, STEP, SQUARE, FOURIER, WALSH,
                      GEOMETRIC, HAMMING, STAIRCASE, DICKE, POLYNOMIAL,
                      TENSOR, SUM, PARTITION)

# Basis vector at index 19 (Hamming weight 3 → 3 gates)
circuit, info = encode(SPARSE([(19, 1.0)]), N=64)

# Prefix uniform superposition [0, 4)
circuit, info = encode(STEP(k_e=4, c=1.0), N=8)

# General interval [2, 6) via Draper adder
circuit, info = encode(SQUARE(k_s=2, k_e=6, c=1.0), N=8)

# Sinusoidal mode via inverse QFT
circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=16)

# Two-level piecewise-constant state
circuit, info = encode(WALSH(k=1, c0=1.0, c1=4.0), N=8)

# Exponential decay — product state, zero two-qubit gates
circuit, info = encode(GEOMETRIC(r=0.95), N=64)

# Hamming-weight structured — product state, identical Ry per qubit, depth 1
circuit, info = encode(HAMMING(r=0.5), N=64)

# Sparse geometric staircase on unary indices — cascaded CR_y, O(m) gates
circuit, info = encode(STAIRCASE(r=0.5), N=16)

# Dicke state — uniform superposition over Hamming weight k
circuit, info = encode(DICKE(k=2), N=16)

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

# Weighted superposition of patterns via SUM (implemented using LCU)
circuit, info = encode(
    SUM([(1.0, SQUARE(k_s=0, k_e=8, c=1.0)),
         (3.0, SQUARE(k_s=8, k_e=16, c=1.0))]),
    N=16)

# Ancilla-free disjoint-support composition via PARTITION
circuit, info = encode(
    PARTITION([SPARSE([(2, 0.3), (5, 0.5), (7, 0.7)]),
               GEOMETRIC(r=0.8, k_s=11)]),
    N=256)
```
All ten patterns and the three compositions accept real or complex
amplitudes. Complex inputs add at most O(m) phase gates, which the
transpiler typically absorbs into adjacent rotations at
`optimization_level=3`; real-amplitude code paths run unchanged
(gate counts identical bit-for-bit).

Every call returns `(circuit, info)` where `circuit` is a Qiskit
`QuantumCircuit` and `info` is an `EncodingInfo` dataclass with gate
counts, circuit depth, complexity class, and optionally the
statevector-validated amplitude vector.

## Supported Patterns

| Pattern    | Constructor                       | Complexity                      | Source                              |
|------------|-----------------------------------|---------------------------------|-------------------------------------|
| Sparse     | `SPARSE([(x, a), ...])`           | O(s·m)                          | Gleinig & Hoefler (2021)            |
| Step       | `STEP(k_e, c)`                    | O(m)                            | Shukla & Vedula (2024)              |
| Square     | `SQUARE(k_s, k_e, c)`             | O(m²) general, O(m) aligned     | Shukla & Vedula (2024) + Draper (2000) |
| Walsh      | `WALSH(k, c0, c1)`                | O(m)                            | Welch et al. (2014)                 |
| Geometric  | `GEOMETRIC(r, k_s=0, k_e=None, c=1)` | O(m) for k_s=0, O(m²) general | Grover & Rudolph (2002), Bentley & Saxe (1980) |
| Hamming    | `HAMMING(r, c)`                   | O(m), 0 CX, depth 1             | Cruz et al. (2019)                  |
| Staircase  | `STAIRCASE(r, c)`                 | O(m)                            | Hackbusch (1999), Möttönen et al. (2005) |
| Dicke      | `DICKE(k, c)`                     | O(k·(m−k))                      | Bärtschi & Eidenbenz (2019)         |
| Polynomial | `POLYNOMIAL(coeffs)`              | O(m^(d+1))                      | Welch et al. (2014), Gonzalez-Conde et al. (2024) |
| Fourier    | `FOURIER(modes=[...])`            | O(m²)                           | Gonzalez-Conde et al. (2024), Moosa et al. (2024) |
| Sum        | `SUM([(w, pattern), ...])`        | Σ component costs               | Childs et al. (2018), Babbush et al. (2018) |
| Partition  | `PARTITION([pattern, ...])`       | O(L·m), ancilla-free            | Bentley & Saxe (1980), Gleinig & Hoefler (2021) |
| Tensor     | `TENSOR([(pattern, N_i), ...])`   | Σ component costs, depth = max  | Nielsen & Chuang (2010)             |

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

- `pattern_name` — name of the recognized pattern (e.g. `"SPARSE"`, `"GEOMETRIC"`)
- `N`, `m` — vector length and number of qubits
- `params` — supplied vector parameters (e.g. `{"r": 0.95, "c": 1.0}`)
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

For design-optimization workflows, `predict_gates(pattern, N)` returns
transpiled gate counts analytically, without ever building a circuit.
It's typically 500–8000× faster than `encode()`, which makes it practical
to evaluate thousands of candidate encodings inside an outer optimization
loop before committing to synthesis.

```python
from pyencode import predict_gates, POLYNOMIAL

p = predict_gates(POLYNOMIAL(coeffs=[0.0, 1.0]), N=4096)
# {'pattern_name': 'POLYNOMIAL', 'N': 4096, 'm': 12,
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
  POLYNOMIAL d=2, SQUARE general, FOURIER T≥2, and DICKE; more
  conservative for SPARSE s≥2, GEOMETRIC with k_s>0, and the composite
  constructors TENSOR, SUM, PARTITION).

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
from pyencode import (predict_gates, TENSOR, SUM, PARTITION,
                      FOURIER, SQUARE, SPARSE, GEOMETRIC)

# TENSOR sums over disjoint subregisters
p = predict_gates(TENSOR([(FOURIER(modes=[(2, 1.0, 0)]), 32),
                          (FOURIER(modes=[(3, 1.0, 0)]), 32)]),
                  N=32*32)

# SUM: weighted superposition (ancilla-based via LCU)
p = predict_gates(SUM([(0.5, SQUARE(k_s=0, k_e=8, c=1.0)),
                       (0.5, SQUARE(k_s=8, k_e=16, c=1.0))]),
                  N=16)

# PARTITION: disjoint-support composition (ancilla-free, p = 1)
p = predict_gates(PARTITION([SPARSE([(2, 0.3), (5, 0.5), (7, 0.7)]),
                             GEOMETRIC(r=0.8, k_s=11)]),
                  N=256)
```

### When to use `encode()` instead

`predict_gates` is for the selection phase; `encode` is for the commit
phase. Use `encode()` when you need:

- the Qiskit circuit itself, to simulate, transpile to hardware, or run
  experiments;
- the emitted source snippet in `info.circuit_code`;
- statevector validation via `validate=True`.

## Reverse Lookup: Which Pattern Fits a Vector?

The forward direction maps a typed declaration to a circuit. The reverse
direction takes a *materialized* amplitude vector and finds which exact
pattern family best represents it. `match_vector(v)` fits every family's
free parameters to `v`, scores each fit, and returns the closest matches
ranked by error — each as a ready-to-`encode` constructor.

```python
import numpy as np
from pyencode import match_vector, print_matches, encode

v = np.zeros(16); v[:5] = 1.0          # a prefix step
matches = match_vector(v)              # top 3 by default
print_matches(matches)
# rank  pattern      rel_error   fidelity   gates  params
#    1  STEP         0.000e+00   1.000000       7  k_e=5, c=1
#    2  SQUARE       0.000e+00   1.000000       7  k_s=0, k_e=5, c=1
#    3  SPARSE       0.000e+00   1.000000      40  loads=[5 entries]

circuit, info = encode(matches[0].pattern, N=len(v))   # encode the winner
```

Each result is a `PatternMatch` with `pattern_name`, the fitted `pattern`
constructor, `params`, `rel_error`, `fidelity`, and the predicted
`gate_count`.

The fit metric is scale- and phase-invariant — a pattern's leading
amplitude `c` is a free parameter and quantum states are normalized, so

```
rel_error = ||v - alpha·w|| / ||v|| = sqrt(1 - |<w, v>|² / (||w||²·||v||²))
```

where `alpha` is the optimal complex scale and `w` is the pattern's vector.
`rel_error` lies in `[0, 1]` (0 is an exact structural match) and
`fidelity = 1 - rel_error²`. When several families tie on error (e.g. the
vector is reproduced exactly by more than one), the cheaper circuit — by
predicted gate count — ranks first.

```python
# Restrict the search, control top-k, tune fits
match_vector(v, top_k=5)
match_vector(v, families=["GEOMETRIC", "HAMMING", "FOURIER"])
match_vector(v, max_fourier_modes=4, max_poly_degree=3)
```

`SPARSE` always reproduces a vector exactly from its nonzero entries, so it
appears as the universal (but often expensive) fallback whenever no
lower-cost structured family fits. `match_vector` operates on a real or
complex numerical vector of power-of-two length; it does not attempt the
composite constructors (`SUM`, `PARTITION`, `TENSOR`).

## Approximate Encoding via Matrix Product States

For amplitude vectors that fall outside the exact pattern families —
e.g., the discretized Gaussian, log-normal density, or other smooth
distributions whose DFT is dense and whose values don't admit bitwise
factorization — PyEncode provides a standalone matrix product state
loader in a separate submodule:

```python
import numpy as np
from pyencode.mps import encode_mps

N = 256
i = np.arange(N)
v = np.exp(-50.0 * ((i - N/2) / N) ** 2)
v /= np.linalg.norm(v)

circuit, info = encode_mps(v, bond_dim=8, validate=True)
# info.params["n_bond"]              → 3
# info.params["truncation_error_sq"] → < 1e-12
# info.success_probability           → 1.0
```

The single user-facing knob is `bond_dim` (χ), which trades approximation
error against gate count. Cost is O(m·χ²) two-qubit gates on n_bond + m
qubits, where n_bond = ⌈log₂ χ⌉. The truncation error is reported in
`info.params["truncation_error_sq"]` — increase χ until it falls below
the application's tolerance.

Unlike `encode`, `encode_mps` operates on a materialized numerical vector
(real or complex; non-power-of-2 lengths are zero-padded) and does not
currently compose with `SUM`, `PARTITION`, or `TENSOR`. A second entry
point, `encode_mps_from_tensors(tensors)`, accepts pre-built
right-canonical site tensors of shape (χ_l, 2, χ_r) and skips the SVD
sweep — useful when tensors come from an external source such as a DMRG
calculation.

## Testing

```bash
pytest                  # 362 tests, ~20s on a laptop
pytest -m slow          # opt-in: 3 deep-coverage tests at m=12
pytest -m "not slow"    # explicit fast path (default)
```

The full suite covers every pattern at small m (m ≤ 8), every pattern
at m = 10 with statevector validation, and a slow tier at m = 12 for
selected patterns (POLYNOMIAL d=2, FOURIER multi-mode, DICKE).

## Paper

Figures and Table 2 in the paper are reproduced by:

```bash
python generate_figures.py
```

The exploration notebook `notebooks/explore.ipynb` provides one
runnable cell per pattern and composition for interactive use; it is
not a paper-reproduction artifact (use `generate_figures.py` for that).

## Citation

If you use PyEncode in your work, please cite:

```bibtex
@misc{suresh2026pyencode,
  title        = {{PyEncode}: An Open-Source Library for Structured
                  Quantum State Preparation},
  author       = {Krishnan Suresh and Sanjay Suresh},
  year         = {2026},
  eprint       = {2603.28259},
  archivePrefix= {arXiv},
  primaryClass = {cs.ET},
  note         = {Code available at \url{https://github.com/UW-ERSL/PyEncode}}
}
```

## License

University of Wisconsin–Madison.