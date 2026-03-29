"""
pyencode.emitter
=================
Generates standalone, human-readable Qiskit Python code for each
recognized load pattern.  The emitted code:

  - imports only what it needs (QuantumCircuit, QFTGate, etc.)
  - uses meaningful variable names and inline comments
  - is directly runnable without PyEncode installed
  - is intended to be tweaked by the user (e.g. swap QFT for AQFT,
    replace mcx with a hardware-native decomposition, adjust angles)

Public entry point
------------------
  emit_code(pattern)  ->  str

References
----------
  Shende, Markov, Bullock, IEEE TCAD 25(6), 1000-1010, 2006.
  Coppersmith, IBM Research Report RC19642, 1994.
  Plesch & Brukner, Phys. Rev. A 83, 032302, 2011.
"""

import math
import numpy as np
from typing import Optional

from .recognizer import LoadPattern, VectorType


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def emit_code(pattern: LoadPattern) -> str:
    """
    Return a standalone Python string that builds the Qiskit circuit
    for *pattern*.

    The returned string can be executed directly::

        exec(compile(info.circuit_code, "<string>", "exec"))
        # qc is now defined in the local namespace

    or saved to a .py file for manual editing.

    Parameters
    ----------
    pattern : LoadPattern

    Returns
    -------
    str
        Standalone Qiskit Python source.  The circuit is bound to the
        variable ``qc`` at the end of the snippet.
    """
    N = pattern.N
    m = int(round(math.log2(N)))

    dispatch = {
        VectorType.STEP:    _emit_step_load,
        VectorType.SQUARE:  _emit_square_load,
        VectorType.WALSH:   _emit_walsh,
        VectorType.SPARSE:  _emit_sparse,
        VectorType.FOURIER: _emit_fourier,
        VectorType.GEOMETRIC: _emit_geometric,
        VectorType.UNKNOWN: _emit_qiskit_fallback,
    }

    fn = dispatch.get(pattern.load_type, _emit_mottonen)
    return fn(m, pattern.params)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(m: int, label: str) -> str:
    """Common imports + circuit creation header."""
    return (
        f"# PyEncode — emitted circuit: {label}\n"
        f"# m = {m} qubits,  N = {2**m} nodes\n"
        f"# Edit freely; run as standalone Qiskit code.\n\n"
        f"from qiskit import QuantumCircuit\n"
    )


def _indent(lines: list[str], n: int = 0) -> str:
    """Join a list of lines, each indented by n spaces."""
    pad = " " * n
    return "\n".join(pad + l for l in lines)


def _binary_str(k: int, m: int) -> str:
    """Return m-bit binary string of k, LSB first."""
    return "".join(str((k >> b) & 1) for b in range(m))


# ---------------------------------------------------------------------------
# Point load  f[k] = P
# ---------------------------------------------------------------------------

def _emit_point_load(m: int, params: dict) -> str:
    k = params["k"]
    bits = _binary_str(k, m)
    lines = [
        _header(m, f"POINT_LOAD  k={k}"),
        f"qc = QuantumCircuit({m}, name='point_load')",
        f"",
        f"# Encode index k={k} = 0b{bits[::-1]} in binary (LSB = qubit 0).",
        f"# Apply X on each qubit where the corresponding bit of k is 1.",
    ]
    flipped = [b for b in range(m) if (k >> b) & 1]
    if flipped:
        for b in flipped:
            lines.append(f"qc.x({b})  # bit {b} of {k} is 1")
    else:
        lines.append(f"# k=0: all qubits stay |0> — no X gates needed")
    lines.append("")
    lines.append("# Circuit prepares |psi> = |k>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Uniform load  f = ones(N) * c
# ---------------------------------------------------------------------------

def _emit_uniform_load(m: int, params: dict) -> str:
    lines = [
        _header(m, "UNIFORM_LOAD"),
        f"qc = QuantumCircuit({m}, name='uniform_load')",
        f"",
        f"# Apply Hadamard to every qubit.",
        f"# H|0> = (|0>+|1>)/sqrt(2), so H^{{⊗m}}|0>^{{⊗m}} = (1/sqrt(N)) sum_k |k>.",
    ]
    for q in range(m):
        lines.append(f"qc.h({q})")
    lines.append("")
    lines.append(f"# Circuit prepares |psi> = (1/sqrt({2**m})) sum_k |k>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step load  f[:k_s] = c
# ---------------------------------------------------------------------------

def _emit_step_load(m: int, params: dict) -> str:
    k_s = params["k_s"]
    N   = 2 ** m
    lines = [_header(m, f"STEP_LOAD  k_s={k_s}")]

    if k_s & (k_s - 1) == 0:
        # Power-of-2: pure H gates
        p = int(round(math.log2(k_s)))
        lines += [
            f"qc = QuantumCircuit({m}, name='step_load')",
            f"",
            f"# k_s={k_s} = 2^{p}: apply H to the {p} lowest qubits.",
            f"# Upper qubits remain |0>, restricting support to [0, {k_s}).",
        ]
        for q in range(p):
            lines.append(f"qc.h({q})")
    else:
        # General: binary-tree Ry
        lines += [
            f"import math",
            f"",
            f"qc = QuantumCircuit({m}, name='step_load')",
            f"",
            f"# k_s={k_s} is not a power of 2 — use a binary-tree Ry decomposition.",
            f"# Each Ry angle splits amplitude between 'within range' and 'outside'.",
        ]
        lines += _step_ry_lines(k_s, m)

    lines.append("")
    lines.append(f"# Circuit prepares (1/sqrt({k_s})) sum_{{k=0}}^{{{k_s-1}}} |k>")
    return "\n".join(lines)


def _step_ry_lines(k: int, m: int) -> list[str]:
    """Generate Ry-tree lines for a general step load."""
    lines = []
    N = 2 ** m

    def recurse(k_rem, m_rem, depth):
        if m_rem == 0 or k_rem <= 0:
            return
        half = 2 ** (m_rem - 1)
        qubit = m_rem - 1
        if k_rem >= 2 ** m_rem:
            for q in range(m_rem):
                lines.append(f"qc.h({q})  # full superposition")
            return
        if k_rem <= half:
            recurse(k_rem, m_rem - 1, depth + 1)
        else:
            k_upper = k_rem - half
            theta = 2 * math.acos(math.sqrt(half / k_rem))
            lines.append(
                f"qc.ry({theta:.8f}, {qubit})  "
                f"# split: {half} states below / {k_upper} above"
            )
            for q in range(m_rem - 1):
                lines.append(f"qc.h({q})")
            if k_upper < half:
                theta2 = 2 * math.acos(math.sqrt(k_upper / half))
                lines.append(
                    f"qc.cry({theta2:.8f}, {qubit}, {qubit-1})  "
                    f"# upper branch: {k_upper} states"
                )

    recurse(k, m, 0)
    return lines


# ---------------------------------------------------------------------------
# Square load  f[k1:k2] = c
# ---------------------------------------------------------------------------

def _emit_square_load(m: int, params: dict) -> str:
    k1 = params["k1"]
    k2 = params["k2"]
    w  = k2 - k1
    lines = [_header(m, f"SQUARE_LOAD  k1={k1}  k2={k2}")]

    if (w & (w - 1)) == 0 and (k1 % w == 0):
        p = int(round(math.log2(w)))
        lines += [
            f"qc = QuantumCircuit({m}, name='square_load')",
            f"",
            f"# Segment [{k1}, {k2}) is w-aligned (k1={k1} is a multiple of w={w}=2^{p}).",
            f"# Encode start address k1={k1} on upper bits, then H on lower {p} bits.",
        ]
        for bit in range(m):
            if (k1 >> bit) & 1:
                lines.append(f"qc.x({bit})  # set address bit {bit}")
        for q in range(p):
            lines.append(f"qc.h({q})  # uniform superposition over segment")
    else:
        lines += [
            f"import numpy as np",
            f"from qiskit.circuit.library import StatePreparation",
            f"",
            f"# General segment [{k1}, {k2}): use Shende StatePreparation.",
            f"N = {2**m}",
            f"f = np.zeros(N)",
            f"f[{k1}:{k2}] = 1.0",
            f"f /= np.linalg.norm(f)",
            f"qc = QuantumCircuit({m}, name='square_load')",
            f"qc.append(StatePreparation(f.astype(complex)), range({m}))",
            f"qc = qc.decompose()",
        ]

    lines.append("")
    lines.append(
        f"# Circuit prepares (1/sqrt({w})) sum_{{k={k1}}}^{{{k2-1}}} |k>"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sinusoidal load
# ---------------------------------------------------------------------------

def _emit_sinusoidal(m: int, params: dict) -> str:
    n   = params["n"]
    phi = params.get("phi", 0.0)
    N   = 2 ** m
    phase_val = math.pi - 2 * phi

    lines = [
        _header(m, f"SINUSOIDAL_LOAD  n={n}  phi={phi:.4f}"),
        f"import math",
        f"from qiskit.circuit.library import QFTGate",
        f"",
        f"m, N, n, phi = {m}, {N}, {n}, {phi}",
        f"",
        f"qc = QuantumCircuit(m, name='sin_n{n}_phi{phi:.4f}')",
        f"",
        f"# ── Step 1: prepare frequency-domain state ────────────────────────",
        f"# Target: (e^{{i*phi}}|{n}> - e^{{-i*phi}}|{N-n}>) / sqrt(2)",
        f"# Hadamard on MSB creates two branches: |0>|...> and |1>|...>",
        f"qc.h(m - 1)  # qubit {m-1}: root of the two branches",
        f"",
        f"# Branch |0> on qubit {m-1}: encode index n={n} on lower qubits",
    ]

    # Branch 0: encode n (controlled on qubit m-1 = 0)
    for bit in range(m - 1):
        if (n >> bit) & 1:
            lines += [
                f"qc.x(m - 1)          # temporarily flip control to make CX active",
                f"qc.cx(m - 1, {bit})  # set bit {bit} of index {n}",
                f"qc.x(m - 1)          # restore",
            ]

    lines.append(f"")
    lines.append(f"# Branch |1> on qubit {m-1}: encode index N-n={N-n} on lower qubits")

    # Branch 1: encode N-n (controlled on qubit m-1 = 1)
    for bit in range(m - 1):
        if ((N - n) >> bit) & 1:
            lines.append(f"qc.cx(m - 1, {bit})  # set bit {bit} of index {N-n}")

    lines += [
        f"",
        f"# ── Step 2: relative phase gate ───────────────────────────────────",
        f"# P(pi - 2*phi) on |1> branch gives factor e^{{i*(pi-2*phi)}} = -e^{{-2i*phi}}",
        f"# Combined with e^{{i*phi}} factor from H: net phase on |1> branch = e^{{-i*phi}}",
        f"# Check: phi=0 -> P(pi) = Z  (sine is odd)  ✓",
        f"qc.p(math.pi - 2 * phi, m - 1)  # = P({phase_val:.6f})",
        f"",
        f"# ── Step 3: Quantum Fourier Transform ─────────────────────────────",
        f"# QFT maps frequency domain -> spatial domain.",
        f"# Replace with approximate QFT (AQFT) to reduce gate count if needed.",
        f"qc.append(QFTGate(num_qubits=m), qargs=list(range(m)))",
        f"",
        f"# Circuit prepares (1/||f||) sum_k sin(2*pi*{n}*k/{N} + {phi:.4f}) |k>",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cosine load
# ---------------------------------------------------------------------------

def _emit_cosine(m: int, params: dict) -> str:
    n   = params["n"]
    phi = params.get("phi", 0.0)
    N   = 2 ** m
    phase_val = -2 * phi

    lines = [
        _header(m, f"COSINE_LOAD  n={n}  phi={phi:.4f}"),
        f"import math",
        f"from qiskit.circuit.library import QFTGate",
        f"",
        f"m, N, n, phi = {m}, {N}, {n}, {phi}",
        f"",
        f"qc = QuantumCircuit(m, name='cos_n{n}_phi{phi:.4f}')",
        f"",
        f"# ── Step 1: prepare frequency-domain state ────────────────────────",
        f"# Target: (e^{{i*phi}}|{n}> + e^{{-i*phi}}|{N-n}>) / sqrt(2)",
        f"# Cosine DFT is symmetric (+), sine is antisymmetric (-).",
        f"qc.h(m - 1)  # qubit {m-1}: root of the two branches",
        f"",
        f"# Branch |0>: encode index n={n}",
    ]

    for bit in range(m - 1):
        if (n >> bit) & 1:
            lines += [
                f"qc.x(m - 1)",
                f"qc.cx(m - 1, {bit})  # bit {bit} of index {n}",
                f"qc.x(m - 1)",
            ]

    lines.append(f"")
    lines.append(f"# Branch |1>: encode index N-n={N-n}")

    for bit in range(m - 1):
        if ((N - n) >> bit) & 1:
            lines.append(f"qc.cx(m - 1, {bit})  # bit {bit} of index {N-n}")

    lines += [
        f"",
        f"# ── Step 2: relative phase gate ───────────────────────────────────",
        f"# P(-2*phi) gives e^{{-2i*phi}} on |1> branch.",
        f"# Check: phi=0 -> P(0) = I  (cosine DFT is real and symmetric)  ✓",
        f"qc.p(-2 * phi, m - 1)  # = P({phase_val:.6f})",
        f"",
        f"# ── Step 3: Quantum Fourier Transform ─────────────────────────────",
        f"qc.append(QFTGate(num_qubits=m), qargs=list(range(m)))",
        f"",
        f"# Circuit prepares (1/||f||) sum_k cos(2*pi*{n}*k/{N} + {phi:.4f}) |k>",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-point load
# ---------------------------------------------------------------------------

def _emit_multi_point_load(m: int, params: dict) -> str:
    loads = params["loads"]
    L     = len(loads)
    N     = 2 ** m

    amps    = np.array([abs(ld["P"]) for ld in loads], dtype=float)
    indices = [ld["k"] for ld in loads]
    norm    = np.linalg.norm(amps)
    amps_n  = amps / norm

    lines = [
        _header(m, f"MULTI_POINT_LOAD  L={L}"),
        f"import math",
        f"",
        f"# Load points (index, normalized amplitude):",
    ]
    for i, (k, a) in enumerate(zip(indices, amps_n)):
        lines.append(f"#   leaf {i}: |{k}>  amplitude = {a:.6f}")

    lines += [
        f"",
        f"qc = QuantumCircuit({m}, name='multi_point_load')",
        f"",
        f"# ── Binary-tree Ry decomposition ──────────────────────────────────",
        f"# d = ceil(log2(L)) path qubits distribute amplitude to L leaves.",
        f"# Each internal node j: Ry(theta_j) where",
        f"#   theta_j = 2*arcsin(||right subtree|| / ||node||)",
        f"# After the Ry tree, controlled-X gates encode the data bits of each index.",
    ]

    # Re-run the same algorithm as the synthesizer, but emit code instead
    if L == 1:
        k = indices[0]
        for bit in range(m):
            if (k >> bit) & 1:
                lines.append(f"qc.x({bit})  # bit {bit} of index {k}")
    else:
        d = math.ceil(math.log2(L))
        for d_try in range(d, m + 1):
            prefixes = [(k >> (m - d_try)) for k in indices]
            if len(set(prefixes)) == L:
                d = d_try
                break

        path_qubits = list(range(m - d, m))
        data_qubits = list(range(0, m - d))

        num_leaves     = 1 << d
        amps_padded    = np.zeros(num_leaves)
        indices_padded = [0] * num_leaves
        for a, k in zip(amps_n, indices):
            leaf_i = k >> (m - d)
            amps_padded[leaf_i]    = a
            indices_padded[leaf_i] = k

        node_norm = np.zeros(2 * num_leaves + 1)
        for i in range(num_leaves):
            node_norm[num_leaves + i] = amps_padded[i]
        for node in range(num_leaves - 1, 0, -1):
            node_norm[node] = math.sqrt(
                node_norm[2 * node] ** 2 + node_norm[2 * node + 1] ** 2)

        lines.append(f"")
        lines.append(
            f"# Path qubits: {path_qubits}  (qubit {path_qubits[-1]} = root)")
        if data_qubits:
            lines.append(f"# Data qubits: {data_qubits}")

        # Emit Ry tree
        _emit_ry_tree_lines(lines, node_norm, 1, 0, d, path_qubits, [])

        # Emit index encoding
        lines.append(f"")
        lines.append(f"# ── Index encoding ────────────────────────────────────────────────")
        for i in range(num_leaves):
            if amps_padded[i] < 1e-14:
                continue
            k = indices_padded[i]
            path_state = [(i >> j) & 1 for j in range(d)]
            lines.append(
                f"# Leaf {i}: index k={k}, path={path_state} on qubits {path_qubits}")
            for b in range(m - d):
                if (k >> b) & 1:
                    flip = [path_qubits[j] for j in range(d) if path_state[j] == 0]
                    for q in flip:
                        lines.append(f"qc.x({q})")
                    if len(path_qubits) == 1:
                        lines.append(f"qc.cx({path_qubits[0]}, {data_qubits[b]})")
                    else:
                        lines.append(
                            f"qc.mcx({path_qubits}, {data_qubits[b]})")
                    for q in flip:
                        lines.append(f"qc.x({q})")

    lines.append(f"")
    lines.append(
        f"# Circuit prepares (1/||P||) sum_i P_i |k_i>  "
        f"for k in {indices}")
    return "\n".join(lines)


def _emit_ry_tree_lines(lines, node_norm, node, depth, d, path_qubits, ctrl_path):
    """Recursively emit Ry tree lines."""
    num_leaves = 1 << d
    if node >= num_leaves:
        return
    norm_node  = node_norm[node]
    norm_right = node_norm[2 * node + 1]
    if norm_node < 1e-14:
        return
    theta = 2.0 * math.asin(min(norm_right / norm_node, 1.0))
    target = path_qubits[d - 1 - depth]

    if not ctrl_path:
        lines.append(
            f"qc.ry({theta:.8f}, {target})  "
            f"# depth {depth}: split norm {norm_node:.4f} -> "
            f"left {node_norm[2*node]:.4f} / right {norm_right:.4f}")
    else:
        ctrl_q = [cp[0] for cp in ctrl_path]
        ctrl_v = [cp[1] for cp in ctrl_path]
        flip   = [q for q, v in zip(ctrl_q, ctrl_v) if v == 0]
        for q in flip:
            lines.append(f"qc.x({q})")
        lines.append(
            f"qc.mcry({theta:.8f}, {ctrl_q}, {target})  "
            f"# depth {depth}, ctrl={list(zip(ctrl_q, ctrl_v))}")
        for q in flip:
            lines.append(f"qc.x({q})")

    _emit_ry_tree_lines(
        lines, node_norm, 2 * node,     depth + 1, d, path_qubits,
        ctrl_path + [(target, 0)])
    _emit_ry_tree_lines(
        lines, node_norm, 2 * node + 1, depth + 1, d, path_qubits,
        ctrl_path + [(target, 1)])


# ---------------------------------------------------------------------------
# Multi-sin load
# ---------------------------------------------------------------------------

def _emit_multi_sin_load(m: int, params: dict) -> str:
    modes = params["modes"]
    N     = 2 ** m

    freq_amps = np.zeros(N, dtype=complex)
    for mode in modes:
        n, A = mode["n"], mode["A"]
        if 0 < n < N:
            freq_amps[n]   += A / (2j)
            freq_amps[N-n] -= A / (2j)
    freq_amps /= np.linalg.norm(freq_amps)

    lines = [
        _header(m, f"MULTI_SIN_LOAD  modes={[(md['n'], md['A']) for md in modes]}"),
        f"import numpy as np",
        f"from qiskit.circuit.library import StatePreparation, QFTGate",
        f"",
        f"m, N = {m}, {N}",
        f"",
        f"# ── Step 1: build frequency-domain amplitude vector ───────────────",
        f"# For sin(2*pi*n*k/N): DFT has spikes at +n and -n with factor 1/(2i).",
        f"freq_amps = np.zeros(N, dtype=complex)",
    ]
    for mode in modes:
        n, A = mode["n"], mode["A"]
        lines += [
            f"freq_amps[{n}]     += {A} / (2j)   # positive frequency, mode n={n}",
            f"freq_amps[{N-n}]  -= {A} / (2j)   # negative frequency (conjugate)",
        ]
    lines += [
        f"freq_amps /= np.linalg.norm(freq_amps)  # normalize",
        f"",
        f"# ── Step 2: prepare frequency-domain state via StatePreparation ───",
        f"# Tweak: replace StatePreparation with a hand-crafted circuit",
        f"# if the frequency support is very sparse.",
        f"from qiskit import QuantumCircuit",
        f"qc = QuantumCircuit(m, name='multi_sin_load')",
        f"sp = StatePreparation(freq_amps, label='freq_domain')",
        f"qc.append(sp, range(m))",
        f"qc = qc.decompose()",
        f"",
        f"# ── Step 3: Quantum Fourier Transform ─────────────────────────────",
        f"qc.append(QFTGate(num_qubits=m), qargs=list(range(m)))",
        f"",
        f"# Circuit prepares (1/||f||) sum_k [",
    ]
    for mode in modes:
        lines.append(
            f"#   {mode['A']}*sin(2*pi*{mode['n']}*k/{N}) +")
        lines.append(f"# ] |k>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Uniform + spike  (Shende fallback with explanation)
# ---------------------------------------------------------------------------

def _emit_uniform_spike(m: int, params: dict) -> str:
    c, k, delta = params["c"], params["k"], params["delta"]
    N = 2 ** m

    lines = [
        _header(m, f"UNIFORM_SPIKE_LOAD  c={c}  k={k}  delta={delta}"),
        f"import numpy as np",
        f"from qiskit.circuit.library import StatePreparation",
        f"",
        f"# NOTE: PyEncode currently uses Shende's general StatePreparation",
        f"# for this pattern.  An analytical O(m^2) Ry-tree circuit exists",
        f"# (see paper §5) but is not yet implemented.",
        f"#",
        f"# To tweak: replace StatePreparation with a hand-crafted Ry tree.",
        f"",
        f"N = {N}",
        f"f = np.full(N, {float(c)})",
        f"f[{k}] = {float(delta)}",
        f"f = f / np.linalg.norm(f)",
        f"",
        f"from qiskit import QuantumCircuit",
        f"qc = QuantumCircuit({m}, name='uniform_spike_load')",
        f"sp = StatePreparation(f.astype(complex), label='uniform_spike')",
        f"qc.append(sp, range({m}))",
        f"qc = qc.decompose()",
        f"",
        f"# Circuit prepares (1/||f||)(c*sum_{{k≠{k}}}|k> + {delta}|{k}>)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Möttönen fallback
# ---------------------------------------------------------------------------

def _emit_mottonen(m: int, params: dict) -> str:
    lines = [
        _header(m, "UNKNOWN — Shende / Möttönen fallback"),
        f"import numpy as np",
        f"from qiskit import QuantumCircuit",
        f"from qiskit.circuit.library import StatePreparation",
        f"",
        f"# Pattern not recognized by PyEncode.",
        f"# Replace 'amplitudes' with your actual normalized amplitude vector.",
        f"# amplitudes = np.array([...], dtype=complex)",
        f"# amplitudes /= np.linalg.norm(amplitudes)",
        f"",
        f"amplitudes = None  # <-- supply your vector here",
        f"assert amplitudes is not None, 'Supply your amplitude vector'",
        f"",
        f"qc = QuantumCircuit({m}, name='mottonen')",
        f"sp = StatePreparation(amplitudes, label='mottonen')",
        f"qc.append(sp, range({m}))",
        f"qc = qc.decompose()",
    ]
    return "\n".join(lines)

def _emit_sparse(m: int, params: dict) -> str:
    """Emit circuit code for SPARSE (Gleinig-Hoefler)."""
    loads = params["loads"]
    lines = [
        _header(m, "SPARSE — Gleinig-Hoefler sparse state"),
        f"from qiskit import QuantumCircuit",
        f"",
        f"m = {m}",
        f"qc = QuantumCircuit(m, name='sparse')",
    ]
    if len(loads) == 1:
        k = loads[0]["k"]
        lines.append(f"# s=1: basis state |{k}>")
        for bit in range(m):
            if (k >> bit) & 1:
                lines.append(f"qc.x({bit})")
    else:
        lines.append("# s>1: Gleinig-Hoefler construction")
        lines.append("# (circuit synthesized internally by PyEncode)")
        lines.append(f"# loads = {loads}")
    return "\n".join(lines)


def _emit_fourier(m: int, params: dict) -> str:
    """Emit circuit code for FOURIER (inverse-QFT)."""
    modes = params["modes"]
    lines = [
        _header(m, "FOURIER — sinusoidal modes via inverse QFT"),
        f"import numpy as np",
        f"from qiskit import QuantumCircuit",
        f"from qiskit.circuit.library import QFT",
        f"",
        f"m = {m}",
        f"N = 2**m",
        f"k = np.arange(N)",
        f"",
        f"# Build the sinusoidal vector",
        f"f = np.zeros(N)",
    ]
    for mode in modes:
        n, A, phi = mode["n"], mode["A"], mode["phi"]
        lines.append(f"f += {A} * np.sin(2 * np.pi * {n} * k / N + {phi})")
    lines += [
        f"",
        f"# Encode as sparse Fourier state + inverse QFT",
        f"# (circuit synthesized internally by PyEncode)",
        f"# modes = {modes}",
    ]
    return "\n".join(lines)


def _emit_qiskit_fallback(m: int, params: dict) -> str:
    """Emit circuit code for Qiskit StatePreparation fallback."""
    lines = [
        _header(m, "UNKNOWN — Qiskit StatePreparation fallback"),
        f"import numpy as np",
        f"from qiskit import QuantumCircuit",
        f"from qiskit.circuit.library import StatePreparation",
        f"",
        f"# Pattern not recognized by PyEncode.",
        f"# Replace 'amplitudes' with your normalized amplitude vector.",
        f"amplitudes = None  # <-- supply your vector here",
        f"assert amplitudes is not None, 'Supply your amplitude vector'",
        f"",
        f"qc = QuantumCircuit({m}, name='qiskit_fallback')",
        f"sp = StatePreparation(amplitudes, label='fallback')",
        f"qc.append(sp, range({m}))",
        f"qc = qc.decompose()",
    ]
    return "\n".join(lines)


def _emit_walsh(m: int, params: dict) -> str:
    """Emit circuit code for generalized WALSH (R_y(theta)_k + H^{otimes m})."""
    import math as _math
    k     = params["k"]
    c_pos = params.get("c_pos", 1.0)
    c_neg = params.get("c_neg", -c_pos)
    denom = c_pos + c_neg
    if abs(denom) < 1e-14:
        theta = _math.pi
    else:
        theta = 2 * _math.atan2(c_pos - c_neg, c_pos + c_neg)
    is_standard = abs(theta - _math.pi) < 1e-12
    gate_line = (f"qc.x({k})          # standard Walsh: X = R_y(pi)"
                 if is_standard else
                 f"qc.ry({theta:.6f}, {k})  # generalized Walsh: R_y(theta)")
    lines = [
        _header(m, f"WALSH k={k}, c_pos={c_pos}, c_neg={c_neg} — period P=2^(k+1)={2**(k+1)}"),
        f"from qiskit import QuantumCircuit",
        f"",
        f"m = {m}",
        f"qc = QuantumCircuit(m, name='walsh')",
        gate_line,
        f"for q in range(m):",
        f"    qc.h(q)",
    ]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Geometric  f_i = c * ratio^i  (product state, m independent Ry)
# ---------------------------------------------------------------------------

def _emit_geometric(m: int, params: dict) -> str:
    """Emit circuit code for GEOMETRIC (product-state R_y per qubit)."""
    import math as _math
    ratio = params["ratio"]
    c = params.get("c", 1.0)

    lines = [
        _header(m, f"GEOMETRIC  ratio={ratio}, c={c}"),
        f"import math",
        f"from qiskit import QuantumCircuit",
        f"",
        f"m = {m}",
        f"ratio = {ratio!r}",
        f"qc = QuantumCircuit(m, name='geometric')",
        f"",
        f"# Product state: each qubit j gets R_y(2*arctan(ratio^(2^j)))",
        f"for j in range(m):",
        f"    theta_j = 2.0 * math.atan(ratio ** (2 ** j))",
        f"    qc.ry(theta_j, j)",
    ]
    return "\n".join(lines)
