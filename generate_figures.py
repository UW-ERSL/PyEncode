"""
generate_figures.py
===================
Generates all vector + circuit figures for the PyEncode paper (Sections 5 & 6).

Run:  python generate_figures.py

Output:  figures/*.png
"""

import math
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pyencode import (
    encode_params, encode_vector, encode_python,
    DISCRETE, UNIFORM, STEP, SQUARE,
    SINE, COSINE, MULTI_DISCRETE, MULTI_SINE,
    VectorType,
)

# ── matplotlib style ──────────────────────────────────────────────
rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

FIGDIR = "figures"


# ── helpers ───────────────────────────────────────────────────────

def plot_vector(f, N, title, filename, ylabel=r"$f_i$"):
    """Bar chart of vector f, matching paper style."""
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(range(N), f, width=1.0, color='steelblue', edgecolor='steelblue',
           linewidth=0.3)
    ax.set_xlabel("Node index $i$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, N - 0.5)
    ticks = list(range(0, N + 1, N // 4))
    ax.set_xticks(ticks)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_title(title, fontsize=10, pad=4)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{filename}")
    plt.close(fig)
    print(f"  saved {FIGDIR}/{filename}")


def plot_vector_2d(f_2d, N, title, filename):
    """Heatmap of a 2D vector."""
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    im = ax.imshow(f_2d, origin='lower', cmap='RdBu_r', aspect='equal')
    ax.set_xlabel("$j$")
    ax.set_ylabel("$i$")
    ax.set_title(title, fontsize=10, pad=4)
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$f_{ij}$")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{filename}")
    plt.close(fig)
    print(f"  saved {FIGDIR}/{filename}")


def save_circuit(circuit, filename, scale=0.7):
    """Save Qiskit circuit diagram as PNG."""
    fig = circuit.draw('mpl', scale=scale)
    fig.savefig(f"{FIGDIR}/{filename}", bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  saved {FIGDIR}/{filename}")


def print_info(label, info):
    """Print key info fields."""
    print(f"  {label}: type={info.vector_type}, gates={info.gate_count}, "
          f"complexity={info.complexity}")


# ===================================================================
# Section 5.1: encode_python examples
# ===================================================================

def sec51_example1():
    """Discrete vector, AST recognition, no hint."""
    print("\n--- Sec 5.1 Example 1: Discrete (encode_python, no hint) ---")
    code = """
N = 64
f = np.zeros(N)
f[20] = 1.0
"""
    circuit, info = encode_python(code)
    print_info("encode_python", info)

    f = np.zeros(64); f[20] = 1.0
    plot_vector(f, 64, "Discrete: $f_{20}=1$", "discrete_vector.png")
    save_circuit(circuit, "discrete_circuit.png", scale=0.7)


def sec51_example2():
    """Sine via for-loop with hint."""
    print("\n--- Sec 5.1 Example 2: Sine for-loop (encode_python, hint) ---")
    code = """
import numpy as np
N = 64
f = np.zeros(N)
for i in range(N):
    f[i] = np.sin(3 * np.pi * i / N)
"""
    circuit, info = encode_python(code, vector_type="SINE")
    print_info("encode_python", info)

    N = 64; k = np.arange(N)
    f = np.sin(3 * np.pi * k / N)
    plot_vector(f, N, r"Sine: $\sin(3\pi i/N)$", "sine_loop_vector.png")
    save_circuit(circuit, "sine_loop_circuit.png", scale=0.5)


def sec51_example3():
    """Damped sine -> Shende fallback."""
    print("\n--- Sec 5.1 Example 3: Damped sine (Shende fallback) ---")
    code = """
import numpy as np
N = 64
x = np.linspace(0, 1, N)
f = np.sin(3 * np.pi * x) * np.exp(-x)
"""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        circuit, info = encode_python(code)
    print_info("encode_python", info)

    N = 64; x = np.linspace(0, 1, N)
    f = np.sin(3 * np.pi * x) * np.exp(-x)
    plot_vector(f, N, r"Damped sine: $\sin(3\pi x)e^{-x}$",
                "damped_sine_vector.png")
    save_circuit(circuit, "damped_sine_circuit.png", scale=0.35)


# ===================================================================
# Section 5.2: encode_vector examples
# ===================================================================

def sec52_example3():
    """Multi-tone signal, auto-detect."""
    print("\n--- Sec 5.2 Example 3: Multi-tone (encode_vector, auto) ---")
    N = 64
    k = np.arange(N)
    f = 0.8 * np.sin(3 * np.pi * k / N) + 0.6 * np.sin(7 * np.pi * k / N)
    circuit, info = encode_vector(f)
    print_info("encode_vector", info)

    plot_vector(f, N, r"Multi-tone: $0.8\sin(3\pi i/N)+0.6\sin(7\pi i/N)$",
                "multi_tone_vector.png")
    save_circuit(circuit, "multi_tone_circuit.png", scale=0.35)


# ===================================================================
# Section 5.3: encode_params examples
# ===================================================================

def sec53_example1a():
    """DISCRETE(k=32, P=1.0), N=64."""
    print("\n--- Sec 5.3 Example 1a: DISCRETE(k=32) ---")
    circuit, info = encode_params(DISCRETE(k=32, P=1.0), N=64)
    print_info("encode_params", info)

    f = np.zeros(64); f[32] = 1.0
    plot_vector(f, 64, "Discrete: $f_{32}=1$", "params_discrete_vector.png")
    save_circuit(circuit, "params_discrete_circuit.png", scale=0.7)


def sec53_example1b():
    """SINE(n=3, A=2.0, phi=pi/4), N=64."""
    print("\n--- Sec 5.3 Example 1b: SINE(n=3, phi=pi/4) ---")
    circuit, info = encode_params(SINE(n=3, A=2.0, phi=np.pi/4), N=64)
    print_info("encode_params", info)

    N = 64; k = np.arange(N)
    f = 2.0 * np.sin(3 * np.pi * k / N + np.pi / 4)
    plot_vector(f, N, r"Sine: $2\sin(3\pi i/N + \pi/4)$",
                "params_sine_vector.png")
    save_circuit(circuit, "params_sine_circuit.png", scale=0.5)


def sec53_example2():
    """Composite: [SINE(n=3), SINE(n=7), DISCRETE(k=32)]."""
    print("\n--- Sec 5.3 Example 2: Composite ---")
    # Build the composite vector for plotting
    N = 64; k = np.arange(N)
    f = (1.0 * np.sin(3 * np.pi * k / N)
         + 0.5 * np.sin(7 * np.pi * k / N))
    f[32] += 2.0

    # Use encode_vector for the composite since encode_params
    # with mixed types falls back to Shende
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        circuit, info = encode_params([
            SINE(n=3, A=1.0),
            SINE(n=7, A=0.5),
            DISCRETE(k=32, P=2.0),
        ], N=64)
    print_info("encode_params", info)

    plot_vector(f, N,
                r"Composite: sine + sine + discrete",
                "composite_vector.png")
    save_circuit(circuit, "composite_circuit.png", scale=0.35)


# ===================================================================
# Section 6: Applications
# ===================================================================

def sec61_hubbard():
    """Fermi-Hubbard PREP oracle."""
    print("\n--- Sec 6.1: Fermi-Hubbard ---")
    L = 8; t = 1.0; U = 4.0
    N_pauli = 4 * L  # round to power of 2: 32
    # Build the coefficient vector
    f = np.zeros(N_pauli)
    f[0:2*L] = t       # hopping terms
    f[2*L:3*L] = U     # on-site terms

    circuit, info = encode_params([
        SQUARE(k1=0, k2=2*L, c=t),
        SQUARE(k1=2*L, k2=3*L, c=U),
    ], N=N_pauli)
    print_info("encode_params", info)

    plot_vector(f, N_pauli,
                r"Fermi-Hubbard: $t=1, U=4, L=8$",
                "hubbard_vector.png")
    save_circuit(circuit, "hubbard_circuit.png", scale=0.35)


def sec62_poisson():
    """2D Poisson separable source."""
    print("\n--- Sec 6.2: 2D Poisson ---")
    N = 32; n_mode = 2; m_mode = 3
    k = np.arange(N)
    u = np.sin(n_mode * np.pi * k / N)
    v = np.sin(m_mode * np.pi * k / N)
    f_2d = np.outer(u, v)

    circ_x, info_x = encode_params(SINE(n=n_mode, A=1.0), N=N)
    circ_y, info_y = encode_params(SINE(n=m_mode, A=1.0), N=N)
    circuit = circ_x.tensor(circ_y)
    print(f"  x-register: {info_x.gate_count} gates, "
          f"y-register: {info_y.gate_count} gates, "
          f"total: {info_x.gate_count + info_y.gate_count} gates")

    plot_vector_2d(f_2d, N,
                   r"$\sin(2\pi i/N)\sin(3\pi j/N)$",
                   "poisson_vector.png")
    save_circuit(circuit, "poisson_circuit.png", scale=0.45)


def sec63_finance():
    """Quantum finance: MULTI_DISCRETE for price distribution."""
    print("\n--- Sec 6.3: Quantum Finance ---")
    N = 64
    # Discretised distribution as weighted discrete vector
    indices = [8, 16, 24, 32, 40, 48, 56]
    weights = [0.05, 0.15, 0.25, 0.30, 0.15, 0.07, 0.03]

    circuit, info = encode_params(
        MULTI_DISCRETE(vectors=[
            DISCRETE(k=k_i, P=w_i)
            for k_i, w_i in zip(indices, weights)
        ]),
        N=N,
    )
    print_info("encode_params", info)

    f = np.zeros(N)
    for k_i, w_i in zip(indices, weights):
        f[k_i] = w_i
    plot_vector(f, N,
                "Price distribution (7 bins)",
                "finance_vector.png",
                ylabel=r"$\sqrt{p_i}$")
    save_circuit(circuit, "finance_circuit.png", scale=0.45)


# ===================================================================
# Gate count table (Table 2)
# ===================================================================

def gate_count_table():
    """Generate gate counts for Table 2."""
    print("\n--- Gate Count Table (N=64, m=6) ---")
    N = 64

    from qiskit.circuit.library import StatePreparation
    from qiskit import transpile

    def shende_gates(f_vec):
        """Gate count for Shende StatePreparation on same vector."""
        norm = np.linalg.norm(f_vec)
        if norm < 1e-14:
            return 0
        sv = f_vec / norm
        sp = StatePreparation(sv.astype(complex))
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(int(np.log2(N)))
        qc.append(sp, range(qc.num_qubits))
        qc = qc.decompose()
        t = transpile(qc, basis_gates=['cx', 'u', 'x', 'h', 'ry', 'rz', 'rx', 'p'],
                       optimization_level=2)
        return sum(t.count_ops().values())

    cases = [
        ("Discrete (k=20)", DISCRETE(k=20, P=1.0),
         lambda: np.eye(N)[20]),
        ("Uniform", UNIFORM(c=1.0),
         lambda: np.ones(N)),
        ("Square (8:16)", SQUARE(k1=8, k2=16, c=1.0),
         lambda: np.array([1.0 if 8 <= i < 16 else 0 for i in range(N)])),
        ("Step (k_s=4)", STEP(k_s=4, c=1.0),
         lambda: np.array([1.0 if i < 4 else 0 for i in range(N)])),
        ("Sine n=1", SINE(n=1, A=1.0),
         lambda: np.sin(1 * np.pi * np.arange(N) / N)),
        ("Sine n=3 phi=pi/4", SINE(n=3, A=2.0, phi=np.pi/4),
         lambda: 2.0 * np.sin(3 * np.pi * np.arange(N) / N + np.pi/4)),
        ("Cosine n=1", COSINE(n=1, A=1.0),
         lambda: np.cos(1 * np.pi * np.arange(N) / N)),
        ("Cosine n=3 phi=pi/4", COSINE(n=3, A=1.0, phi=np.pi/4),
         lambda: np.cos(3 * np.pi * np.arange(N) / N + np.pi/4)),
        ("Multi-discrete", MULTI_DISCRETE(vectors=[
            DISCRETE(k=10, P=3.0), DISCRETE(k=50, P=4.0)]),
         lambda: np.array([3.0 if i == 10 else 4.0 if i == 50 else 0 for i in range(N)])),
    ]

    print(f"{'Pattern':<30} {'PyEncode':>10} {'Shende':>10} {'Complexity':<15}")
    print("-" * 70)
    for name, vobj, f_fn in cases:
        circuit, info = encode_params(vobj, N=N)
        f_vec = f_fn()
        sg = shende_gates(f_vec)
        print(f"{name:<30} {info.gate_count:>10} {sg:>10} {info.complexity:<15}")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    import os
    os.makedirs(FIGDIR, exist_ok=True)

    # Section 5.1
    sec51_example1()
    sec51_example2()
    sec51_example3()

    # Section 5.2
    sec52_example3()

    # Section 5.3
    sec53_example1a()
    sec53_example1b()
    sec53_example2()

    # Section 6
    sec61_hubbard()
    sec62_poisson()
    sec63_finance()

    # Gate count table
    gate_count_table()

    print("\nAll figures generated in figures/")
