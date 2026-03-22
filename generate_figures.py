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
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

FIGDIR = "figures"


# ── helpers ───────────────────────────────────────────────────────

def plot_vector(f, N, title, filename, ylabel=r"$f_i$", smooth=False):
    """Plot vector f. Use smooth=True for continuous functions (sine, cosine, etc.)."""
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    if smooth:
        x = np.arange(N)
        # Interpolate for smooth curve
        from scipy.interpolate import make_interp_spline
        x_smooth = np.linspace(0, N - 1, 500)
        spl = make_interp_spline(x, f, k=3)
        f_smooth = spl(x_smooth)
        ax.plot(x_smooth, f_smooth, color='steelblue', linewidth=1.5)
        ax.fill_between(x_smooth, 0, f_smooth, alpha=0.15, color='steelblue')
        ax.plot(x, f, 'o', color='steelblue', markersize=2.0, alpha=0.4)
    else:
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
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    im = ax.imshow(f_2d, origin='lower', cmap='RdBu_r', aspect='equal')
    ax.set_xlabel("$j$")
    ax.set_ylabel("$i$")
    ax.set_title(title, fontsize=10, pad=4)
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$f_{ij}$")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{filename}")
    plt.close(fig)
    print(f"  saved {FIGDIR}/{filename}")


def save_circuit(circuit, filename, scale=0.8, fold=-1):
    """Save Qiskit circuit diagram as PNG."""
    fig = circuit.draw('mpl', scale=scale, fold=fold)
    fig.savefig(f"{FIGDIR}/{filename}", bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  saved {FIGDIR}/{filename}")


def save_shende_block(m, filename, label=None):
    """Draw a clean single-block diagram for Shende fallback."""
    if label is None:
        label = "Shende\nState Prep\n$\\mathcal{O}(2^m)$"
    fig, ax = plt.subplots(figsize=(3.4, 0.45 * m + 0.8))
    ax.set_xlim(-0.8, 3.5)
    ax.set_ylim(-0.5, m - 0.5)
    ax.invert_yaxis()
    for i in range(m):
        ax.plot([-0.5, 0.5], [i, i], 'k-', lw=1.5)
        ax.plot([2.5, 3.3], [i, i], 'k-', lw=1.5)
        ax.text(-0.7, i, f'$q_{{{i}}}$', ha='right', va='center', fontsize=12)
    rect = plt.Rectangle((0.5, -0.4), 2.0, m - 0.2,
                          facecolor='#b5446e', edgecolor='#8b3456',
                          linewidth=2.0, alpha=0.92)
    ax.add_patch(rect)
    ax.text(1.5, (m - 1) / 2, label,
            ha='center', va='center', fontsize=10, color='white',
            fontweight='bold', linespacing=1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{filename}", bbox_inches='tight', pad_inches=0.08)
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
N = 8
f = np.zeros(N)
f[3] = 1.0
"""
    circuit, info = encode_python(code)
    print_info("encode_python", info)

    f = np.zeros(8); f[3] = 1.0
    plot_vector(f, 8, "Discrete: $f_3=1$, $N=8$", "discrete_vector.png")
    save_circuit(circuit, "discrete_circuit.png", scale=0.6)


def sec51_example2():
    """Sine via for-loop with hint."""
    print("\n--- Sec 5.1 Example 2: Sine for-loop (encode_python, hint) ---")
    code = """
import numpy as np
N = 8
f = np.zeros(N)
for i in range(N):
    f[i] = np.sin(1 * 2 * np.pi * i / N)
"""
    circuit, info = encode_python(code, vector_type="SINE")
    print_info("encode_python", info)

    N = 8; k = np.arange(N)
    f = np.sin(1 * 2 * np.pi * k / N)
    plot_vector(f, N, r"Sine: $\sin(2\pi i/N)$, $N=8$", "sine_loop_vector.png", smooth=True)
    save_circuit(circuit, "sine_loop_circuit.png", scale=1.0)


def sec51_example3():
    """Damped sine -> Shende fallback."""
    print("\n--- Sec 5.1 Example 3: Damped sine (Shende fallback) ---")
    code = """
import numpy as np
N = 8
x = np.linspace(0, 1, N)
f = np.sin(3 * 2 * np.pi * x) * np.exp(-x)
"""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        circuit, info = encode_python(code)
    print_info("encode_python", info)

    N = 8; x = np.linspace(0, 1, N)
    f = np.sin(3 * 2 * np.pi * x) * np.exp(-x)
    plot_vector(f, N, r"Damped sine: $\sin(6\pi x)e^{-x}$, $N=8$",
                "damped_sine_vector.png", smooth=True)
    save_shende_block(3, "damped_sine_circuit.png")


# ===================================================================
# Section 5.2: encode_vector examples
# ===================================================================

def sec52_uniform():
    """Uniform vector, auto-detect."""
    print("\n--- Sec 5.2: Uniform (encode_vector, auto) ---")
    f = np.full(8, 3.0)
    circuit, info = encode_vector(f)
    print_info("encode_vector", info)

    plot_vector(f, 8, r"Uniform: $f_i=3$, $N=8$", "uniform_vector.png")
    save_circuit(circuit, "uniform_circuit.png", scale=0.6)


def sec52_step():
    """Step vector, auto-detect."""
    print("\n--- Sec 5.2: Step (encode_vector, auto) ---")
    N = 8
    f = np.zeros(N); f[:4] = 2.0
    circuit, info = encode_vector(f)
    print_info("encode_vector", info)

    plot_vector(f, N, r"Step: $f_{[:4]}=2$, $N=8$", "step_vector.png")
    save_circuit(circuit, "step_circuit.png", scale=0.6)


def sec52_square():
    """Square vector, auto-detect."""
    print("\n--- Sec 5.2: Square (encode_vector, auto) ---")
    N = 8
    f = np.zeros(N); f[4:8] = 1.0
    circuit, info = encode_vector(f)
    print_info("encode_vector", info)

    plot_vector(f, N, r"Square: $f_{[4:8]}=1$, $N=8$", "square_vector.png")
    save_circuit(circuit, "square_circuit.png", scale=0.6)


def sec52_multi_discrete():
    """Multi-discrete signal, auto-detect."""
    print("\n--- Sec 5.2: Multi-discrete (encode_vector, auto) ---")
    N = 8
    f = np.zeros(N); f[1] = 3.0; f[6] = 4.0
    circuit, info = encode_vector(f)
    print_info("encode_vector", info)

    plot_vector(f, N, r"Multi-discrete: $f_1=3,\;f_6=4$, $N=8$",
                "multi_discrete_vector.png")
    save_circuit(circuit, "multi_discrete_circuit.png", scale=1.0)


# ===================================================================
# Section 5.3: encode_params examples
# ===================================================================

def sec53_example1a():
    """DISCRETE(k=5, P=1.0), N=8."""
    print("\n--- Sec 5.3 Example 1a: DISCRETE(k=5) ---")
    circuit, info = encode_params(DISCRETE(k=5, P=1.0), N=8)
    print_info("encode_params", info)

    f = np.zeros(8); f[5] = 1.0
    plot_vector(f, 8, "Discrete: $f_5=1$, $N=8$", "params_discrete_vector.png")
    save_circuit(circuit, "params_discrete_circuit.png", scale=0.6)


def sec53_sine():
    """SINE(n=1, A=1.0), N=16."""
    print("\n--- Sec 5.3: SINE(n=1) ---")
    circuit, info = encode_params(SINE(n=1, A=1.0), N=16)
    print_info("encode_params", info)

    N = 16; k = np.arange(N)
    f = np.sin(1 * 2 * np.pi * k / N)
    plot_vector(f, N, r"Sine: $\sin(2\pi i/N)$, $N=16$",
                "params_sine_vector.png", smooth=True)
    save_circuit(circuit, "params_sine_circuit.png", scale=1.0)


def sec53_cosine():
    """COSINE(n=1, A=1.0), N=16."""
    print("\n--- Sec 5.3: COSINE(n=1) ---")
    circuit, info = encode_params(COSINE(n=1, A=1.0), N=16)
    print_info("encode_params", info)

    N = 16; k = np.arange(N)
    f = np.cos(1 * 2 * np.pi * k / N)
    plot_vector(f, N, r"Cosine: $\cos(2\pi i/N)$, $N=16$",
                "cosine_vector.png", smooth=True)
    save_circuit(circuit, "cosine_circuit.png", scale=1.0)


def sec53_multi_sine():
    """MULTI_SINE."""
    print("\n--- Sec 5.3: MULTI_SINE ---")
    circuit, info = encode_params(
        MULTI_SINE(modes=[SINE(n=1, A=2.0), SINE(n=3, A=1.0)]),
        N=8,
    )
    print_info("encode_params", info)

    N = 8; k = np.arange(N)
    f = 2.0 * np.sin(2 * np.pi * 1 * k / N) + np.sin(2 * np.pi * 3 * k / N)
    plot_vector(f, N, r"Multi-sine: $2\sin(2\pi i/N)+\sin(6\pi i/N)$, $N=8$",
                "multi_sine_vector.png", smooth=True)
    save_circuit(circuit, "multi_sine_circuit.png", scale=0.8)


def sec53_composite():
    """Multi-discrete via list: [DISCRETE(k=1), DISCRETE(k=6)]."""
    print("\n--- Sec 5.3: Multi-discrete list ---")
    N = 8

    circuit, info = encode_params([
            DISCRETE(k=1, P=3.0),
            DISCRETE(k=6, P=4.0),
        ], N=N)
    print_info("encode_params", info)

    f = np.zeros(N); f[1] = 3.0; f[6] = 4.0
    plot_vector(f, N,
                r"Multi-discrete: $f_1=3,\;f_6=4$, $N=8$",
                "composite_vector.png")
    save_circuit(circuit, "composite_circuit.png", scale=1.0)


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
    save_circuit(circuit, "hubbard_circuit.png", scale=0.7)


def sec62_poisson():
    """2D Poisson separable source."""
    print("\n--- Sec 6.2: 2D Poisson ---")
    N = 32; n_mode = 2; m_mode = 3
    k = np.arange(N)
    u = np.sin(n_mode * 2 * np.pi * k / N)
    v = np.sin(m_mode * 2 * np.pi * k / N)
    f_2d = np.outer(u, v)

    circ_x, info_x = encode_params(SINE(n=n_mode, A=1.0), N=N)
    circ_y, info_y = encode_params(SINE(n=m_mode, A=1.0), N=N)
    circuit = circ_x.tensor(circ_y)
    print(f"  x-register: {info_x.gate_count} gates, "
          f"y-register: {info_y.gate_count} gates, "
          f"total: {info_x.gate_count + info_y.gate_count} gates")

    plot_vector_2d(f_2d, N,
                   r"$\sin(4\pi i/N)\sin(6\pi j/N)$",
                   "poisson_vector.png")
    save_circuit(circuit, "poisson_circuit.png", scale=0.7)


def sec63_finance():
    """Quantum finance: MULTI_DISCRETE for price distribution."""
    print("\n--- Sec 6.3: Quantum Finance ---")
    N = 8
    # Discretised distribution as weighted discrete vector
    indices = [2, 4, 6]
    weights = [0.25, 0.50, 0.25]

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
                "Price distribution (3 bins), $N=8$",
                "finance_vector.png",
                ylabel=r"$\sqrt{p_i}$")
    save_circuit(circuit, "finance_circuit.png", scale=1.0)


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
    sec52_uniform()
    sec52_step()
    sec52_square()
    sec52_multi_discrete()

    # Section 5.3
    sec53_example1a()
    sec53_sine()
    sec53_cosine()
    sec53_multi_sine()
    sec53_composite()

    # Section 6
    sec61_hubbard()
    sec62_poisson()
    sec63_finance()

    # Gate count table
    gate_count_table()

    print("\nAll figures generated in figures/")
