"""
generate_figures.py
===================
Generates all vector + circuit figures for the PyEncode paper.

Run:  python generate_figures.py

Output:  figures/*.png
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pyencode import (
    encode, SPARSE, STEP, SQUARE, FOURIER, WALSH, GEOMETRIC,
    POPCOUNT, STAIRCASE, TENSOR, POLYNOMIAL, LCU,
)
from pyencode.config import BASIS_GATES, OPTIMIZATION_LEVEL, DECOMPOSE_REPS

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
    """Plot vector f. Use smooth=True for continuous functions."""
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    if smooth:
        from scipy.interpolate import make_interp_spline
        x = np.arange(N)
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
    ax.set_xticks(list(range(0, N + 1, N // 4)))
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
    ax.set_xlabel("$j$"); ax.set_ylabel("$i$")
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


def print_info(label, info):
    print(f"  {label}: type={info.vector_type}, gates={info.gate_count}, "
          f"complexity={info.complexity}")


def qiskit_gates(f_vec, N):
    """Gate count for Qiskit StatePreparation on same vector."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation
    norm = np.linalg.norm(f_vec)
    if norm < 1e-14:
        return 0
    sv = (f_vec / norm).astype(complex)
    m = int(round(math.log2(N)))
    qc = QuantumCircuit(m)
    qc.append(StatePreparation(sv), range(m))
    qc = qc.decompose(reps=DECOMPOSE_REPS)
    t = transpile(qc, basis_gates=BASIS_GATES, optimization_level=OPTIMIZATION_LEVEL)
    return sum(t.count_ops().values())


# ===================================================================
# Figures: vector types
# ===================================================================

def fig_sparse_single():
    """SPARSE s=1: basis vector (motivating example)."""
    print("\n--- SPARSE s=1: e_19, N=64 ---")
    circuit, info = encode(SPARSE([(19, 1.0)]), N=64)
    print_info("encode", info)
    f = np.zeros(64); f[19] = 1.0
    plot_vector(f, 64, r"SPARSE $s=1$: $f_{19}=1$, $N=64$", "discrete_vector.png")
    save_circuit(circuit, "discrete_circuit.png", scale=0.6)


def fig_sparse_two():
    """SPARSE s=2: two point masses."""
    print("\n--- SPARSE s=2: f_1=3, f_6=4, N=8 ---")
    circuit, info = encode(SPARSE([(1, 3.0), (6, 4.0)]), N=8)
    print_info("encode", info)
    f = np.zeros(8); f[1] = 3.0; f[6] = 4.0
    plot_vector(f, 8, r"SPARSE $s=2$: $f_1=3,\;f_6=4$, $N=8$", "composite_vector.png")
    save_circuit(circuit, "composite_circuit.png", scale=1.0)


def fig_step():
    """STEP: prefix uniform superposition."""
    print("\n--- STEP: k_s=4, N=8 ---")
    circuit, info = encode(STEP(k_s=4, c=1.0), N=8)
    print_info("encode", info)
    f = np.zeros(8); f[:4] = 1.0
    plot_vector(f, 8, r"STEP: $f_{[:4]}=1$, $N=8$", "step_vector.png")
    save_circuit(circuit, "step_circuit.png", scale=0.6)


def fig_square():
    """SQUARE: general interval."""
    print("\n--- SQUARE: [4,8), N=8 ---")
    circuit, info = encode(SQUARE(k1=4, k2=8, c=1.0), N=8)
    print_info("encode", info)
    f = np.zeros(8); f[4:8] = 1.0
    plot_vector(f, 8, r"SQUARE: $f_{[4:8]}=1$, $N=8$", "square_vector.png")
    save_circuit(circuit, "square_circuit.png", scale=0.6)


def fig_fourier_sine():
    """FOURIER T=1: sine."""
    print("\n--- FOURIER T=1 (sine), N=16 ---")
    circuit, info = encode(FOURIER(modes=[(1, 1.0, 0)]), N=16)
    print_info("encode", info)
    N = 16; k = np.arange(N)
    f = np.sin(1 * 2 * np.pi * k / N)
    plot_vector(f, N, r"FOURIER: $\sin(2\pi i/N)$, $N=16$",
                "params_sine_vector.png", smooth=True)
    save_circuit(circuit, "params_sine_circuit.png", scale=1.0)


def fig_fourier_multi():
    """FOURIER T=2: multi-sine."""
    print("\n--- FOURIER T=2 (multi-sine), N=8 ---")
    circuit, info = encode(FOURIER(modes=[(1, 2.0, 0), (3, 1.0, 0)]), N=8)
    print_info("encode", info)
    N = 8; k = np.arange(N)
    f = 2.0 * np.sin(2 * np.pi * 1 * k / N) + np.sin(2 * np.pi * 3 * k / N)
    plot_vector(f, N, r"FOURIER $T=2$: $2\sin(2\pi i/N)+\sin(6\pi i/N)$, $N=8$",
                "multi_sine_vector.png", smooth=True)
    save_circuit(circuit, "multi_sine_circuit.png", scale=0.8)


def fig_geometric():
    """GEOMETRIC: exponential decay as product state."""
    print("\n--- GEOMETRIC: ratio=0.5, N=8 ---")
    N = 8
    circuit, info = encode(GEOMETRIC(ratio=0.5), N=N)
    print_info("encode", info)
    f = 0.5 ** np.arange(N)
    plot_vector(f, N, r"GEOMETRIC: $f_i = 0.5^{\,i}$, $N=8$",
                "geometric_vector.png")
    save_circuit(circuit, "geometric_circuit.png", scale=0.6)


def fig_popcount():
    """POPCOUNT: product state, amplitudes depend only on Hamming weight."""
    print("\n--- POPCOUNT: r=0.5, N=16 ---")
    N = 16
    r = 0.5
    circuit, info = encode(POPCOUNT(r=r), N=N)
    print_info("encode", info)
    pops = np.array([bin(i).count("1") for i in range(N)], dtype=float)
    f = r ** pops
    plot_vector(f, N, r"POPCOUNT: $f_i = 0.5^{\,\mathrm{popcount}(i)}$, $N=16$",
                "popcount_vector.png")
    save_circuit(circuit, "popcount_circuit.png", scale=0.6)


def fig_staircase():
    """STAIRCASE: sparse geometric staircase on unary indices."""
    print("\n--- STAIRCASE: r=0.5, N=16 ---")
    N = 16
    m = int(round(math.log2(N)))
    r = 0.5
    circuit, info = encode(STAIRCASE(r=r), N=N)
    print_info("encode", info)
    f = np.zeros(N)
    for k in range(m + 1):
        f[(1 << k) - 1] = r ** k
    plot_vector(f, N,
                r"STAIRCASE: $f_{2^k-1} = 0.5^{\,k}$, $N=16$",
                "staircase_vector.png")
    save_circuit(circuit, "staircase_circuit.png", scale=0.8)


def fig_polynomial_ramp():
    """POLYNOMIAL d=1: ramp function (Walsh-sparse construction)."""
    print("\n--- POLYNOMIAL d=1 (ramp), N=16 ---")
    N = 16
    circuit, info = encode(POLYNOMIAL(coeffs=[0.0, 1.0]), N=N)
    print_info("encode", info)
    x = np.arange(N) / (N - 1)
    f = x
    plot_vector(f, N,
                r"POLYNOMIAL $d=1$: $f(x)=x$ on $[0,1]$, $N=16$",
                "polynomial_ramp_vector.png", smooth=True)
    save_circuit(circuit, "polynomial_ramp_circuit.png", scale=0.5)


def fig_polynomial_poiseuille():
    """POLYNOMIAL d=2: Poiseuille parabolic profile."""
    print("\n--- POLYNOMIAL d=2 (Poiseuille), N=32 ---")
    N = 32
    circuit, info = encode(POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]), N=N)
    print_info("encode", info)
    x = np.arange(N) / (N - 1)
    f = 4 * x - 4 * x * x
    plot_vector(f, N,
                r"POLYNOMIAL $d=2$: $f(x)=4x(1-x)$ (Poiseuille), $N=32$",
                "polynomial_poiseuille_vector.png", smooth=True)
    save_circuit(circuit, "polynomial_poiseuille_circuit.png", scale=0.4)


def fig_tensor():
    """TENSOR: 2D separable state via disjoint subregister composition."""
    print("\n--- TENSOR: 2D FOURIER x FOURIER ---")
    N_sub = 16
    n_x = 2
    n_y = 3
    k = np.arange(N_sub)
    circuit, info = encode(
        TENSOR([(FOURIER(modes=[(n_x, 1.0, 0)]), N_sub),
                (FOURIER(modes=[(n_y, 1.0, 0)]), N_sub)]),
        N=N_sub * N_sub)
    print_info("encode", info)
    f_2d = np.outer(
        np.sin(n_y * 2 * np.pi * k / N_sub),   # MSB subregister (y)
        np.sin(n_x * 2 * np.pi * k / N_sub),   # LSB subregister (x)
    )
    plot_vector_2d(f_2d, N_sub,
                   r"TENSOR: $\sin(2\pi n_x i/N)\sin(2\pi n_y j/N)$",
                   "tensor_vector.png")
    save_circuit(circuit, "tensor_circuit.png", scale=0.6)


def fig_lcu_disjoint():
    """LCU: two disjoint SQUARE intervals."""
    print("\n--- LCU disjoint: two SQUARE intervals, N=16 ---")
    N = 16
    circuit, info = encode(
        LCU([(1.0, SQUARE(k1=0, k2=8, c=1.0)),
             (3.0, SQUARE(k1=8, k2=16, c=1.0))]),
        N=N)
    print_info("encode", info)
    print(f"  success_probability = {info.success_probability:.4f}")
    f = np.zeros(N)
    f[:8] = 1.0; f[8:] = 3.0
    plot_vector(f, N,
                r"LCU (disjoint): $f_{[:8]}=1,\;f_{[8:]}=3$, $N=16$",
                "lcu_disjoint_vector.png")
    save_circuit(circuit, "lcu_disjoint_circuit.png", scale=0.5)


def fig_lcu_overlap():
    """LCU: overlapping STEP + FOURIER."""
    import warnings
    print("\n--- LCU overlapping: STEP + FOURIER, N=16 ---")
    N = 16
    k = np.arange(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        circuit, info = encode(
            LCU([(1.0, STEP(k_s=16, c=1.0)),
                 (1.0, FOURIER(modes=[(1, 1.0, 0)]))]),
            N=N)
    print_info("encode", info)
    print(f"  success_probability = {info.success_probability:.4f}")
    f = np.ones(N) + np.sin(2 * np.pi * k / N)
    plot_vector(f, N,
                r"LCU (overlap): uniform $+$ $\sin(2\pi i/N)$, $N=16$",
                "lcu_overlap_vector.png", smooth=True)
    save_circuit(circuit, "lcu_overlap_circuit.png", scale=0.4)


# ===================================================================
# Figures: applications
# ===================================================================

def fig_hubbard():
    """Fermi-Hubbard PREP oracle via generalized Walsh."""
    print("\n--- Fermi-Hubbard PREP ---")
    import math
    L = 8; t = 1.0; U = 4.0
    N = 2 * L  # = 16
    circuit, info = encode(
        WALSH(k=int(math.log2(L)), c_pos=t, c_neg=U),
        N=N)
    print_info("encode", info)
    # coefficient vector: t on [0, L), U on [L, 2L)
    f = np.zeros(N)
    f[:L] = t; f[L:] = U
    plot_vector(f, N, r"Fermi-Hubbard PREP: $t=1,\;U=4,\;L=8,\;N=16$",
                "hubbard_vector.png")
    save_circuit(circuit, "hubbard_circuit.png", scale=1.0)


def fig_poisson():
    """2D Poisson separable source — demonstrates TENSOR pattern."""
    print("\n--- 2D Poisson (via TENSOR) ---")
    N = 32; n_x = 2; n_y = 3
    k = np.arange(N)
    circuit, info = encode(
        TENSOR([(FOURIER(modes=[(n_x, 1.0, 0)]), N),
                (FOURIER(modes=[(n_y, 1.0, 0)]), N)]),
        N=N * N)
    print_info("encode", info)
    f_2d = np.outer(
        np.sin(n_y * 2 * np.pi * k / N),   # MSB subregister (y)
        np.sin(n_x * 2 * np.pi * k / N),   # LSB subregister (x)
    )
    plot_vector_2d(f_2d, N,
                   r"$\sin(4\pi i/N)\sin(6\pi j/N)$",
                   "poisson_vector.png")
    save_circuit(circuit, "poisson_circuit.png", scale=0.5)


def fig_finance():
    """Quantum finance: piecewise-constant distribution via LCU of SQUARE bins."""
    print("\n--- Quantum Finance (LCU) ---")
    N = 16
    circuit, info = encode(
        LCU([(0.10, SQUARE(k1=0,  k2=4,  c=1.0)),
             (0.40, SQUARE(k1=4,  k2=8,  c=1.0)),
             (0.35, SQUARE(k1=8,  k2=12, c=1.0)),
             (0.15, SQUARE(k1=12, k2=16, c=1.0))]),
        N=N)
    print_info("encode", info)
    print(f"  success_probability = {info.success_probability:.4f}")
    f = np.zeros(N)
    f[0:4] = 0.10; f[4:8] = 0.40; f[8:12] = 0.35; f[12:16] = 0.15
    plot_vector(f, N,
                r"Price distribution (4 bins), $N=16$",
                "finance_vector.png", ylabel=r"$\sqrt{p_i}$")
    save_circuit(circuit, "finance_circuit.png", scale=0.4)


# ===================================================================
# Figure: gate count vs m (scaling figure)
# ===================================================================

def fig_gate_count_vs_m():
    """
    Gate count vs number of qubits m for each PyEncode pattern,
    compared against Qiskit StatePreparation on a random vector.

    m values: 4, 6, 8, 10, 12  (N = 16 to 4096)

    SQUARE uses non-aligned intervals (k1 = N//4 + 1, k2 = 3*N//4 + 1)
    to reflect realistic usage rather than power-of-2 boundaries.
    """
    print("\n--- Gate Count vs m figure ---")
    print("  (Qiskit transpile at m=16 takes ~3 minutes; patience.)")

    M_VALS = [6, 8, 10, 12, 16]

    patterns = {
        "SPARSE ($s=2$)":  [],
        "STEP":            [],
        "SQUARE":          [],
        "WALSH":           [],
        "GEOMETRIC":       [],
        "POPCOUNT":        [],
        "STAIRCASE":       [],
        "FOURIER":         [],
        "POLYNOMIAL ($d=1$)": [],
        "POLYNOMIAL ($d=2$)": [],
        "Qiskit (random)": [],
    }

    def pyencode_transpile_total(circuit):
        """Transpile PyEncode circuit to {CX, U} and sum all gates,
        for apples-to-apples comparison with Qiskit measurements."""
        from qiskit import transpile
        t = transpile(circuit.decompose(reps=DECOMPOSE_REPS),
                      basis_gates=BASIS_GATES,
                      optimization_level=OPTIMIZATION_LEVEL)
        return sum(t.count_ops().values())

    np.random.seed(42)

    for m in M_VALS:
        N = 2 ** m

        # SPARSE s=2
        idx1 = min(int(0.3 * N) + 3, N - 2)
        idx2 = min(int(0.7 * N) + 5, N - 1)
        c, _ = encode(SPARSE([(idx1, 1.0), (idx2, 2.0)]), N=N)
        patterns["SPARSE ($s=2$)"].append(pyencode_transpile_total(c))

        # STEP: non-power-of-2 cutoff
        c, _ = encode(STEP(k_s=3 * N // 4, c=1.0), N=N)
        patterns["STEP"].append(pyencode_transpile_total(c))

        # SQUARE: non-aligned interval
        k1 = N // 4 + 1
        k2 = 3 * N // 4 + 1
        c, _ = encode(SQUARE(k1=k1, k2=k2, c=1.0), N=N)
        patterns["SQUARE"].append(pyencode_transpile_total(c))

        # WALSH: mid-register bit, generalized
        c, _ = encode(WALSH(k=m // 2, c_pos=1.0, c_neg=4.0), N=N)
        patterns["WALSH"].append(pyencode_transpile_total(c))

        # GEOMETRIC: exponential decay (product state, 0 CX)
        c, _ = encode(GEOMETRIC(ratio=0.95), N=N)
        patterns["GEOMETRIC"].append(pyencode_transpile_total(c))

        # POPCOUNT: product state with Hamming-weight structure (0 CX, depth 1)
        c, _ = encode(POPCOUNT(r=0.7), N=N)
        patterns["POPCOUNT"].append(pyencode_transpile_total(c))

        # STAIRCASE: sparse geometric on unary indices
        c, _ = encode(STAIRCASE(r=0.5), N=N)
        patterns["STAIRCASE"].append(pyencode_transpile_total(c))

        # FOURIER T=1
        c, _ = encode(FOURIER(modes=[(1, 1.0, 0)]), N=N)
        patterns["FOURIER"].append(pyencode_transpile_total(c))

        # POLYNOMIAL d=1 (ramp)
        c, _ = encode(POLYNOMIAL(coeffs=[0.0, 1.0]), N=N)
        patterns["POLYNOMIAL ($d=1$)"].append(pyencode_transpile_total(c))

        # POLYNOMIAL d=2 (Poiseuille)
        c, _ = encode(POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]), N=N)
        patterns["POLYNOMIAL ($d=2$)"].append(pyencode_transpile_total(c))

        # Qiskit baseline: random unit vector
        f_rand = np.random.randn(N)
        qk = qiskit_gates(f_rand, N)
        patterns["Qiskit (random)"].append(qk)

        print(f"  m={m}: "
              f"SPARSE={patterns['SPARSE ($s=2$)'][-1]}, "
              f"STEP={patterns['STEP'][-1]}, "
              f"SQUARE={patterns['SQUARE'][-1]}, "
              f"WALSH={patterns['WALSH'][-1]}, "
              f"GEOMETRIC={patterns['GEOMETRIC'][-1]}, "
              f"POPCOUNT={patterns['POPCOUNT'][-1]}, "
              f"STAIRCASE={patterns['STAIRCASE'][-1]}, "
              f"FOURIER={patterns['FOURIER'][-1]}, "
              f"POLY_d1={patterns['POLYNOMIAL ($d=1$)'][-1]}, "
              f"POLY_d2={patterns['POLYNOMIAL ($d=2$)'][-1]}, "
              f"Qiskit={patterns['Qiskit (random)'][-1]}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    style = {
        "SPARSE ($s=2$)":      dict(color="#2166ac", marker="o",  ls="-",  lw=1.6),
        "STEP":                dict(color="#4dac26", marker="s",  ls="-",  lw=1.6),
        "SQUARE":              dict(color="#d01c8b", marker="^",  ls="-",  lw=1.6),
        "WALSH":               dict(color="#f4a582", marker="D",  ls="-",  lw=1.6),
        "GEOMETRIC":           dict(color="#7570b3", marker="P",  ls="-",  lw=1.6),
        "POPCOUNT":            dict(color="#1b9e77", marker="h",  ls="-",  lw=1.6),
        "STAIRCASE":           dict(color="#e6ab02", marker="<",  ls="-",  lw=1.6),
        "FOURIER":             dict(color="#b2182b", marker="v",  ls="--", lw=1.6),
        "POLYNOMIAL ($d=1$)":  dict(color="#08519c", marker=">",  ls="-.", lw=1.6),
        "POLYNOMIAL ($d=2$)":  dict(color="#6a51a3", marker="*",  ls="-.", lw=1.6),
        "Qiskit (random)":     dict(color="#555555", marker="x",  ls=":",  lw=1.8),
    }

    for label, counts in patterns.items():
        s = style[label]
        ax.semilogy(M_VALS, counts, label=label,
                    color=s["color"], marker=s["marker"],
                    ls=s["ls"], lw=s["lw"], markersize=6)

    ax.set_xlabel("Number of qubits $m$  ($N = 2^m$)")
    ax.set_ylabel("Transpiled gate count  (log scale)")
    ax.set_xticks(M_VALS)
    ax.set_xticklabels([f"$m={m}$\n$N={2**m}$" for m in M_VALS], fontsize=9)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              framealpha=0.9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_title(r"Gate count vs. $m$: PyEncode patterns vs. Qiskit",
                 fontsize=10, pad=6)

    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/gate_count_vs_m.png")
    plt.close(fig)
    print(f"  saved {FIGDIR}/gate_count_vs_m.png")


# ===================================================================
# Gate count table
# ===================================================================

def gate_count_table():
    """Print gate count comparison table (N=64, m=6)."""
    print("\n--- Gate Count Table (N=64, m=6) ---")
    N = 64
    k = np.arange(N)

    x_cts = np.arange(N, dtype=float) / (N - 1)
    # Expected vectors for clarity
    m_bits = int(round(math.log2(N)))
    staircase_vec = np.zeros(N)
    for kk in range(m_bits + 1):
        staircase_vec[(1 << kk) - 1] = 0.5 ** kk
    popcount_vec = np.array([0.7 ** bin(i).count("1") for i in range(N)])

    cases = [
        ("SPARSE s=1 (k=20)",    SPARSE([(20, 1.0)]),
         np.eye(N)[20]),
        ("STEP (k_s=4)",         STEP(k_s=4, c=1.0),
         np.r_[np.ones(4), np.zeros(N-4)]),
        ("SQUARE ([12,52), general)", SQUARE(k1=12, k2=52, c=1.0),
         np.r_[np.zeros(12), np.ones(40), np.zeros(N-52)]),
        ("FOURIER T=1 n=1",      FOURIER(modes=[(1, 1.0, 0)]),
         np.sin(2*np.pi*k/N)),
        ("FOURIER T=1 n=3 phi",  FOURIER(modes=[(3, 1.0, math.pi/4)]),
         np.sin(2*np.pi*3*k/N + math.pi/4)),
        ("SPARSE s=2",           SPARSE([(10, 3.0), (50, 4.0)]),
         np.array([3.0 if i==10 else 4.0 if i==50 else 0.0 for i in range(N)])),
        ("GEOMETRIC (r=0.95)",   GEOMETRIC(ratio=0.95),
         0.95 ** np.arange(N)),
        ("POPCOUNT (r=0.7)",     POPCOUNT(r=0.7),
         popcount_vec),
        ("STAIRCASE (r=0.5)",    STAIRCASE(r=0.5),
         staircase_vec),
        ("POLYNOMIAL d=1 (ramp)",     POLYNOMIAL(coeffs=[0.0, 1.0]),
         x_cts),
        ("POLYNOMIAL d=2 (Poiseuille)", POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]),
         4 * x_cts - 4 * x_cts * x_cts),
        ("POLYNOMIAL d=3",       POLYNOMIAL(coeffs=[0.1, 0.5, 1.0, -0.3]),
         0.1 + 0.5 * x_cts + x_cts ** 2 - 0.3 * x_cts ** 3),
    ]

    print(f"{'Pattern':<32} {'PyEncode':>10} {'Qiskit':>8}  Complexity")
    print("-" * 65)
    for name, vobj, f_vec in cases:
        circuit, info = encode(vobj, N=N)
        qk = qiskit_gates(f_vec, N)
        print(f"{name:<32} {info.gate_count:>10} {qk:>8}  {info.complexity}")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    import os
    os.makedirs(FIGDIR, exist_ok=True)

    # Vector type figures
    fig_sparse_single()
    fig_sparse_two()
    fig_step()
    fig_square()
    fig_fourier_sine()
    fig_fourier_multi()
    fig_geometric()
    fig_popcount()
    fig_staircase()
    fig_polynomial_ramp()
    fig_polynomial_poiseuille()
    fig_tensor()
    fig_lcu_disjoint()
    fig_lcu_overlap()

    # Application figures
    fig_hubbard()
    fig_poisson()
    fig_finance()

    # Gate count scaling figure
    fig_gate_count_vs_m()

    # Gate count table
    gate_count_table()

    print("\nAll figures generated in figures/")