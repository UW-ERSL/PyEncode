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
from pyencode.mps import encode_mps

from pyencode import (
    encode, SPARSE, STEP, SQUARE, FOURIER, WALSH, GEOMETRIC,
    HAMMING, STAIRCASE, DICKE, TENSOR, POLYNOMIAL, SUM,PARTITION
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
    fig.savefig(f"{FIGDIR}/{filename}", dpi=300)
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
    fig.savefig(f"{FIGDIR}/{filename}", dpi=300)
    plt.close(fig)
    print(f"  saved {FIGDIR}/{filename}")


def save_circuit(circuit, filename, scale=0.8, fold=12, dpi=400):
    """Save Qiskit circuit diagram as PNG."""
    fig = circuit.draw('mpl', scale=scale, fold=fold,
    style={'fontsize': 16, 'subfontsize': 14})
    fig.savefig(f"{FIGDIR}/{filename}", bbox_inches='tight', pad_inches=0.05, dpi=dpi)
    plt.close(fig)
    print(f"  saved {FIGDIR}/{filename}")


def print_info(label, info):
    print(f"  {label}: type={info.pattern_name}, gates={info.gate_count}, "
          f"complexity={info.complexity}")


def qiskit_gate_counts(f_vec, N):
    """Return (1q, 2q, depth) for Qiskit StatePreparation on the same vector.

    Matches the {cx, u} basis at optimization_level=3 used throughout the
    rest of this script.  Returns (0, 0, 0) for the zero vector.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation
    norm = np.linalg.norm(f_vec)
    if norm < 1e-14:
        return 0, 0, 0
    sv = (f_vec / norm).astype(complex)
    m = int(round(math.log2(N)))
    qc = QuantumCircuit(m)
    qc.append(StatePreparation(sv), range(m))
    qc = qc.decompose(reps=DECOMPOSE_REPS)
    t = transpile(qc, basis_gates=BASIS_GATES,
                  optimization_level=OPTIMIZATION_LEVEL)
    ops = t.count_ops()
    return ops.get("u", 0), ops.get("cx", 0), t.depth()


def qiskit_gates(f_vec, N):
    """Backwards-compatible wrapper: total transpiled gate count only.

    Prefer :func:`qiskit_gate_counts` for the (1q, 2q, depth) breakdown.
    """
    u, cx, _ = qiskit_gate_counts(f_vec, N)
    return u + cx


# ===================================================================
# Figures: pattern families
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
    print("\n--- SPARSE s=2: f_1=3, f_6=-4, N=8 ---")
    circuit, info = encode(SPARSE([(1, 3.0), (6, -4.0)]), N=8)
    print_info("encode", info)
    f = np.zeros(8); f[1] = 3.0; f[6] = -4.0
    plot_vector(f, 8, r"SPARSE $s=2$: $f_1=3,\;f_6=-4$, $N=8$", "composite_vector.png")
    save_circuit(circuit, "composite_circuit.png", scale=1.0)


def fig_step():
    """STEP: prefix uniform superposition."""
    print("\n--- STEP: k_e=4, N=8 ---")
    circuit, info = encode(STEP(k_e=4, c=1.0), N=8)
    print_info("encode", info)
    f = np.zeros(8); f[:4] = 1.0
    plot_vector(f, 8, r"STEP: $f_{[:4]}=1$, $N=8$", "step_vector.png")
    save_circuit(circuit, "step_circuit.png", scale=0.6)


def fig_square():
    """SQUARE: general interval."""
    print("\n--- SQUARE: [4,8), N=8 ---")
    circuit, info = encode(SQUARE(k_s=4, k_e=8, c=1.0), N=8)
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
                "fourier_sine_vector.png", smooth=True)
    save_circuit(circuit, "fourier_sine_circuit.png", scale=1.0)


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


def fig_walsh():
    """WALSH: two-level piecewise-constant state (generalized, two positive levels)."""
    print("\n--- WALSH: k=2, c0=1, c1=4, N=8 ---")
    N = 8
    k = 2
    c0, c1 = 1.0, 4.0
    circuit, info = encode(WALSH(k=k, c0=c0, c1=c1), N=N)
    print_info("encode", info)
    # Generalized Walsh: f_i = c0 if bit_k(i)==0 else c1; here k=2, so period P = 2^(k+1) = 8.
    # f = [1,1,1,1,4,4,4,4] — single block of each level at N=8.
    idx = np.arange(N)
    f = np.where(((idx >> k) & 1) == 0, c0, c1)
    plot_vector(f, N,
                rf"WALSH: $k=2$ (generalized), $c_0={c0:g}$, $c_1={c1:g}$, $N=8$",
                "walsh_vector.png")
    save_circuit(circuit, "walsh_circuit.png", scale=0.7)


def fig_geometric():
    """GEOMETRIC: exponential decay as product state."""
    print("\n--- GEOMETRIC: ratio=0.5, N=8 ---")
    N = 8
    circuit, info = encode(GEOMETRIC(r=0.5), N=N)
    print_info("encode", info)
    f = 0.5 ** np.arange(N)
    plot_vector(f, N, r"GEOMETRIC: $f_i = 0.5^{\,i}$, $N=8$",
                "geometric_vector.png")
    save_circuit(circuit, "geometric_circuit.png", scale=0.6)

def fig_geometric_planewave():
    """GEOMETRIC with complex r = e^{i*omega}: discrete plane wave.

    Produces two figures matching the LaTeX example in Section 3.6:
      - geometric_planewave_vector.png  : Re and Im parts (stacked bars)
      - geometric_planewave_circuit.png : depth-1 product-state circuit
    """
    import cmath
    print("\n--- GEOMETRIC: r = exp(i*0.7), N=64 (plane wave) ---")
    omega = 0.7
    N = 64
    r = cmath.exp(1j * omega)
    circuit, info = encode(GEOMETRIC(r=r), N=N)
    print_info("GEOMETRIC (plane wave)", info)

    # Build the analytic vector for plotting
    f = (r ** np.arange(N))
    f = f / np.linalg.norm(f)

    # Two-panel Re/Im plot (custom, since plot_vector handles only real f)
    fig, (axr, axi) = plt.subplots(2, 1, figsize=(4.2, 3.2), sharex=True)
    axr.bar(range(N), f.real, width=1.0, color='steelblue',
            edgecolor='steelblue', linewidth=0.3)
    axr.set_ylabel(r"$\mathrm{Re}\,f_i$")
    axr.axhline(0, color='gray', linewidth=0.5)
    axr.set_xlim(-0.5, N - 0.5)
    axi.bar(range(N), f.imag, width=1.0, color='indianred',
            edgecolor='indianred', linewidth=0.3)
    axi.set_ylabel(r"$\mathrm{Im}\,f_i$")
    axi.set_xlabel("Node index $i$")
    axi.axhline(0, color='gray', linewidth=0.5)
    axi.set_xticks(list(range(0, N + 1, N // 4)))
    fig.suptitle(rf"GEOMETRIC: $r=e^{{i\,{omega}}}$, $N={N}$ (plane wave)",
                 fontsize=10, y=0.98)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/geometric_planewave_vector.png")
    plt.close(fig)
    print(f"  saved {FIGDIR}/geometric_planewave_vector.png")

    save_circuit(circuit, "geometric_planewave_circuit.png", scale=0.6)

def fig_geometric_arbitrary():
    """GEOMETRIC with arbitrary k_s: dyadic decomposition path (O(m^2))."""
    print("\n--- GEOMETRIC: ratio=0.8, k_s=5, N=16 ---")
    N = 16
    ratio = 0.8
    k_s = 5
    circuit, info = encode(GEOMETRIC(r=ratio, k_s=k_s), N=N)
    print_info("encode", info)
    # Support: [5, 16) decomposes into [5,6) U [6,8) U [8,16) — three aligned blocks.
    f = np.zeros(N)
    for i in range(k_s, N):
        f[i] = ratio ** (i - k_s)
    plot_vector(f, N,
                rf"GEOMETRIC: $r={ratio:g}$, $\mathrm{{k_s}}={k_s}$, $N={N}$",
                "geometric_arbitrary_vector.png")
    save_circuit(circuit, "geometric_arbitrary_circuit.png", scale=0.5, fold=12)


def fig_hamming():
    """HAMMING: product state, amplitudes depend only on Hamming weight."""
    print("\n--- HAMMING: r=0.5, N=16 ---")
    N = 16
    r = 0.5
    circuit, info = encode(HAMMING(r=r), N=N)
    print_info("encode", info)
    wts = np.array([bin(i).count("1") for i in range(N)], dtype=float)
    f = r ** wts
    plot_vector(f, N, r"HAMMING: $f_i = 0.5^{\,\mathrm{wt}(i)}$, $N=16$",
                "hamming_vector.png")
    save_circuit(circuit, "hamming_circuit.png", scale=0.6)


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


def fig_dicke():
    """DICKE: uniform superposition over all weight-k basis states."""
    print("\n--- DICKE: k=2, N=16 ---")
    N = 16
    k = 2
    circuit, info = encode(DICKE(k=k), N=N)
    print_info("encode", info)
    f = np.array([1.0 if bin(i).count("1") == k else 0.0 for i in range(N)])
    f = f / np.linalg.norm(f)
    plot_vector(f, N,
                r"DICKE: $|D^4_2\rangle$, $N=16$",
                "dicke_vector.png")
    save_circuit(circuit, "dicke_circuit.png", scale=0.5, fold=12)


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
    """SUM: two disjoint SQUARE intervals (ancilla-based; PARTITION is cheaper here)."""
    print("\n--- SUM disjoint: two SQUARE intervals, N=16 ---")
    N = 16
    circuit, info = encode(
        SUM([(1.0, SQUARE(k_s=0, k_e=8, c=1.0)),
             (3.0, SQUARE(k_s=8, k_e=16, c=1.0))]),
        N=N)
    print_info("encode", info)
    print(f"  success_probability = {info.success_probability:.4f}")
    f = np.zeros(N)
    f[:8] = 1.0; f[8:] = 3.0
    plot_vector(f, N,
                r"SUM (disjoint): $f_{[:8]}=1,\;f_{[8:]}=3$, $N=16$",
                "lcu_disjoint_vector.png")
    save_circuit(circuit, "lcu_disjoint_circuit.png", scale=0.5)


def fig_lcu_overlap():
    """SUM: overlapping STEP + FOURIER (requires post-selection)."""
    import warnings
    print("\n--- SUM overlapping: STEP + FOURIER, N=16 ---")
    N = 16
    k = np.arange(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        circuit, info = encode(
            SUM([(1.0, STEP(k_e=16, c=1.0)),
                 (1.0, FOURIER(modes=[(1, 1.0, 0)]))]),
            N=N)
    print_info("encode", info)
    print(f"  success_probability = {info.success_probability:.4f}")
    f = np.ones(N) + np.sin(2 * np.pi * k / N)
    plot_vector(f, N,
                r"SUM (overlap): uniform $+$ $\sin(2\pi i/N)$, $N=16$",
                "lcu_overlap_vector.png", smooth=True)
    save_circuit(circuit, "lcu_overlap_circuit.png", scale=0.4)


# ===================================================================
# Figures: applications
# ===================================================================


def fig_hubbard():
    """Extended Fermi-Hubbard PREP via PARTITION of three disjoint intervals."""
    print("\n--- Extended Fermi-Hubbard (via PARTITION) ---")
    import math
    L = 8
    t = 1.0
    U = 4.0
    V = 0.5
    N = 1 << (3 * L - 1).bit_length()  # next power of two >= 3L

    circuit, info = encode(
        PARTITION([
            STEP(k_e=L,                  c=math.sqrt(t)),
            SQUARE(k_s=L,     k_e=2*L,   c=math.sqrt(U)),
            SQUARE(k_s=2*L,   k_e=3*L,   c=math.sqrt(V)),
        ]),
        N=N)
    print_info("encode", info)

    # Build expected vector for plotting (matches what PARTITION prepares)
    f = np.concatenate([
        np.full(L, math.sqrt(t)),
        np.full(L, math.sqrt(U)),
        np.full(L, math.sqrt(V)),
        np.zeros(N - 3*L),
    ])
    f /= np.linalg.norm(f)

    plot_vector(f,N,
                r"Extended Hubbard: $t{=}1$, $U{=}4$, $V{=}0.5$, $L{=}8$",
                "hubbard_vector.png")
    save_circuit(circuit, "hubbard_circuit.png", scale=0.6, fold=24)

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
    """Log-normal density via MPS — demonstrates encode_mps for finance."""
    print("\n--- Log-normal density (via MPS) ---")
    from pyencode.mps import encode_mps

    m = 16
    N = 2**m
    S0, r, sig, T = 100.0, 0.05, 0.2, 1.0
    mu = np.log(S0) + (r - 0.5 * sig**2) * T

    # Grid spans +/- 3 sigma*sqrt(T) in log-space
    S = np.linspace(S0 * np.exp(-3 * sig * np.sqrt(T)),
                    S0 * np.exp( 3 * sig * np.sqrt(T)), N)
    p = np.exp(-(np.log(S) - mu)**2 / (2 * sig**2 * T)) \
        / (S * sig * np.sqrt(2 * np.pi * T))
    v = np.sqrt(p)
    v /= np.linalg.norm(v)

    circuit, info = encode_mps(v, bond_dim=8, transpile_for_counts=True)
    print_info("encode_mps", info)

    plot_vector(v,N,
                r"Log-normal $\sqrt{p(S)}$: $S_0=100$, $\sigma=0.2$, $T=1$",
                "lognormal_vector.png")
    save_circuit(circuit, "lognormal_circuit.png", scale=0.3)


# ===================================================================
# Figure: gate count vs m (scaling figure)
# ===================================================================

def fig_gate_count_vs_m():
    """
    Gate count vs number of qubits m for each PyEncode pattern,
    compared against Qiskit StatePreparation on a random vector.

    m values: 4, 6, 8, 10, 12  (N = 16 to 4096)

    SQUARE uses non-aligned intervals (k_s = N//4 + 1, k_e = 3*N//4 + 1)
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
        "HAMMING":        [],
        "STAIRCASE":       [],
        "FOURIER":         [],
        "POLYNOMIAL ($d=1$)": [],
        "POLYNOMIAL ($d=2$)": [],
        "Qiskit (Shende)": [],
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
        c, _ = encode(STEP(k_e=3 * N // 4, c=1.0), N=N)
        patterns["STEP"].append(pyencode_transpile_total(c))

        # SQUARE: non-aligned interval
        k_s = N // 4 + 1
        k_e = 3 * N // 4 + 1
        c, _ = encode(SQUARE(k_s=k_s, k_e=k_e, c=1.0), N=N)
        patterns["SQUARE"].append(pyencode_transpile_total(c))

        # WALSH: mid-register bit, generalized
        c, _ = encode(WALSH(k=m // 2, c0=1.0, c1=4.0), N=N)
        patterns["WALSH"].append(pyencode_transpile_total(c))

        # GEOMETRIC: exponential decay (product state, 0 CX)
        c, _ = encode(GEOMETRIC(r=0.95), N=N)
        patterns["GEOMETRIC"].append(pyencode_transpile_total(c))

        # HAMMING: product state with Hamming-weight structure (0 CX, depth 1)
        c, _ = encode(HAMMING(r=0.7), N=N)
        patterns["HAMMING"].append(pyencode_transpile_total(c))

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
        patterns["Qiskit (Shende)"].append(qk)

        print(f"  m={m}: "
              f"SPARSE={patterns['SPARSE ($s=2$)'][-1]}, "
              f"STEP={patterns['STEP'][-1]}, "
              f"SQUARE={patterns['SQUARE'][-1]}, "
              f"WALSH={patterns['WALSH'][-1]}, "
              f"GEOMETRIC={patterns['GEOMETRIC'][-1]}, "
              f"HAMMING={patterns['HAMMING'][-1]}, "
              f"STAIRCASE={patterns['STAIRCASE'][-1]}, "
              f"FOURIER={patterns['FOURIER'][-1]}, "
              f"POLY_d1={patterns['POLYNOMIAL ($d=1$)'][-1]}, "
              f"POLY_d2={patterns['POLYNOMIAL ($d=2$)'][-1]}, "
              f"Qiskit={patterns['Qiskit (Shende)'][-1]}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    style = {
        "SPARSE ($s=2$)":      dict(color="#2166ac", marker="o",  ls="-",  lw=1.6),
        "STEP":                dict(color="#4dac26", marker="s",  ls="-",  lw=1.6),
        "SQUARE":              dict(color="#d01c8b", marker="^",  ls="-",  lw=1.6),
        "WALSH":               dict(color="#f4a582", marker="D",  ls="-",  lw=1.6),
        "GEOMETRIC":           dict(color="#7570b3", marker="P",  ls="-",  lw=1.6),
        "HAMMING":            dict(color="#1b9e77", marker="h",  ls="-",  lw=1.6),
        "STAIRCASE":           dict(color="#e6ab02", marker="<",  ls="-",  lw=1.6),
        "FOURIER":             dict(color="#b2182b", marker="v",  ls="--", lw=1.6),
        "POLYNOMIAL ($d=1$)":  dict(color="#08519c", marker=">",  ls="-.", lw=1.6),
        "POLYNOMIAL ($d=2$)":  dict(color="#6a51a3", marker="*",  ls="-.", lw=1.6),
        "Qiskit (Shende)":     dict(color="#555555", marker="x",  ls=":",  lw=1.8),
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

def fig_gate_count_vs_m_reduced():
    """
    Gate count vs number of qubits m for each PyEncode pattern,
    compared against Qiskit StatePreparation on a random vector.

    m values: 4, 6, 8, 10, 12  (N = 16 to 4096)

    SQUARE uses non-aligned intervals (k_s = N//4 + 1, k_e = 3*N//4 + 1)
    to reflect realistic usage rather than power-of-2 boundaries.
    """
    print("\n--- Gate Count vs m figure ---")
    print("  (Qiskit transpile at m=16 takes ~3 minutes; patience.)")

    M_VALS = [6, 8, 10, 12, 16]

    patterns = {
        "SPARSE ($s=2$)":  [],
        "STEP":            [],
        "WALSH":           [],
        "GEOMETRIC":       [],
        "HAMMING":        [],
        "STAIRCASE":       [],
        "FOURIER":         [],
        "POLYNOMIAL ($d=1$)": [],
        "Qiskit (Shende)": [],
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
        c, _ = encode(STEP(k_e=3 * N // 4, c=1.0), N=N)
        patterns["STEP"].append(pyencode_transpile_total(c))

        # WALSH: mid-register bit, generalized
        c, _ = encode(WALSH(k=m // 2, c0=1.0, c1=4.0), N=N)
        patterns["WALSH"].append(pyencode_transpile_total(c))

        # GEOMETRIC: exponential decay (product state, 0 CX)
        c, _ = encode(GEOMETRIC(r=0.95), N=N)
        patterns["GEOMETRIC"].append(pyencode_transpile_total(c))

        # HAMMING: product state with Hamming-weight structure (0 CX, depth 1)
        c, _ = encode(HAMMING(r=0.7), N=N)
        patterns["HAMMING"].append(pyencode_transpile_total(c))

        # STAIRCASE: sparse geometric on unary indices
        c, _ = encode(STAIRCASE(r=0.5), N=N)
        patterns["STAIRCASE"].append(pyencode_transpile_total(c))

        # FOURIER T=1
        c, _ = encode(FOURIER(modes=[(1, 1.0, 0)]), N=N)
        patterns["FOURIER"].append(pyencode_transpile_total(c))

        # POLYNOMIAL d=1 (ramp)
        c, _ = encode(POLYNOMIAL(coeffs=[0.0, 1.0]), N=N)
        patterns["POLYNOMIAL ($d=1$)"].append(pyencode_transpile_total(c))

        # Qiskit baseline: random unit vector
        f_rand = np.random.randn(N)
        qk = qiskit_gates(f_rand, N)
        patterns["Qiskit (Shende)"].append(qk)

        print(f"  m={m}: "
              f"SPARSE={patterns['SPARSE ($s=2$)'][-1]}, "
              f"STEP={patterns['STEP'][-1]}, "
              f"WALSH={patterns['WALSH'][-1]}, "
              f"GEOMETRIC={patterns['GEOMETRIC'][-1]}, "
              f"HAMMING={patterns['HAMMING'][-1]}, "
              f"STAIRCASE={patterns['STAIRCASE'][-1]}, "
              f"FOURIER={patterns['FOURIER'][-1]}, "
              f"POLY_d1={patterns['POLYNOMIAL ($d=1$)'][-1]}, "
              f"Qiskit={patterns['Qiskit (Shende)'][-1]}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    style = {
        "SPARSE ($s=2$)":      dict(color="#2166ac", marker="o",  ls="-",  lw=1.6),
        "STEP":                dict(color="#4dac26", marker="s",  ls="-",  lw=1.6),
        "WALSH":               dict(color="#f4a582", marker="D",  ls="-",  lw=1.6),
        "GEOMETRIC":           dict(color="#7570b3", marker="P",  ls="-",  lw=1.6),
        "HAMMING":            dict(color="#1b9e77", marker="h",  ls="-",  lw=1.6),
        "STAIRCASE":           dict(color="#e6ab02", marker="<",  ls="-",  lw=1.6),
        "FOURIER":             dict(color="#b2182b", marker="v",  ls="--", lw=1.6),
        "POLYNOMIAL ($d=1$)":  dict(color="#08519c", marker=">",  ls="-.", lw=1.6),
        "Qiskit (Shende)":     dict(color="#555555", marker="x",  ls=":",  lw=1.8),
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
    fig.savefig(f"{FIGDIR}/gate_count_vs_m_reduced.png")
    plt.close(fig)
    print(f"  saved {FIGDIR}/gate_count_vs_m_reduced.png")

def fig_mps_gaussian():
    """MPS encoding of a discretized Gaussian (canonical hard case)."""

    print("\n--- MPS: Gaussian, alpha=50, chi=4, N=256 ---")
    N = 256
    i = np.arange(N)
    alpha = 50.0
    v = np.exp(-alpha * ((i - N/2) / N) ** 2)
    v /= np.linalg.norm(v)
    circuit, info = encode_mps(v, bond_dim=8,validate=True,transpile_for_counts=True)
    print(info)
    print(f"  truncation_error_sq = {info.params['truncation_error_sq']:.2e}")
    print(f"  gate_count_2q       = {info.gate_count_2q}")
    plot_vector(v, N,
                r"MPS: Gaussian $\exp(-\alpha(i-N/2)^2/N^2)$, $\alpha=50$, $N=256$",
                f"mps_gaussian_vector.png", smooth=True)
    save_circuit(circuit, f"mps_gaussian_circuit.png", scale=0.4)

# ===================================================================
# Gate count table
# ===================================================================

def gate_count_table():
    """Reproduce Table 2 from the paper: gate counts and circuit depth
    at N = 4096 (m = 12), transpiled to {cx, u} at optimization_level=3.

    Every row reports (1q, 2q, depth) triples for both PyEncode and
    Qiskit StatePreparation.  The 15 rows exactly match the 15 rows in
    the paper's Table 2.
    """
    print("\n--- Gate Count Table  (N = 4096, m = 12) ---")
    N = 4096
    m = int(round(math.log2(N)))
    k = np.arange(N)
    x_cts = k.astype(float) / (N - 1)

    # Analytic vectors used to give Qiskit an equal-footing target.
    hamming_vec   = np.array([0.7 ** bin(i).count("1") for i in range(N)])
    staircase_vec = np.zeros(N)
    for j in range(m + 1):
        staircase_vec[(1 << j) - 1] = 0.5 ** j

    def dicke_vec(kk):
        return np.array([1.0 if bin(i).count("1") == kk else 0.0
                         for i in range(N)])

    cases = [
        # ---- SPARSE ----
        ("Sparse s=1, k=N/4",
         SPARSE([(N // 4, 1.0)]),
         np.eye(N)[N // 4]),
        ("Sparse s=2",
         SPARSE([(10, 3.0), (50, 4.0)]),
         np.array([3.0 if i == 10 else 4.0 if i == 50 else 0.0
                   for i in range(N)])),
        # ---- STEP ----
        ("Step k_e=N/2",
         STEP(k_e=N // 2, c=1.0),
         np.r_[np.ones(N // 2), np.zeros(N // 2)]),
        # ---- SQUARE (general non-aligned interval) ----
        ("Square [N/4+1, 3N/4+1)",
         SQUARE(k_s=N // 4 + 1, k_e=3 * N // 4 + 1, c=1.0),
         np.r_[np.zeros(N // 4 + 1),
               np.ones(N // 2),
               np.zeros(N // 4 - 1)]),
        # ---- WALSH ----
        ("Walsh k=6, c0=1, c1=4",
         WALSH(k=6, c0=1.0, c1=4.0),
         np.where((k >> 6) & 1, 4.0, 1.0).astype(float)),
        # ---- GEOMETRIC ----
        ("Geometric r=0.95",
         GEOMETRIC(r=0.95),
         0.95 ** k),
        # ---- HAMMING ----
        ("Hamming r=0.7",
         HAMMING(r=0.7),
         hamming_vec),
        # ---- STAIRCASE ----
        ("Staircase r=0.5",
         STAIRCASE(r=0.5),
         staircase_vec),
        # ---- DICKE (both k values from the paper) ----
        ("Dicke k=2",
         DICKE(k=2),
         dicke_vec(2)),
        ("Dicke k=11",
         DICKE(k=11),
         dicke_vec(11)),
        # ---- POLYNOMIAL ----
        ("Polynomial d=1 (ramp)",
         POLYNOMIAL(coeffs=[0.0, 1.0]),
         x_cts),
        ("Polynomial d=2 (Poiseuille)",
         POLYNOMIAL(coeffs=[0.0, 4.0, -4.0]),
         4 * x_cts * (1 - x_cts)),
        # ---- FOURIER ----
        ("Fourier T=1 n=1 phi=0",
         FOURIER(modes=[(1, 1.0, 0.0)]),
         np.sin(2 * np.pi * k / N)),
        ("Fourier T=1 n=3 phi=pi/4",
         FOURIER(modes=[(3, 1.0, math.pi / 4)]),
         np.sin(2 * np.pi * 3 * k / N + math.pi / 4)),
        ("Fourier T=2",
         FOURIER(modes=[(1, 1.0, 0.0), (3, 0.5, 0.0)]),
         np.sin(2 * np.pi * k / N) + 0.5 * np.sin(2 * np.pi * 3 * k / N)),
    ]

    print(f"{'Pattern':<28}{'PyEncode':>19}{'Qiskit':>19}  {'Complexity':<14}")
    print(f"{'':<28}{'1q/2q/depth':>19}{'1q/2q/depth':>19}")
    print("-" * (28 + 19 + 19 + 16))
    for name, pat, f_vec in cases:
        _, info = encode(pat, N=N)
        pe_str = f"{info.gate_count_1q}/{info.gate_count_2q}/{info.circuit_depth}"
        qi1, qi2, qid = qiskit_gate_counts(f_vec, N)
        qi_str = f"{qi1}/{qi2}/{qid}"
        print(f"{name:<28}{pe_str:>19}{qi_str:>19}  {info.complexity}")


# ===================================================================
# Main
# ===================================================================


if __name__ == "__main__":
    import os
    os.makedirs(FIGDIR, exist_ok=True)

    # Pattern figures
    fig_sparse_single()
    fig_sparse_two()
    fig_step()
    fig_square()
    fig_fourier_sine()
    fig_fourier_multi()
    fig_walsh()
    fig_geometric()
    fig_geometric_planewave()
    fig_geometric_arbitrary()
    fig_hamming()
    fig_staircase()
    fig_dicke()
    fig_polynomial_ramp()
    fig_polynomial_poiseuille()
    fig_tensor()
    fig_lcu_disjoint()
    fig_lcu_overlap()
    fig_mps_gaussian()

    # Application figures
    fig_hubbard()
    fig_poisson()
    fig_finance()

    gen_tables = False

    if gen_tables: # Takes time to run
        # Gate count scaling figure
        fig_gate_count_vs_m()

        # Gate count table
        gate_count_table()

    print("\nAll figures generated in figures/")