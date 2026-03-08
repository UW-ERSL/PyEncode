"""
demo.py  —  PyEncode demonstration
====================================
All examples use N=64 (m=6 qubits): the smallest discretisation
relevant in practice for FEM/FDM problems.

Run with:  python demo.py

Generates:
  figures/load_point.png
  figures/load_uniform.png
  figures/load_step.png  (square load)
  figures/load_sin.png
  figures/load_disjoint.png
  figures/load_uniform_spike.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from qiskit.circuit.library import StatePreparation
from qiskit import QuantumCircuit, transpile
from pyencode import encode

SEP   = "─" * 60
N     = 64
m     = 6
BASIS = ['cx', 'u', 'x', 'h', 'ry', 'rz', 'rx', 'p']

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FIG_W  = 3.2   # inches — width for both load and circuit figures
FIG_H  = 2.8   # inches — height for both (matched for proportion)
FSIZE  = 16    # base font size for load figures
DPI    = 150

# ── Plotting helper ───────────────────────────────────────────────────────

def save_load_figure(f, filename, title, ylabel="f (N/m)",
                     style="bar", highlight=None):
    """
    Save a force vector as a clean bar or line plot.

    Parameters
    ----------
    f         : array-like, length N
    filename  : output filename (saved into FIG_DIR)
    title     : figure title
    ylabel    : y-axis label
    style     : 'bar' for sparse loads, 'line' for continuous loads
    highlight : list of indices to mark in red (optional)
    """
    x = np.arange(len(f))
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    if style == "bar":
        colors = ["#c0392b" if (highlight and i in highlight) else "#2c7bb6"
                  for i in x]
        ax.bar(x, f, color=colors, width=0.8, linewidth=0)
    else:
        ax.plot(x, f, color="#2c7bb6", linewidth=2.0)
        ax.fill_between(x, f, alpha=0.15, color="#2c7bb6")
        if highlight:
            for idx in highlight:
                ax.axvline(idx, color="#c0392b", linewidth=1.2,
                           linestyle="--", alpha=0.8)

    ax.set_xlabel("Node index $i$", fontsize=FSIZE)
    ax.set_ylabel(ylabel, fontsize=FSIZE)
    ax.tick_params(labelsize=FSIZE - 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(16))
    ax.set_xlim(-1, len(f))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.5)
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}  ({FIG_W:.1f} x {FIG_H:.1f} in)")


# ── Generate all load figures ─────────────────────────────────────────────

x = np.linspace(0, 1.0, N)

# 1. Point load
f_point = np.zeros(N); f_point[21] = 1.0
save_load_figure(f_point, "load_point.png",
                 "Point Load  $f_{21} = 1$",
                 style="spike", highlight=[21])

# 2. Uniform load
f_uniform = np.ones(N) * 2.0
save_load_figure(f_uniform, "load_uniform.png",
                 "Uniform Load  $f_i = c$",
                 style="box")

# 3. Square load (nonzero over middle segment)
f_step = np.zeros(N); f_step[16:48] = 1.0
f_step /= np.linalg.norm(f_step)
save_load_figure(f_step, "load_step.png",
                 r"Square Load  $f_i = 1/\sqrt{32}$ for $16 \leq i < 48$",
                 style="box")

# 4. Sinusoidal n=1, phi=0  (circuit encodes sin(2*pi*k/N))
f_sin = np.sin(2 * np.pi * x)
f_sin /= np.linalg.norm(f_sin)
save_load_figure(f_sin, "load_sin.png",
                 r"Sinusoidal Load  $n=1$, $\varphi=0$",
                 style="line")

# 4b. Sinusoidal n=3, phi=pi/4
f_sin_phase = np.sin(3 * np.pi * x + np.pi/4)
f_sin_phase /= np.linalg.norm(f_sin_phase)
save_load_figure(f_sin_phase, "load_sin_phase.png",
                 r"Sinusoidal Load  $n=3$, $\varphi=\pi/4$",
                 style="line")


# 4c. Cosine n=1, phi=0
f_cos = np.cos(np.pi * x)
f_cos /= np.linalg.norm(f_cos)
save_load_figure(f_cos, "load_cos.png",
                 r"Cosine Load  $n=1$, $\varphi=0$",
                 style="line")

# 4d. Cosine n=3, phi=pi/4
f_cos_phase = np.cos(3 * np.pi * x + np.pi/4)
f_cos_phase /= np.linalg.norm(f_cos_phase)
save_load_figure(f_cos_phase, "load_cos_phase.png",
                 r"Cosine Load  $n=3$, $\varphi=\pi/4$",
                 style="line")

# 5. Disjoint point loads
f_disjoint = np.zeros(N); f_disjoint[10] = 3.0; f_disjoint[50] = 4.0
f_disjoint /= np.linalg.norm(f_disjoint)
save_load_figure(f_disjoint, "load_disjoint.png",
                 r"Two Point Loads  $f_{10}=3/5,\ f_{50}=4/5$",
                 style="spike", highlight=[10, 50])

# 6. Uniform + spike
f_uniform_spike = np.ones(N) * 1.0; f_uniform_spike[21] = 8.0
f_uniform_spike /= np.linalg.norm(f_uniform_spike)
save_load_figure(f_uniform_spike, "load_uniform_spike.png",
                 r"Uniform + Spike (normalized)",
                 style="box", highlight=[21])

print()

# ── Console demo ──────────────────────────────────────────────────────────

def show(title, code, fallback_vector=None):
    print(f"\n{SEP}\n  {title}\n{SEP}")
    print("  Code:")
    for line in code.strip().splitlines():
        print(f"    {line}")
    print()
    try:
        circuit, info = encode(code, fallback_vector=fallback_vector)
        print(info)
        print("\n  Circuit:")
        print(circuit.draw("text"))
    except Exception as e:
        print(f"  ERROR: {e}")
    print()


def shende_gates(fv):
    fv = np.array(fv, dtype=complex)
    fv /= np.linalg.norm(fv)
    sp = StatePreparation(fv)
    qc = QuantumCircuit(m); qc.append(sp, range(m))
    qc = qc.decompose(reps=10)
    t = transpile(qc, basis_gates=BASIS, optimization_level=0)
    return sum(t.count_ops().values())


show("1. Point load  f[21] = 5.0  (N=64)", """
import numpy as np
N = 64
f = np.zeros(N)
f[21] = 1.0
""")

show("2. Uniform distributed load  f = ones(N)  (N=64)", """
import numpy as np
N = 64
f = np.ones(N)
""")

show("3. Square load  f[16:48] = 2.0  (N=64)", """
import numpy as np
N = 64
f = np.zeros(N)
f[16:48] = 1.0
""")

show("4. Sinusoidal load  n=1, phi=0  (N=64)", """
import numpy as np
N = 64
x = np.linspace(0, 1, N)
f = np.sin(np.pi * x)
""")

show("4b. Sinusoidal  n=3, phi=pi/4  (N=64)", """
import numpy as np
N = 64
x = np.linspace(0, 1, N)
f = np.sin(3 * np.pi * x + np.pi / 4)
""")


show("4c. Cosine load  n=1, phi=0  (N=64)", """
import numpy as np
N = 64
x = np.linspace(0, 1, N)
f = np.cos(np.pi * x)
""")

show("4d. Cosine  n=3, phi=pi/4  (N=64)", """
import numpy as np
N = 64
x = np.linspace(0, 1, N)
f = np.cos(3 * np.pi * x + np.pi / 4)
""")

show("5. Disjoint point loads  f[10]=3, f[50]=4  (N=64)", """
import numpy as np
N = 64
f = np.zeros(N)
f[10] = 3.0
f[50] = 4.0
""")

show("6. Uniform + spike  (N=64)", """
import numpy as np
N = 64
f = np.ones(N) * 1.0
f[21] = 8.0
""")

print(f"\n{SEP}\n  7. Unrecognised pattern  →  Shende fallback  (N=64)\n{SEP}")
rng = np.random.default_rng(42)
fallback_f = rng.random(N); fallback_f /= np.linalg.norm(fallback_f)
show("7. Shende fallback",
     "# arbitrarily structured vector — pattern not in supported class",
     fallback_vector=fallback_f)

# ── Gate count comparison ─────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  Gate count comparison: PyEncode vs Shende  (N={N}, m={m} qubits)")
print(SEP)
print(f"  Note: Shende column = StatePreparation applied to the same vector.")
print()

cases = [
    ("Point load",
     f"import numpy as np\nN={N}\nf=np.zeros(N)\nf[21]=1.0",
     f_point),
    ("Uniform load",
     f"import numpy as np\nN={N}\nf=np.ones(N)",
     f_uniform),
    ("Square load (16:48)",
     f"import numpy as np\nN={N}\nf=np.zeros(N)\nf[16:48]=1.0",
     f_step),
    ("Sinusoidal n=1",
     f"import numpy as np\nN={N}\nx=np.linspace(0,1,N)\nf=np.sin(2*np.pi*x)",
     f_sin),
    ("Sinusoidal n=3, phi=pi/4",
     f"import numpy as np\nN={N}\nx=np.linspace(0,1,N)\nf=np.sin(2*3*np.pi*x+np.pi/4)",
     f_sin_phase),
    ("Disjoint point loads",
     f"import numpy as np\nN={N}\nf=np.zeros(N)\nf[10]=3.0\nf[50]=4.0",
     f_disjoint),
    ("Uniform + spike",
     f"import numpy as np\nN={N}\nf=np.ones(N)\nf[21]=8.0",
     f_uniform_spike),
]

print(f"  {'Pattern':<22}  {'PyEncode':>9}  {'Shende':>8}  Complexity")
print(f"  {'─'*22}  {'─'*9}  {'─'*8}  {'─'*30}")
for name, code, fv in cases:
    _, info = encode(code)
    g = info.gate_count
    s = shende_gates(fv)
    print(f"  {name:<22}  {g:>9}  {s:>8}  {info.gate_complexity}")

print(f"\n  All gate counts after transpilation to {{cx, u, x, h}}.")
print()

# ── Generate circuit figures ──────────────────────────────────────────────

def save_circuit_figure(code, filename, decompose_reps=0, fallback_vector=None):
    """
    Draw the PyEncode circuit to PNG using Qiskit's matplotlib backend.

    For simple circuits (point, uniform, square, disjoint point) decompose_reps=0
    gives the cleanest drawing. For circuits containing high-level gates
    (QFTGate, StatePreparation) we leave them as named boxes.
    """
    circuit, info = encode(code, fallback_vector=fallback_vector)
    qc = circuit.decompose(reps=decompose_reps) if decompose_reps > 0 else circuit

    fig = qc.draw(
        'mpl',
        style='iqp',
        fold=-1,               # no line folding — single row
        plot_barriers=False,
        scale=0.5,             # native downscale — preserves font quality
    )

    import matplotlib.pyplot as _plt
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    _plt.close(fig)
    print(f"  Saved: {path}")


print("Generating circuit figures...")

save_circuit_figure(
    "import numpy as np\nN=64\nf=np.zeros(N)\nf[21]=1.0",
    "circuit_point.png")

save_circuit_figure(
    "import numpy as np\nN=64\nf=np.ones(N)",
    "circuit_uniform.png")

save_circuit_figure(
    "import numpy as np\nN=64\nf=np.zeros(N)\nf[16:48]=1.0",
    "circuit_step.png")

# Sinusoidal: keep QFTGate as a named box — readable at column width
save_circuit_figure(
    "import numpy as np\nN=64\nx=np.linspace(0,1,N)\nf=np.sin(np.pi*x)",
    "circuit_sin.png")

save_circuit_figure(
    "import numpy as np\nN=64\nx=np.linspace(0,1,N)\nf=np.sin(3*np.pi*x+np.pi/4)",
    "circuit_sin_phase.png")

save_circuit_figure(
    "import numpy as np\nN=64\nx=np.linspace(0,1,N)\nf=np.cos(np.pi*x)",
    "circuit_cos.png")

save_circuit_figure(
    "import numpy as np\nN=64\nx=np.linspace(0,1,N)\nf=np.cos(3*np.pi*x+np.pi/4)",
    "circuit_cos_phase.png")

save_circuit_figure(
    "import numpy as np\nN=64\nf=np.zeros(N)\nf[10]=3.0\nf[50]=4.0",
    "circuit_disjoint.png")

# Uniform spike: StatePreparation shown as a named box
save_circuit_figure(
    "import numpy as np\nN=64\nf=np.ones(N)*1.0\nf[21]=8.0",
    "circuit_uniform_spike.png")

print()