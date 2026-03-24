"""
pyencode.recognizer
=====================
AST-based pattern recognizer for structured load vector construction code.

Each recognizer function inspects the parsed AST of user-provided Python
code and attempts to identify a known load pattern, returning a
LoadPattern dataclass if successful, or None otherwise.

Supported patterns
------------------
  POINT_LOAD      : f[k] = P  (single node)
  UNIFORM_LOAD    : f = ones(N) * c
  STEP_LOAD       : f[:k] = c  (prefix slice)
  SQUARE_LOAD     : f[k1:k2] = c  (interior segment)
  SINUSOIDAL_LOAD : f = sin(2*pi*n*k/N + phi)  (single mode, optional phase)
  COSINE_LOAD     : f = cos(2*pi*n*k/N + phi)  (single mode, optional phase)
  MULTI_POINT_LOAD: multiple f[k_i] = P_i with L >= 2, arbitrary weights
  MULTI_SIN_LOAD  : sum of sinusoidal modes (same Fourier basis)
  UNIFORM_SPIKE_LOAD: uniform + single point perturbation (rank-1)

References
----------
  Möttönen et al., "Transformation of quantum states using uniformly
    controlled rotations", Quantum Inf. Comput. 5(6), 2005.
  Shende, Markov, Bullock, "Synthesis of quantum-logic circuits",
    IEEE TCAD 25(6), 2006.
"""

import ast
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class VectorType(Enum):
    DISCRETE        = auto()
    UNIFORM         = auto()
    STEP            = auto()
    SQUARE          = auto()   # f[k1:k2] = c  (interior segment)
    SINE            = auto()
    COSINE          = auto()
    MULTI_DISCRETE  = auto()   # L >= 2 point loads, arbitrary weights
    MULTI_SINE      = auto()   # sum of sinusoidal modes
    UNIFORM_SPIKE   = auto()   # uniform + point perturbation (internal)
    UNKNOWN         = auto()   # fallback to Mottonen

# Backward-compatible alias
LoadType = VectorType


@dataclass
class LoadPattern:
    """Recognised load pattern with extracted numerical parameters."""
    load_type: LoadType
    N: int                        # number of nodes (must be power of 2)
    params: dict = field(default_factory=dict)
    # params keys depend on load_type — documented per pattern below


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def recognize(code: str) -> LoadPattern:
    """
    Parse *code* and return the best-matching LoadPattern.

    Parameters
    ----------
    code : str
        Python source that constructs a load vector, typically ending
        with an assignment to a variable named ``f``.

    Returns
    -------
    LoadPattern
        If recognized, load_type is one of the structured types and
        params carries the extracted parameters.  If not recognized,
        load_type is UNKNOWN and params is empty.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return LoadPattern(VectorType.UNKNOWN, N=0)

    ctx = _ExecutionContext(code)

    # Run a lightweight symbolic executor to collect assignments
    ctx.visit(tree)

    N = ctx.N
    if N is None or N == 0 or (N & (N - 1)) != 0:
        # N unknown or not a power of two — cannot encode
        return LoadPattern(VectorType.UNKNOWN, N=N or 0)

    # Try patterns in order of specificity
    for recognizer_fn in [
        _try_uniform_spike_load,          # must come before point load (also has subscript assign)
        _try_point_load,
        _try_uniform_load,
        _try_step_load,
        _try_square_load,
        _try_sinusoidal_load,
        _try_cosine_load,
        _try_multi_point_load,
        _try_multi_sin_load,
    ]:
        result = recognizer_fn(ctx)
        if result is not None:
            return result

    return LoadPattern(VectorType.UNKNOWN, N=N)


# Backward-compatible alias (British spelling kept for v0.4 compatibility)
recognise = recognize  # noqa: E305


# ---------------------------------------------------------------------------
# Lightweight symbolic executor
# ---------------------------------------------------------------------------

class _ExecutionContext(ast.NodeVisitor):
    """
    Walk the AST and collect:
      - self.N          : inferred grid size
      - self.assignments: list of (target_desc, value_desc) tuples
      - self.f_name     : name of the load vector variable
    """

    def __init__(self, source: str):
        self.source = source
        self.N: Optional[int] = None
        self.assignments: list[dict] = []
        self.f_name: Optional[str] = None
        self._vars: dict = {}   # simple constant folding

    # ------------------------------------------------------------------
    # Node visitors
    # ------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign):
        value_desc = self._eval_expr(node.value)

        for target in node.targets:
            target_desc = self._describe_target(target)
            if target_desc is None:
                continue

            # Track simple integer / float variable bindings
            if isinstance(target, ast.Name):
                if value_desc is not None and isinstance(value_desc.get("value"), (int, float)):
                    self._vars[target.id] = value_desc["value"]
                # Detect N = <integer>
                if target.id in ("N", "n_nodes", "ndof", "num_nodes"):
                    if value_desc and isinstance(value_desc.get("value"), int):
                        self.N = int(value_desc["value"])
                # Detect the load vector variable name.
                # Heuristic: variable named 'f' (or 'force', 'load') that is
                # assigned from a structured expression is likely the vector.
                # Also catch: f = ones * c, f = sin_call, f = A * sin_call
                _is_vector_init = False
                if value_desc:
                    kind = value_desc.get("kind")
                    if kind in ("zeros", "ones", "linspace"):
                        _is_vector_init = True
                    elif kind in ("sin_call", "cos_call"):
                        _is_vector_init = True
                    elif kind == "binop":
                        # A * sin_call  or  ones * c
                        left = value_desc.get("left") or {}
                        right = value_desc.get("right") or {}
                        if left.get("kind") in ("sin_call", "cos_call", "ones") or \
                           right.get("kind") in ("sin_call", "cos_call", "ones"):
                            _is_vector_init = True
                if _is_vector_init and target.id not in ("N", "n_nodes", "ndof",
                                                          "num_nodes", "x", "y",
                                                          "z", "L", "dx"):
                    self.f_name = target.id

            # Track subscript / slice assignments  f[k] = v
            if isinstance(target, ast.Subscript):
                base = self._name(target.value)
                if base and (self.f_name is None or base == self.f_name):
                    self.f_name = base
                self.assignments.append({
                    "kind": "subscript_assign",
                    "base": base,
                    "index": self._eval_slice(target.slice),
                    "value": value_desc,
                })
                self.generic_visit(node)
                return

            # Track f = zeros(N) / ones(N) style initialisation
            if isinstance(target, ast.Name) and value_desc:
                self.assignments.append({
                    "kind": "name_assign",
                    "name": target.id,
                    "value": value_desc,
                })

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Expression evaluator — returns a descriptor dict or None
    # ------------------------------------------------------------------

    def _eval_expr(self, node) -> Optional[dict]:
        if node is None:
            return None

        # Numeric literal
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return {"kind": "number", "value": node.value}

        # Name reference — constant folding
        if isinstance(node, ast.Name):
            val = self._vars.get(node.id)
            if val is not None:
                return {"kind": "number", "value": val}
            return {"kind": "name", "id": node.id}

        # Unary minus
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._eval_expr(node.operand)
            if inner and isinstance(inner.get("value"), (int, float)):
                return {"kind": "number", "value": -inner["value"]}

        # Binary operations  a * b,  a / b,  a + b
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node)

        # Function call  np.zeros(N),  np.sin(...),  etc.
        if isinstance(node, ast.Call):
            return self._eval_call(node)

        # Attribute  np.pi
        if isinstance(node, ast.Attribute):
            if self._name(node.value) == "np" and node.attr == "pi":
                return {"kind": "number", "value": math.pi}
            if self._name(node.value) == "math" and node.attr == "pi":
                return {"kind": "number", "value": math.pi}

        return None

    def _eval_binop(self, node: ast.BinOp) -> Optional[dict]:
        left  = self._eval_expr(node.left)
        right = self._eval_expr(node.right)

        # Both are plain numbers — fold
        if (left and right and
                isinstance(left.get("value"), (int, float)) and
                isinstance(right.get("value"), (int, float))):
            lv, rv = left["value"], right["value"]
            if isinstance(node.op, ast.Mult):  return {"kind": "number", "value": lv * rv}
            if isinstance(node.op, ast.Div):   return {"kind": "number", "value": lv / rv}
            if isinstance(node.op, ast.Add):   return {"kind": "number", "value": lv + rv}
            if isinstance(node.op, ast.Sub):   return {"kind": "number", "value": lv - rv}

        return {"kind": "binop", "op": type(node.op).__name__, "left": left, "right": right}

    def _eval_call(self, node: ast.Call) -> Optional[dict]:
        func_name = self._func_name(node)

        # np.zeros(N) or zeros(N)
        if func_name in ("zeros", "np.zeros"):
            N = self._extract_int_arg(node, 0)
            if N: self.N = N
            return {"kind": "zeros", "N": N}

        # np.ones(N)
        if func_name in ("ones", "np.ones"):
            N = self._extract_int_arg(node, 0)
            if N: self.N = N
            return {"kind": "ones", "N": N}

        # np.linspace(0, L, N)
        if func_name in ("linspace", "np.linspace"):
            N = self._extract_int_arg(node, 2)
            if N: self.N = N
            return {"kind": "linspace", "N": N}

        # np.sin(expr)
        if func_name in ("sin", "np.sin"):
            arg = self._eval_expr(node.args[0]) if node.args else None
            return {"kind": "sin_call", "arg": arg}

        # np.cos(expr)
        if func_name in ("cos", "np.cos"):
            arg = self._eval_expr(node.args[0]) if node.args else None
            return {"kind": "cos_call", "arg": arg}

        # np.sin applied to array  (detected as sin_array downstream)
        return {"kind": "call", "func": func_name}

    def _eval_slice(self, node) -> dict:
        """Return a descriptor for an index or slice."""
        # Simple integer index
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return {"kind": "index", "value": node.value}

        # Name-based index
        if isinstance(node, ast.Name):
            val = self._vars.get(node.id)
            if isinstance(val, int):
                return {"kind": "index", "value": val}
            return {"kind": "name_index", "id": node.id}

        # Arithmetic index  (e.g.  N//2)
        if isinstance(node, ast.BinOp):
            ev = self._eval_expr(node)
            if ev and isinstance(ev.get("value"), (int, float)):
                return {"kind": "index", "value": int(ev["value"])}

        # Slice  e.g.  :k  or  k:
        if isinstance(node, ast.Slice):
            lower = self._eval_expr(node.lower) if node.lower else None
            upper = self._eval_expr(node.upper) if node.upper else None
            return {
                "kind": "slice",
                "lower": lower["value"] if lower and "value" in lower else None,
                "upper": upper["value"] if upper and "value" in upper else None,
            }

        return {"kind": "unknown"}

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _describe_target(self, target) -> Optional[dict]:
        if isinstance(target, ast.Name):
            return {"kind": "name", "id": target.id}
        if isinstance(target, ast.Subscript):
            return {"kind": "subscript"}
        return None

    def _name(self, node) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._name(node.value)
            if base:
                return f"{base}.{node.attr}"
        return None

    def _func_name(self, node: ast.Call) -> str:
        return self._name(node.func) or ""

    def _extract_int_arg(self, node: ast.Call, pos: int) -> Optional[int]:
        if len(node.args) <= pos:
            return None
        ev = self._eval_expr(node.args[pos])
        if ev and isinstance(ev.get("value"), (int, float)):
            return int(ev["value"])
        # Try looking up variable
        if isinstance(node.args[pos], ast.Name):
            val = self._vars.get(node.args[pos].id)
            if isinstance(val, (int, float)):
                return int(val)
        return None


# ---------------------------------------------------------------------------
# Individual pattern matchers
# ---------------------------------------------------------------------------

def _try_point_load(ctx: _ExecutionContext) -> Optional[LoadPattern]:
    """
    Pattern: f[k] = P
    The vector is initialised to zero and exactly one index is set.
    """
    subscript_assigns = [a for a in ctx.assignments
                         if a["kind"] == "subscript_assign"
                         and a.get("base") == ctx.f_name]
    if len(subscript_assigns) != 1:
        return None

    idx_desc = subscript_assigns[0]["index"]
    val_desc = subscript_assigns[0]["value"]

    if idx_desc.get("kind") != "index":
        return None
    if val_desc is None or not isinstance(val_desc.get("value"), (int, float)):
        return None

    k = int(idx_desc["value"])
    P = float(val_desc["value"])
    N = ctx.N
    if N is None or k >= N:
        return None

    return LoadPattern(VectorType.DISCRETE, N=N, params={"k": k, "P": P})


def _try_uniform_load(ctx: _ExecutionContext) -> Optional[LoadPattern]:
    """
    Pattern: f = ones(N) * c   (or just ones(N))
    No subscript assignments allowed.
    """
    subscript_assigns = [a for a in ctx.assignments
                         if a["kind"] == "subscript_assign"
                         and a.get("base") == ctx.f_name]
    if subscript_assigns:
        return None

    name_assigns = [a for a in ctx.assignments
                    if a["kind"] == "name_assign"
                    and a["name"] == ctx.f_name]
    if not name_assigns:
        return None

    last = name_assigns[-1]["value"]
    if last is None:
        return None

    # ones(N) * c
    if last.get("kind") == "binop" and last.get("op") == "Mult":
        left, right = last.get("left"), last.get("right")
        c = None
        if left and left.get("kind") == "ones":
            c = right.get("value") if right else 1.0
        elif right and right.get("kind") == "ones":
            c = left.get("value") if left else 1.0
        if c is not None and ctx.N:
            return LoadPattern(VectorType.UNIFORM, N=ctx.N,
                               params={"c": float(c)})

    # bare ones(N)
    if last.get("kind") == "ones" and ctx.N:
        return LoadPattern(VectorType.UNIFORM, N=ctx.N, params={"c": 1.0})

    return None


def _try_step_load(ctx: _ExecutionContext) -> Optional[LoadPattern]:
    """
    Pattern: f[:k_s] = c   (prefix slice assignment)
    """
    subscript_assigns = [a for a in ctx.assignments
                         if a["kind"] == "subscript_assign"
                         and a.get("base") == ctx.f_name]
    if len(subscript_assigns) != 1:
        return None

    idx_desc = subscript_assigns[0]["index"]
    val_desc = subscript_assigns[0]["value"]

    if idx_desc.get("kind") != "slice":
        return None
    if idx_desc.get("lower") is not None:
        return None   # must start from 0

    k_s = idx_desc.get("upper")
    if k_s is None:
        return None
    if val_desc is None or not isinstance(val_desc.get("value"), (int, float)):
        return None

    c = float(val_desc["value"])
    N = ctx.N
    if N is None or k_s > N:
        return None

    return LoadPattern(VectorType.STEP, N=N, params={"k_s": int(k_s), "c": c})


def _try_square_load(ctx: _ExecutionContext) -> Optional[LoadPattern]:
    """
    Pattern: f[k1:k2] = c   (interior segment, k1 > 0)

    The prepared state is (1/sqrt(k2-k1)) * sum_{j=k1}^{k2-1} |j>.
    Circuit: difference of two step-load circuits via amplitude arithmetic.
    Gate count: O(m).
    """
    subscript_assigns = [a for a in ctx.assignments
                         if a["kind"] == "subscript_assign"
                         and a.get("base") == ctx.f_name]
    if len(subscript_assigns) != 1:
        return None

    idx_desc = subscript_assigns[0]["index"]
    val_desc  = subscript_assigns[0]["value"]

    if idx_desc.get("kind") != "slice":
        return None

    k1 = idx_desc.get("lower")
    k2 = idx_desc.get("upper")
    if k1 is None or k2 is None:
        return None   # must have explicit lower AND upper bound
    if k1 == 0:
        return None   # that is a step load, not a square load

    if val_desc is None or not isinstance(val_desc.get("value"), (int, float)):
        return None

    c  = float(val_desc["value"])
    N  = ctx.N
    if N is None or k2 > N or k1 >= k2:
        return None

    return LoadPattern(VectorType.SQUARE, N=N,
                       params={"k1": int(k1), "k2": int(k2), "c": c})


def _try_sinusoidal_load(ctx: _ExecutionContext) -> Optional[LoadPattern]:
    """
    Pattern: f = A * sin(2*pi*n*k/N + phi)  (k = 0..N-1)
    Matches code of the form::
        x = np.linspace(0, 1, N)   # or any linspace
        f = np.sin(2 * np.pi * n * x + phi)
    Extracts integer mode n >= 1, amplitude A, and phase phi (default 0.0).
    """
    name_assigns = [a for a in ctx.assignments
                    if a["kind"] == "name_assign"
                    and a["name"] == ctx.f_name]
    if not name_assigns:
        return None

    last_val = name_assigns[-1]["value"]
    if last_val is None:
        return None

    # A * sin(...) or sin(...)
    A = 1.0
    sin_desc = None
    if last_val.get("kind") == "sin_call":
        sin_desc = last_val
    elif last_val.get("kind") == "binop" and last_val.get("op") == "Mult":
        left, right = last_val.get("left"), last_val.get("right")
        if left and left.get("kind") == "sin_call":
            sin_desc = left
            A = float(right["value"]) if right and "value" in right else None
        elif right and right.get("kind") == "sin_call":
            sin_desc = right
            A = float(left["value"]) if left and "value" in left else None

    if sin_desc is None or A is None:
        return None

    # Try to extract mode n and phase phi from the sin argument
    result = _extract_sin_mode_and_phase(sin_desc.get("arg"), ctx)
    if result is None:
        return None
    n, phi = result

    N = ctx.N
    if N is None:
        return None

    return LoadPattern(VectorType.SINE, N=N,
                       params={"n": n, "A": A, "phi": phi})


def _extract_sin_mode_and_phase(arg_desc, ctx: _ExecutionContext) -> Optional[tuple]:
    """
    Extract (n, phi) from sin argument  n*pi*x/L + phi  or  n*pi*x/L.

    The argument may be:
      - A pure product tree  n * pi * x / L          -> (n, 0.0)
      - An Add/Sub binop     n * pi * x / L ± phi     -> (n, ±phi)

    phi is a constant (float, possibly involving pi, e.g. np.pi/4).
    Returns None if the frequency part cannot be recognized.
    """
    if arg_desc is None:
        return None

    # Check for an Add/Sub at the top level: freq_part ± phase_part
    phi = 0.0
    freq_desc = arg_desc
    if arg_desc.get("kind") == "binop" and arg_desc.get("op") in ("Add", "Sub"):
        left  = arg_desc.get("left")
        right = arg_desc.get("right")
        # The phase term is whichever side is a plain number (no x variable)
        left_is_num  = (left  is not None and left.get("kind")  == "number")
        right_is_num = (right is not None and right.get("kind") == "number")
        if right_is_num:
            freq_desc = left
            phi = float(right["value"])
            if arg_desc["op"] == "Sub":
                phi = -phi
        elif left_is_num:
            freq_desc = right
            phi = float(left["value"])
            # left ± freq: only sensible as Add (phi + freq_part)
        # else: neither side is a plain constant — fall through with phi=0

    n = _extract_sin_mode(freq_desc, ctx)
    if n is None:
        return None
    return (n, phi)


def _extract_sin_mode(arg_desc, ctx: _ExecutionContext) -> Optional[int]:
    """
    Attempt to extract integer mode n from  n * pi * x / L  style expression.
    Handles both:
      - Symbolic form:   3 * np.pi * x / L  (pi and 3 appear as separate factors)
      - Folded form:     9.4248... * x / L  (3*pi already constant-folded)
    """
    if arg_desc is None:
        return None

    # Flatten the expression tree into a list of numeric factors
    factors = _collect_product_factors(arg_desc)
    if not factors:
        return None

    pi_found = False
    n = None

    for f in factors:
        if isinstance(f, (int, float)):
            fv = float(f)
            # Exact pi
            if abs(fv - math.pi) < 1e-9:
                pi_found = True
            # n * pi  (already folded): check if fv / pi is a small integer
            elif fv > 0.5:
                ratio = fv / math.pi
                if abs(ratio - round(ratio)) < 1e-6 and 1 <= round(ratio) <= 64:
                    n_candidate = int(round(ratio))
                    if n_candidate > 1 or pi_found:
                        if n is None or n_candidate != 1:
                            n = n_candidate
                            pi_found = True   # pi is implicit
            # Plain small integer (not pi): candidate mode
            elif abs(fv - round(fv)) < 1e-9 and 1 <= int(round(fv)) <= 64:
                if n is None:
                    n = int(round(fv))

    if pi_found and n is not None:
        return n
    if pi_found and n is None:
        return 1   # mode 1: only pi appears, n implicit

    return None


def _collect_product_factors(desc, acc=None):
    """Recursively collect numeric factors from a nested binop multiply tree.
    Non-numeric leaves (variable names, linspace refs) are silently skipped."""
    if acc is None:
        acc = []
    if desc is None:
        return acc
    if desc.get("kind") == "number":
        acc.append(desc["value"])
        return acc
    if desc.get("kind") in ("name", "name_index", "linspace", "call"):
        # Variable reference (e.g. x, L) — skip, not a numeric factor
        return acc
    if desc.get("kind") == "binop":
        op = desc.get("op")
        if op in ("Mult", "Div"):
            _collect_product_factors(desc.get("left"), acc)
            _collect_product_factors(desc.get("right"), acc)
            return acc
    return acc



def _try_cosine_load(ctx: _ExecutionContext) -> Optional[LoadPattern]:
    """
    Pattern: f = A * cos(n * pi * x + phi)  (x = linspace(0, 1, N))
    Mirrors _try_sinusoidal_load but detects cos_call.
    The circuit encodes cos(2*pi*n*k/N + phi) via QFT with P(-2*phi) gate.
    """
    name_assigns = [a for a in ctx.assignments
                    if a["kind"] == "name_assign"
                    and a["name"] == ctx.f_name]
    if not name_assigns:
        return None

    last_val = name_assigns[-1]["value"]
    if last_val is None:
        return None

    A = 1.0
    cos_desc = None
    if last_val.get("kind") == "cos_call":
        cos_desc = last_val
    elif last_val.get("kind") == "binop" and last_val.get("op") == "Mult":
        left, right = last_val.get("left"), last_val.get("right")
        if left and left.get("kind") == "cos_call":
            cos_desc = left
            A = float(right["value"]) if right and "value" in right else None
        elif right and right.get("kind") == "cos_call":
            cos_desc = right
            A = float(left["value"]) if left and "value" in left else None

    if cos_desc is None or A is None:
        return None

    result = _extract_sin_mode_and_phase(cos_desc.get("arg"), ctx)
    if result is None:
        return None
    n, phi = result

    N = ctx.N
    if N is None:
        return None

    return LoadPattern(VectorType.COSINE, N=N,
                       params={"n": n, "A": A, "phi": phi})

def _try_multi_point_load(ctx: _ExecutionContext) -> Optional[LoadPattern]:
    """
    Multi-point load: L >= 2 assignments f[k_i] = P_i with distinct indices.

    Weights P_i may be arbitrary (need not be equal).  The synthesizer
    uses a binary-tree Ry decomposition giving gate count O(m * L),
    strictly cheaper than Shende's O(2^m) for sparse L << 2^m.
    """
    subscript_assigns = [a for a in ctx.assignments
                         if a["kind"] == "subscript_assign"
                         and a.get("base") == ctx.f_name
                         and a["index"].get("kind") == "index"
                         and a["value"] is not None
                         and isinstance(a["value"].get("value"), (int, float))]

    if len(subscript_assigns) < 2:
        return None

    indices = [a["index"]["value"] for a in subscript_assigns]
    if len(set(indices)) != len(indices):
        return None  # repeated index — not disjoint

    N = ctx.N
    if N is None or any(k >= N for k in indices):
        return None

    loads = [{"k": int(a["index"]["value"]), "P": float(a["value"]["value"])}
             for a in subscript_assigns]

    return LoadPattern(VectorType.MULTI_DISCRETE, N=N, params={"loads": loads})


def _try_multi_sin_load(ctx: _ExecutionContext) -> Optional[LoadPattern]:
    """
    Case B: sum of sinusoidal modes.
    Detected when f is assigned a sum of sin-containing expressions.
    Only handles up to 4 modes for circuit efficiency.
    """
    name_assigns = [a for a in ctx.assignments
                    if a["kind"] == "name_assign"
                    and a["name"] == ctx.f_name]
    if not name_assigns:
        return None

    last_val = name_assigns[-1]["value"]
    if last_val is None:
        return None

    modes = _collect_sin_sum(last_val, ctx)
    if modes is None or len(modes) < 2:
        return None

    N = ctx.N
    if N is None:
        return None

    return LoadPattern(VectorType.MULTI_SINE, N=N, params={"modes": modes})


def _collect_sin_sum(desc, ctx: _ExecutionContext) -> Optional[list]:
    """Recursively collect (amplitude, mode) pairs from a sum of sin terms."""
    if desc is None:
        return None

    if desc.get("kind") == "sin_call":
        n = _extract_sin_mode(desc.get("arg"), ctx)
        return [{"A": 1.0, "n": n}] if n else None

    if desc.get("kind") == "binop" and desc.get("op") == "Mult":
        left, right = desc.get("left"), desc.get("right")
        if left and left.get("kind") == "sin_call":
            n = _extract_sin_mode(left.get("arg"), ctx)
            A = right.get("value") if right else 1.0
            return [{"A": float(A) if A else 1.0, "n": n}] if n else None
        if right and right.get("kind") == "sin_call":
            n = _extract_sin_mode(right.get("arg"), ctx)
            A = left.get("value") if left else 1.0
            return [{"A": float(A) if A else 1.0, "n": n}] if n else None

    if desc.get("kind") == "binop" and desc.get("op") == "Add":
        left_modes  = _collect_sin_sum(desc.get("left"), ctx)
        right_modes = _collect_sin_sum(desc.get("right"), ctx)
        if left_modes is not None and right_modes is not None:
            return left_modes + right_modes

    return None


def _try_uniform_spike_load(ctx: _ExecutionContext) -> Optional[LoadPattern]:
    """
    Case C: uniform load + single point perturbation.
    Pattern: f = ones(N) * c; f[k] = delta
    """
    subscript_assigns = [a for a in ctx.assignments
                         if a["kind"] == "subscript_assign"
                         and a.get("base") == ctx.f_name]
    if len(subscript_assigns) != 1:
        return None

    idx_desc = subscript_assigns[0]["index"]
    val_desc = subscript_assigns[0]["value"]
    if idx_desc.get("kind") != "index":
        return None
    if val_desc is None or not isinstance(val_desc.get("value"), (int, float)):
        return None

    # Check there was also a ones-based initialisation
    name_assigns = [a for a in ctx.assignments
                    if a["kind"] == "name_assign"
                    and a["name"] == ctx.f_name]
    if not name_assigns:
        return None

    base_val = name_assigns[-1]["value"]
    if base_val is None:
        return None

    c = None
    if base_val.get("kind") == "ones":
        c = 1.0
    elif (base_val.get("kind") == "binop" and base_val.get("op") == "Mult"):
        left, right = base_val.get("left"), base_val.get("right")
        if left and left.get("kind") == "ones":
            c = float(right["value"]) if right and "value" in right else None
        elif right and right.get("kind") == "ones":
            c = float(left["value"]) if left and "value" in left else None

    if c is None:
        return None

    N   = ctx.N
    k   = int(idx_desc["value"])
    delta = float(val_desc["value"])
    if N is None or k >= N:
        return None

    return LoadPattern(VectorType.UNIFORM_SPIKE, N=N,
                       params={"c": c, "k": k, "delta": delta})