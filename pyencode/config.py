"""
pyencode.config
===============
Shared transpilation settings used by all PyEncode entry points
and by the gate-count comparison notebook.

All gate counts reported in the paper use these settings.
"""

# Qiskit version used for all experiments: 2.3.1
QISKIT_VERSION = "2.3.1"

# Universal gate basis for transpilation.
# All PyEncode and Qiskit gate counts in the paper are reported
# after transpilation to this basis.
BASIS_GATES = ['cx', 'u', 'x', 'h', 'ry', 'rz', 'rx', 'p']

# Optimization level used for all transpilation.
# Level 3 applies the full Qiskit optimization pass including
# commutation analysis, single-qubit gate consolidation, and
# two-qubit gate cancellation.
OPTIMIZATION_LEVEL = 3

# Number of decomposition repetitions used when decomposing
# Qiskit StatePreparation circuits before transpilation.
DECOMPOSE_REPS = 3
