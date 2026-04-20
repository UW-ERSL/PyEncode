from pyencode import encode, GEOMETRIC
circuit, info = encode(GEOMETRIC(ratio=0.95, start=32), N=64)
print(f'✓ GEOMETRIC start parameter working: {info.gate_count_1q} gates, depth {info.circuit_depth}')