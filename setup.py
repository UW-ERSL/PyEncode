from setuptools import setup, find_packages
setup(
    name="pyencode",
    version="3.0.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=["qiskit>=2.3.0", "numpy>=2.0.0"],
)