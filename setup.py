from setuptools import setup, find_packages
setup(
    name="pyencode",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["qiskit>=2.3.0", "numpy>=2.0.0"],
)