# setup.py
from setuptools import setup, find_packages

setup(
    name="cvaei",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "seaborn",
        "hyperopt",
        "gillespy2",
    ],
)
