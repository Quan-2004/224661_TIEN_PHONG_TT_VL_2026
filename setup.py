"""
setup.py – Cài đặt thư viện mocsvm như một package Python
=============================================================
Cho phép cài bằng: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name         = "mocsvm",
    version      = "1.0.0",
    description  = "Modified One-Class Incremental SVM (mOC-iSVM)",
    author       = "mOC-iSVM Team",
    packages     = find_packages(include=["mocsvm", "mocsvm.*"]),
    python_requires = ">=3.10",
    install_requires = [
        "scikit-learn>=1.5.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "joblib>=1.4.0",
    ],
)
