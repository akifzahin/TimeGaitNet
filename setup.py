from setuptools import setup, find_packages

setup(
    name="timegaitnet",
    version="0.1.0",
    description="TimeGaitNet: BiMamba-based Multitask Learning for Freezing of Gait Detection",
    author="Akif",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyarrow>=12.0.0",
        "mamba-ssm>=1.0.0",
    ],
    python_requires=">=3.8",
)
