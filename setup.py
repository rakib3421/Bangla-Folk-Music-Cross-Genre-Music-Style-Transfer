"""Setup configuration for Bangla Folk to Rock/Jazz Style Transfer System."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    """Read README.md file."""
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt."""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bangla-folk-style-transfer",
    version="1.0.0",
    author="Style Transfer Research Team",
    author_email="contact@example.com",
    description="A comprehensive neural audio style transfer system for converting Bangla Folk music to Rock and Jazz styles",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bangla-folk-style-transfer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "jupyter>=1.0",
            "notebook>=6.0",
        ],
        "gpu": [
            "torch-audio>=0.8.0",
            "torchaudio>=0.8.0",
        ],
        "visualization": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bangla-style-transfer=src.interactive.interactive_control:main",
            "train-style-model=src.training.training_pipeline:main",
            "evaluate-model=src.evaluation.style_transfer_evaluation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.txt", "*.md"],
    },
    zip_safe=False,
)