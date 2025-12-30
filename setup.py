"""Setup script for HealthML-Hub."""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="healthml-hub",
    version="1.0.0",
    description="A production-grade medical machine learning repository",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="HealthML-Hub Contributors",
    author_email="",
    url="https://github.com/yourusername/HealthML-Hub",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="medical machine learning healthcare disease prediction",
    project_urls={
        "Documentation": "https://github.com/yourusername/HealthML-Hub/tree/main/docs",
        "Source": "https://github.com/yourusername/HealthML-Hub",
        "Tracker": "https://github.com/yourusername/HealthML-Hub/issues",
    },
    include_package_data=True,
    zip_safe=False,
)

