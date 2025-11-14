"""
Setup configuration for MusicHal 9000 package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="musichal",
    version="2.0.0",
    author="MusicHal Team",
    description="AI Co-Improviser for Live Musical Performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/musichal",  # Update with actual URL
    packages=find_packages(include=["musichal", "musichal.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.13",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "musichal-train=scripts.train.train_modular:main",
            "musichal-perform=scripts.performance.MusicHal_9000:main",
        ],
    },
    include_package_data=True,
    package_data={
        "musichal": ["py.typed"],
    },
    zip_safe=False,
)
