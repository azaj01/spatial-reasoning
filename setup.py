"""Setup configuration for spatial_reasoning package."""

import os

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="spatial-reasoning",
    version="0.1.8",
    author="Qasim Wani",
    author_email="qasim31wani@gmail.com",
    description="A PyPI package for object detection using advanced vision models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QasimWani/spatial-reasoning",
    project_urls={
        "Bug Tracker": "https://github.com/QasimWani/spatial-reasoning/issues",
        "Documentation": "https://github.com/QasimWani/spatial-reasoning#readme",
        "Source Code": "https://github.com/QasimWani/spatial-reasoning",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.990",
        ],
    },
    entry_points={
        "console_scripts": [
            "spatial-reasoning=spatial_reasoning.run_cli:main",
        ],
    },
    include_package_data=True,
    keywords="computer vision, object detection, AI, machine learning, OpenAI, Gemini",
)
