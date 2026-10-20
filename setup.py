# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Setup script for isaac-eval package."""

from setuptools import find_packages, setup

setup(
    name="isaac-eval",
    version="0.1.0",
    description="Isaac Lab evaluation framework for Unidroid",
    author="Unidroid",
    license="BSD-3-Clause",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.20.0",
        "pyyaml>=6.0",
        "torch>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "isaac-eval=isaac_eval.scripts.evaluate:main",
        ],
    },
)

