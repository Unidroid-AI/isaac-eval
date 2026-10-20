#!/usr/bin/env python3
# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Repo-root launcher. ``isaac-eval`` and this script run the same entry point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaac_eval.scripts.evaluate import main

if __name__ == "__main__":
    main()
