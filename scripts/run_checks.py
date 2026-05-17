"""Run all SM-bundle consistency checks.

Includes:
  - Gell-Mann-Nishijima  Q = T_3 + Y/2
  - Anomaly cancellation per generation
  - 168-vertex catalog closure
  - Kinematic thresholds
  - Missing-particle inference
  - Conservation algebra recovery from signed incidence

Usage:
    python scripts/run_checks.py
"""

from __future__ import annotations

import os
import runpy
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)


if __name__ == "__main__":
    # Run the module's __main__ guard
    runpy.run_module("smsae.sm.checks", run_name="__main__")
