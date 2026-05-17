"""Phase B: alignment-score all trained SAEs against ground-truth features.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from smsae.sae.evaluation import main

if __name__ == "__main__":
    main()
