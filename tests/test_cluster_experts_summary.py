"""Unit test for the consolidated validation-summary builder in
``scripts/cluster_experts_demo.py``. Pure dict->dict; trains nothing and
touches no polygram — it only verifies the headline/verdict logic that
produces ``runs/cluster_experts/summary.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cluster_experts_demo import build_validation_summary


def _run(thr, multi80, multi70, k=20, multi=10, gt95=15, mu=0.9):
    return {
        "coherence_threshold": thr, "n_experts": k, "n_multi_member": multi,
        "n_multi_meta_at_0.80": multi80, "n_multi_meta_at_0.70": multi70,
        "n_gt_covered_at_0.95": gt95, "cluster_mean_auc_mean": mu,
    }


def test_picks_best_run_by_multi_meta_and_passes():
    """Best run per cell is the one maximising multi-meta>=.80 then >=.70;
    a cell passes iff >=1 multi-member cluster recovers a META label at .80."""
    per_cell = {
        "cascade__jumprelu": {"encoding": "rung5", "runs": [
            _run(0.3, 2, 10), _run(0.5, 0, 0),     # 0.3 wins (2 vs 0)
        ]},
        "embedded__topk": {"encoding": "rung5", "runs": [
            _run(0.4, 3, 3), _run(0.5, 12, 12),    # 0.5 wins (12 vs 3)
        ]},
    }
    s = build_validation_summary(per_cell, "0.12.0")
    assert s["polygram_version"] == "0.12.0"
    assert s["n_cells"] == 2
    assert s["validation_passes"] is True
    casc = next(c for c in s["cells"] if c["run_id"] == "cascade__jumprelu")
    assert casc["best_threshold"] == 0.3
    assert casc["n_multi_meta_ge_0.80"] == 2
    assert casc["validation_passes"] is True
    emb = next(c for c in s["cells"] if c["run_id"] == "embedded__topk")
    assert emb["best_threshold"] == 0.5
    assert emb["n_multi_meta_ge_0.80"] == 12


def test_cell_fails_when_no_multi_meta_at_080():
    per_cell = {"x": {"encoding": "rung5", "runs": [
        _run(0.3, 0, 5), _run(0.5, 0, 0),
    ]}}
    s = build_validation_summary(per_cell, "0.12.0")
    assert s["validation_passes"] is False
    assert s["cells"][0]["validation_passes"] is False


def test_empty_is_not_a_pass():
    s = build_validation_summary({}, "0.12.0")
    assert s["validation_passes"] is False
    assert s["n_cells"] == 0
    assert s["cells"] == []
