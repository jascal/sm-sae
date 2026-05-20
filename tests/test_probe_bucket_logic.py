"""Unit tests for the bucket A/B/C/ambiguous heuristic in
`scripts/probe_full_gt_recoverability.py`. The live probe's bucket is
the empirical finding; these tests verify the routing logic itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probe_full_gt_recoverability import _bucket_summary


def _aucs(values: list[float | None]) -> dict[str, float | None]:
    """Helper: build a {feature_name: auc} dict from a flat list of AUCs."""
    return {f"f{i}": v for i, v in enumerate(values)}


def test_bucket_A_host_and_sae_both_carry_signal():
    """≥80% residual ≥0.9 AND ≥80% projected ≥0.9 → bucket A."""
    residual = _aucs([0.95] * 90 + [0.6] * 10)
    projected = _aucs([0.95] * 90 + [0.6] * 10)
    summary = _bucket_summary(residual, projected)
    assert summary["bucket"] == "A"


def test_bucket_B_host_carries_but_sae_drops():
    """≥80% residual ≥0.9 but <50% projected ≥0.9 → bucket B."""
    residual = _aucs([0.95] * 90 + [0.6] * 10)
    projected = _aucs([0.95] * 40 + [0.6] * 60)
    summary = _bucket_summary(residual, projected)
    assert summary["bucket"] == "B"


def test_bucket_C_host_lacks_signal():
    """≥50% residual <0.7 → bucket C, regardless of projected."""
    residual = _aucs([0.6] * 60 + [0.95] * 40)
    summary = _bucket_summary(residual, None)
    assert summary["bucket"] == "C"


def test_bucket_ambiguous_neither_threshold():
    """Mid-range residual; neither C threshold (<0.7) nor A/B threshold
    (≥0.9 high frac) met → ambiguous."""
    residual = _aucs([0.85] * 100)
    summary = _bucket_summary(residual, None)
    assert summary["bucket"] == "ambiguous"


def test_bucket_ambiguous_need_projected_when_high_residual_no_flag():
    """≥80% residual ≥0.9 but `--from-projected` not set → caller can't
    distinguish A from B. Surface that explicitly."""
    residual = _aucs([0.95] * 90 + [0.6] * 10)
    summary = _bucket_summary(residual, None)
    assert summary["bucket"] == "ambiguous_need_projected"


def test_bucket_ambiguous_borderline_projected():
    """Residual high but projected lands in the 50%-80% middle → flagged
    as borderline A/B."""
    residual = _aucs([0.95] * 90 + [0.6] * 10)
    projected = _aucs([0.95] * 60 + [0.6] * 40)  # 60% projected high
    summary = _bucket_summary(residual, projected)
    assert summary["bucket"] == "ambiguous_borderline_projected"


def test_none_aucs_excluded_from_fractions():
    """Degenerate (None) labels SHALL be excluded from the bucket
    fraction denominator; otherwise a few NaNs could move the bucket."""
    residual = {**_aucs([0.95] * 90 + [0.6] * 10), "extra1": None, "extra2": None}
    summary = _bucket_summary(residual, None)
    assert summary["n_features_measured_residual"] == 100
    # NaNs excluded — 90/100 = 0.9 ≥ 0.8 still A territory
    assert summary["bucket"] == "ambiguous_need_projected"  # because no --from-projected


def test_summary_includes_interpretation_string():
    residual = _aucs([0.95] * 90 + [0.6] * 10)
    projected = _aucs([0.95] * 90 + [0.6] * 10)
    summary = _bucket_summary(residual, projected)
    assert "interpretation" in summary
    assert "both carry signal" in summary["interpretation"]


def test_empty_aucs_does_not_crash():
    """Defensive: an empty AUC dict (e.g., no GT features measured)
    shouldn't crash; bucket falls into the ambiguous default."""
    summary = _bucket_summary({}, None)
    assert summary["bucket"] in {"ambiguous", "ambiguous_need_projected"}
    assert summary["n_features_measured_residual"] == 0
