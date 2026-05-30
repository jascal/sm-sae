"""Unit tests for the training-budget-sweep plumbing in
`scripts/cascade_host_capacity_sweep.py`: config normalisation (per-row
budget + host-dir disambiguation) and the gate-B.3 budget-trend summary.

The live sweep result is the empirical finding; these tests verify the
wiring that produces it — they do NOT train any host.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cascade_host_capacity_sweep import (
    BUDGET_GRID,
    _budget_trend,
    _normalize_configs,
)

_DEFAULTS = dict(default_n_trajectories=2000, default_epochs=5, default_lr=5e-3)


def test_tuple_configs_inherit_globals_and_legacy_dir_names():
    """Capacity/depth 2-tuples inherit the global budget and keep the
    legacy `sweep_NE{ne}_L{nl}` host dir (so re-runs reuse the hosts)."""
    norm = _normalize_configs([(61, 2), (96, 2)], **_DEFAULTS)
    assert norm[0]["dir_name"] == "sweep_NE61_L2"
    assert norm[1]["dir_name"] == "sweep_NE96_L2"
    for c in norm:
        assert c["n_trajectories"] == 2000
        assert c["epochs"] == 5
        assert c["lr"] == 5e-3


def test_budget_dicts_carry_per_row_budget_and_unique_dirs():
    """Budget dicts override the globals and get disambiguated host dirs
    so the four same-architecture rows don't collide."""
    norm = _normalize_configs(BUDGET_GRID, **_DEFAULTS)
    names = [c["dir_name"] for c in norm]
    # Row that matches the defaults keeps the legacy name (it IS the
    # depth-sweep L6 baseline); the rest append their budget.
    assert names[0] == "sweep_NE61_L6"
    assert names[1] == "sweep_NE61_L6_T2000_E20"
    assert names[2] == "sweep_NE61_L6_T2000_E40"
    assert names[3] == "sweep_NE61_L6_T5000_E20"
    assert len(set(names)) == len(names), "budget host dirs must be unique"
    # epochs override is threaded through
    assert [c["epochs"] for c in norm] == [5, 20, 40, 20]
    assert [c["n_trajectories"] for c in norm] == [2000, 2000, 2000, 5000]


def test_budget_dict_partial_override_inherits_missing_knobs():
    """A dict omitting a knob inherits the global default for it."""
    norm = _normalize_configs(
        [{"n_embd": 61, "n_layer": 6, "epochs": 20}], **_DEFAULTS
    )
    assert norm[0]["n_trajectories"] == 2000  # inherited
    assert norm[0]["epochs"] == 20            # overridden
    assert norm[0]["dir_name"] == "sweep_NE61_L6_T2000_E20"


def _row(traj, ep, steps, delta, color_r):
    return {
        "n_trajectories": traj, "epochs": ep, "n_train_steps": steps,
        "train_loss_final": 1.5, "forge_delta_vs_random": delta,
        "probe_color_r_auc": color_r, "probe_spotlight_median_auc": 0.88,
    }


def test_budget_trend_separates_axes_and_flags_passes():
    """gradient-step rows (smallest corpus) split from the corpus control;
    monotone color:r above the 0.82 midpoint passes B.3; any Δ≥0.05 passes
    B.2."""
    rows = [
        _row(2000, 5, 500, 0.024, 0.74),
        _row(2000, 20, 2000, 0.04, 0.80),
        _row(2000, 40, 4000, 0.06, 0.85),
        _row(5000, 20, 4960, 0.03, 0.83),
    ]
    bt = _budget_trend(rows)
    assert len(bt["gradient_step_axis"]) == 3
    assert len(bt["corpus_scaling_controls"]) == 1
    assert bt["color_r_best"] == 0.85
    assert bt["color_r_lift"] == 0.11
    assert bt["gradient_step_axis_monotonic"] is True
    assert bt["prediction_holds"] is True          # 0.85 ≥ 0.82 midpoint
    assert bt["gate_7_3_closed"] is True            # 0.06 ≥ 0.05


def test_budget_trend_fails_when_color_r_stays_flat():
    """Flat/dropping color:r fails B.3; sub-0.05 deltas fail B.2."""
    rows = [
        _row(2000, 5, 500, 0.01, 0.74),
        _row(2000, 20, 2000, 0.012, 0.745),
        _row(2000, 40, 4000, 0.011, 0.742),
    ]
    bt = _budget_trend(rows)
    assert bt["prediction_holds"] is False
    assert bt["gate_7_3_closed"] is False
    assert bt["best_forge_delta_vs_random"] == 0.012


def test_budget_trend_monotonic_but_subthreshold_lift_fails():
    """Re-anchoring: a host that starts high (depth already lifted it) and
    barely moves fails B.3 even if monotonic — the lift must clear the
    0.02 threshold over its own baseline. This is the smoke's L6 case
    (color:r starts ~0.877, not 0.74)."""
    rows = [
        _row(2000, 5, 500, 0.024, 0.877),
        _row(2000, 20, 2000, 0.02, 0.880),   # +0.003: monotonic but tiny
        _row(2000, 40, 4000, 0.018, 0.882),  # +0.005 total over baseline
    ]
    bt = _budget_trend(rows)
    assert bt["gradient_step_axis_monotonic"] is True
    assert bt["color_r_lift"] == 0.005
    assert bt["prediction_holds"] is False   # 0.005 < 0.02 threshold
    assert bt["color_r_l2_reference"] == 0.74
