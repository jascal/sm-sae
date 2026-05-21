"""Unit tests for ``smsae.host.aux_labels``.

Covers the contract from
``openspec/changes/archive/aux-supervise-cascade-host/specs/cascade-host-aux-labels/spec.md``:

- Stable label order
- Shape + dtype + 0/1 binarity
- Conservation labels match hand-computed expectations
- Lineage `originated_from_top` requires `initial_parent`
- Existence labels read directly from the state dict
- Module has no torch dependency
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from smsae.host.aux_labels import aux_label_names, compute_aux_labels


# ---------------------------------------------------------------------------
# Label vocabulary + order
# ---------------------------------------------------------------------------


def test_aux_label_names_returns_v1_vocabulary():
    assert aux_label_names() == [
        "total_charge_neutral",
        "total_baryon_neutral",
        "originated_from_top",
        "state_has_higgs",
        "state_has_top",
    ]


def test_aux_label_names_is_stable_across_calls():
    a = aux_label_names()
    b = aux_label_names()
    assert a == b


def test_aux_label_names_caller_mutation_does_not_leak():
    """Caller-side mutation of the returned list must not affect later calls."""
    a = aux_label_names()
    a.clear()
    assert aux_label_names() == [
        "total_charge_neutral",
        "total_baryon_neutral",
        "originated_from_top",
        "state_has_higgs",
        "state_has_top",
    ]


# ---------------------------------------------------------------------------
# Shape / dtype / binarity
# ---------------------------------------------------------------------------


def test_returns_5_float32_array():
    arr = compute_aux_labels({"e-": 1}, initial_parent=None)
    assert arr.shape == (5,)
    assert arr.dtype == np.float32


def test_values_are_strict_0_or_1():
    arr = compute_aux_labels({"e-": 1, "e+": 1, "H": 1, "t_r": 1}, "t_r")
    for v in arr:
        assert float(v) in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Conservation labels
# ---------------------------------------------------------------------------


def test_charge_neutral_on_particle_antiparticle_pair():
    arr = compute_aux_labels({"e-": 1, "e+": 1}, None)
    assert arr[0] == 1.0  # total_charge_neutral
    assert arr[1] == 1.0  # total_baryon_neutral (leptons carry no baryon number)


def test_charge_neutral_zero_on_lone_charge():
    arr = compute_aux_labels({"e-": 1}, None)
    assert arr[0] == 0.0  # not charge-neutral
    assert arr[1] == 1.0  # still baryon-neutral (lepton)


def test_baryon_neutral_on_quark_antiquark_pair():
    arr = compute_aux_labels({"u_r": 1, "~u_r": 1}, None)
    assert arr[1] == 1.0  # baryon (+1/3) + (-1/3) = 0


def test_baryon_non_neutral_on_three_same_baryon():
    arr = compute_aux_labels({"u_r": 1, "d_g": 1, "s_b": 1}, None)
    # baryon-number = 1/3 + 1/3 + 1/3 = 1 → not neutral
    assert arr[1] == 0.0


def test_empty_state_both_conservation_labels_true():
    arr = compute_aux_labels({}, None)
    assert arr[0] == 1.0
    assert arr[1] == 1.0


# ---------------------------------------------------------------------------
# Lineage: originated_from_top
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parent", ["t_r", "t_g", "t_b"])
def test_top_lineage_fires_for_each_color(parent):
    arr = compute_aux_labels({"b_r": 1}, initial_parent=parent)
    assert arr[2] == 1.0


@pytest.mark.parametrize("parent", ["H", "Z", "W+", "u_r", "mu-"])
def test_top_lineage_does_not_fire_for_non_top(parent):
    arr = compute_aux_labels({"b_r": 1}, initial_parent=parent)
    assert arr[2] == 0.0


def test_top_lineage_requires_initial_parent():
    """The lineage label is a property of the rollout, not the current state.
    When `initial_parent` is None it must be 0.0 even if the state contains
    a top quark."""
    arr = compute_aux_labels({"t_r": 1}, initial_parent=None)
    assert arr[2] == 0.0
    # But state_has_top should still fire.
    assert arr[4] == 1.0


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------


def test_state_has_higgs():
    assert compute_aux_labels({"H": 1}, None)[3] == 1.0
    assert compute_aux_labels({"H": 1, "Z": 1}, None)[3] == 1.0
    assert compute_aux_labels({"Z": 1, "W+": 1}, None)[3] == 0.0


def test_state_has_top_for_any_color():
    for top in ("t_r", "t_g", "t_b"):
        arr = compute_aux_labels({top: 1}, None)
        assert arr[4] == 1.0, top
    assert compute_aux_labels({"u_r": 1}, None)[4] == 0.0


def test_state_has_higgs_zero_count_treated_as_absent():
    """A particle in the dict with count 0 means absent, not present."""
    arr = compute_aux_labels({"H": 0, "Z": 1}, None)
    assert arr[3] == 0.0


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_aux_labels_module_does_not_import_torch():
    """``compute_aux_labels`` must work in a torch-free environment.
    Asserts torch isn't transitively imported by the aux_labels module's
    top-level import (the build_sm lookup is cached lazily inside the
    function body)."""
    # Note: torch may already be in sys.modules from earlier test imports.
    # The contract is that the aux_labels module itself doesn't import
    # torch — check that by looking at the module's source.
    import smsae.host.aux_labels as aux_module

    src = open(aux_module.__file__).read()
    assert "import torch" not in src, "aux_labels.py should not import torch"
    assert "from torch" not in src, "aux_labels.py should not import torch"


# ---------------------------------------------------------------------------
# v2 GT-vocabulary helpers (richer-cascade-host-supervision-v2)
# ---------------------------------------------------------------------------

from smsae.host.aux_labels import compute_gt_aux_labels, gt_aux_label_names


def test_gt_aux_label_names_is_full_gt_vocab():
    """The v2 vocabulary matches sm-sae's GT vocabulary in length and
    order (so column indices match between this module and the probe)."""
    from smsae.sae.data import all_ground_truth_features
    expected = all_ground_truth_features()
    actual = gt_aux_label_names()
    assert actual == expected


def test_gt_aux_label_names_is_stable_across_calls():
    a = gt_aux_label_names()
    b = gt_aux_label_names()
    assert a == b
    # Each call returns a fresh list so callers can't mutate the canonical
    assert a is not b


def test_compute_gt_aux_labels_shape_and_dtype():
    names = gt_aux_label_names()
    state = {"t_r": 1, "b_g": 1, "W+": 1}
    labels = compute_gt_aux_labels(state)
    assert labels.shape == (len(names),)
    assert labels.dtype == np.float32
    # All values are 0 or 1
    assert set(np.unique(labels).tolist()) <= {0.0, 1.0}


def test_compute_gt_aux_labels_top_quark_state_hits_expected_features():
    """A state with t_r should light up `particle:t_r`, `flavor:t`,
    `color:r`, `is_colored`, `is_charged`, etc."""
    names = gt_aux_label_names()
    name_to_idx = {n: i for i, n in enumerate(names)}
    state = {"t_r": 1}
    labels = compute_gt_aux_labels(state)

    # Particle identity
    assert labels[name_to_idx["particle:t_r"]] == 1.0
    # Flavor (use the actual flavor key produced by particle_info)
    if "flavor:t" in name_to_idx:
        assert labels[name_to_idx["flavor:t"]] == 1.0
    # Color
    if "color:r" in name_to_idx:
        assert labels[name_to_idx["color:r"]] == 1.0
    # is_colored / is_charged
    if "is_colored" in name_to_idx:
        assert labels[name_to_idx["is_colored"]] == 1.0


def test_compute_gt_aux_labels_empty_state_is_all_zero():
    labels = compute_gt_aux_labels({})
    assert labels.sum() == 0.0


def test_compute_gt_aux_labels_zero_counts_ignored():
    """A particle with count=0 SHALL NOT contribute to the label vector."""
    names = gt_aux_label_names()
    name_to_idx = {n: i for i, n in enumerate(names)}
    labels = compute_gt_aux_labels({"t_r": 0})
    # No top-quark-related features should fire
    if "particle:t_r" in name_to_idx:
        assert labels[name_to_idx["particle:t_r"]] == 0.0


def test_compute_gt_aux_labels_multi_particle_or():
    """The label vector is the OR over per-particle feature sets — so a
    multi-particle state lights every feature that any constituent
    carries."""
    names = gt_aux_label_names()
    name_to_idx = {n: i for i, n in enumerate(names)}
    state = {"t_r": 1, "e-": 1}
    labels = compute_gt_aux_labels(state)
    # Both particle:t_r AND particle:e- should be 1
    assert labels[name_to_idx["particle:t_r"]] == 1.0
    assert labels[name_to_idx["particle:e-"]] == 1.0
