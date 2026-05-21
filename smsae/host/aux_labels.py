"""Per-state binary aux labels for the cascade-host trainer.

Five v1 labels covering three structural classes:

- Conservation: ``total_charge_neutral``, ``total_baryon_neutral``
- Lineage:     ``originated_from_top``
- Existence:   ``state_has_higgs``, ``state_has_top``

The label vocabulary is consumed solely by the optional aux-supervision
head added by ``aux-supervise-cascade-host``. It is **not** part of the
sm-sae benchmark's GT grading vocabulary.

Module is pure-Python (no torch import). The Standard Model dict from
``smsae.sm.embeddings.build_sm`` is cached at module scope so the
per-label charge/baryon lookups don't pay rebuild cost per call.

See ``openspec/changes/aux-supervise-cascade-host/specs/cascade-host-aux-labels/spec.md``
for the load-bearing contract.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


# Top quark color triplet. `originated_from_top` and `state_has_top`
# both branch on membership in this set.
_TOP_QUARKS: frozenset[str] = frozenset({"t_r", "t_g", "t_b"})

# Numerical tolerance for the conservation checks. The simulator emits
# integer-count states so exact arithmetic over float charges should
# stay well under this — anything bigger than 1e-9 is a real violation.
_CONSERVATION_TOLERANCE: float = 1e-9


# Label order is load-bearing: callers (the trainer's aux head, the
# `compute_aux_labels` output, the `config.json` `aux_labels` field) all
# rely on this ordering. Do not reorder without bumping the spec.
_AUX_LABEL_NAMES: tuple[str, ...] = (
    "total_charge_neutral",
    "total_baryon_neutral",
    "originated_from_top",
    "state_has_higgs",
    "state_has_top",
)


def aux_label_names() -> list[str]:
    """Return the v1 aux-label vocabulary in stable order.

    Output is a fresh list each call (so callers can't accidentally
    mutate the canonical tuple), but the contents are frozen.
    """
    return list(_AUX_LABEL_NAMES)


@lru_cache(maxsize=1)
def _sm_charge_baryon() -> dict[str, tuple[float, float]]:
    """Cached per-particle (charge, baryon) lookup table built from
    ``build_sm()``. Imported lazily so the module's top-level stays
    torch-free even though ``build_sm`` lives in the heavier
    ``smsae.sm`` package."""
    from smsae.sm.embeddings import build_sm

    sm = build_sm()
    # `vec` layout: [Q, B, Le, Lmu, Ltau, c3, c8, spin, mass]. Index 0 is
    # electric charge; index 1 is baryon number.
    return {name: (float(p.vec[0]), float(p.vec[1])) for name, p in sm.items()}


def compute_aux_labels(
    state: dict[str, int],
    initial_parent: str | None,
) -> np.ndarray:
    """Compute the 5 aux labels for a cascade state + its rollout origin.

    Args:
        state: ``dict[particle_name, count]`` — the cascade state.
            Counts SHALL be non-negative integers; the function does
            not validate but assumes simulator-produced states.
        initial_parent: the particle name that started the rollout
            this state was reached from. ``None`` is accepted and
            sets ``originated_from_top`` to 0 (the lineage signal is
            unavailable without it).

    Returns:
        A ``(5,)``-shaped ``float32`` array of 0/1 values matching the
        column order of :func:`aux_label_names`.
    """
    charge_baryon = _sm_charge_baryon()

    total_charge = 0.0
    total_baryon = 0.0
    for particle, count in state.items():
        q, b = charge_baryon.get(particle, (0.0, 0.0))
        total_charge += q * count
        total_baryon += b * count

    total_charge_neutral = 1.0 if abs(total_charge) <= _CONSERVATION_TOLERANCE else 0.0
    total_baryon_neutral = 1.0 if abs(total_baryon) <= _CONSERVATION_TOLERANCE else 0.0

    originated_from_top = (
        1.0 if (initial_parent is not None and initial_parent in _TOP_QUARKS) else 0.0
    )

    state_has_higgs = 1.0 if state.get("H", 0) > 0 else 0.0
    state_has_top = 1.0 if any(state.get(t, 0) > 0 for t in _TOP_QUARKS) else 0.0

    return np.array(
        [
            total_charge_neutral,
            total_baryon_neutral,
            originated_from_top,
            state_has_higgs,
            state_has_top,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# v2 aux vocabulary: 110-feature GT labels
# ---------------------------------------------------------------------------
# `richer-cascade-host-supervision-v2` extends the v1 5-label vocabulary to
# the full sm-sae GT vocabulary surfaced by
# `smsae.sae.data.all_ground_truth_features()` — ~110 binary labels per
# state, one per (particle | kind | flavor | color | is_* | chirality |
# generation | charge_sign) value. The probe diagnosis showed the host
# under-trains per-particle / per-color / per-flavor identity at the
# medium-strength level (0.74-0.85 AUC); explicit supervision against
# this vocabulary is the falsifiable lever.


@lru_cache(maxsize=1)
def _gt_aux_label_names() -> tuple[str, ...]:
    """Cached, frozen tuple of the GT aux vocabulary (deferred import to
    keep this module torch-free at the top level)."""
    from smsae.sae.data import all_ground_truth_features

    return tuple(all_ground_truth_features())


def gt_aux_label_names() -> list[str]:
    """Return the v2 aux-label vocabulary (full sm-sae GT vocabulary) in
    stable order. Fresh list each call (mutation-safe).
    """
    return list(_gt_aux_label_names())


@lru_cache(maxsize=1)
def _particle_feature_lookup() -> dict[str, frozenset[str]]:
    """Cached `particle_name -> feature-set` map used by
    :func:`compute_gt_aux_labels` to avoid recomputing per-state."""
    from smsae.sae.data import particle_features
    from smsae.sm.embeddings import build_sm

    sm = build_sm()
    return {name: frozenset(particle_features(name)) for name in sm}


def compute_gt_aux_labels(state: dict[str, int]) -> np.ndarray:
    """Compute the 110-feature GT label vector for a cascade state.

    Per-state binary label: bit-i is 1 iff ANY particle in ``state``
    carries GT feature i (the OR over per-particle feature sets).
    Matches the labelling convention used by
    ``scripts/probe_full_gt_recoverability.py``.

    Args:
        state: ``dict[particle_name, count]`` — the cascade state.
            Counts SHALL be non-negative integers; the function does not
            validate but assumes simulator-produced states.

    Returns:
        A ``(len(gt_aux_label_names()),)``-shaped ``float32`` array of
        0/1 values matching the column order of
        :func:`gt_aux_label_names`.
    """
    names = _gt_aux_label_names()
    lookup = _particle_feature_lookup()
    name_to_idx = {n: i for i, n in enumerate(names)}

    row = np.zeros(len(names), dtype=np.float32)
    for particle, count in state.items():
        if count <= 0:
            continue
        for feat in lookup.get(particle, ()):  # empty set for unknown particles
            idx = name_to_idx.get(feat)
            if idx is not None:
                row[idx] = 1.0
    return row
