"""Unit tests for cascade-host shim data plumbing.

Covers smsae.sae.data: encode_state_as_ids, encode_state_as_targets,
cascade_transitions.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def test_encode_state_round_trip():
    """A state encoded to input_ids should reconstruct the multiset."""
    from smsae.sae.data import (
        PAD_TOKEN_ID, _particle_to_id, encode_state_as_ids,
    )
    name_to_id = _particle_to_id()
    id_to_name = {i: n for n, i in name_to_id.items()}

    state = {"e-": 2, "nu_e": 1, "photon": 3}
    ids = encode_state_as_ids(state, max_seq=32)
    assert ids.dtype == np.int64
    assert ids.shape == (32,)

    # Non-PAD positions reconstruct the multiset
    tokens = [int(t) for t in ids if t != PAD_TOKEN_ID]
    multiset = Counter(id_to_name[t] for t in tokens)
    assert dict(multiset) == state

    # PAD positions are at the tail
    pad_start = sum(state.values())
    assert (ids[pad_start:] == PAD_TOKEN_ID).all()
    assert (ids[:pad_start] != PAD_TOKEN_ID).all()


def test_encode_state_canonical_order():
    """Encoding is alphabetical so the model doesn't have to learn
    permutation invariance."""
    from smsae.sae.data import _particle_to_id, encode_state_as_ids
    name_to_id = _particle_to_id()
    state = {"Z": 1, "H": 1, "W+": 1}
    ids = encode_state_as_ids(state, max_seq=8)
    head = [int(t) for t in ids[:3]]
    # alphabetical: H, W+, Z (by Python sort)
    assert head == [name_to_id["H"], name_to_id["W+"], name_to_id["Z"]]


def test_encode_state_truncation():
    """When a state overflows max_seq, the right tail is dropped."""
    from smsae.sae.data import PAD_TOKEN_ID, encode_state_as_ids
    state = {"photon": 50}  # 50 > max_seq=8
    ids = encode_state_as_ids(state, max_seq=8)
    assert (ids != PAD_TOKEN_ID).all()  # no PAD; all 8 slots used
    assert len(set(ids.tolist())) == 1   # all the same particle


def test_encode_targets_ignores_pad_positions():
    """Loss should skip PAD positions in the input."""
    from smsae.sae.data import (
        PAD_TOKEN_ID, _particle_to_id, encode_state_as_ids,
        encode_state_as_targets,
    )
    name_to_id = _particle_to_id()
    input_state = {"H": 1}
    next_state = {"e-": 2}  # next state has more tokens than input
    input_ids = encode_state_as_ids(input_state, max_seq=8)
    targets = encode_state_as_targets(next_state, input_ids, max_seq=8)
    # Position 0 (input=H, non-PAD) → target is next-state token 0 (e-)
    assert targets[0] == name_to_id["e-"]
    # Positions 1..7 (input=PAD) → target is ignore_index
    assert (targets[1:] == -100).all()


def test_cascade_transitions_yields_pairs():
    """Smoke test: a tiny rollout yields at least one (input, target) pair."""
    from smsae.sae.data import cascade_transitions
    pairs = list(cascade_transitions(n_trajectories=5, seed=0, max_seq=16))
    assert len(pairs) >= 1
    for input_ids, target_ids in pairs:
        assert input_ids.shape == (16,)
        assert target_ids.shape == (16,)
        assert input_ids.dtype == np.int64
        assert target_ids.dtype == np.int64


def test_cascade_transitions_is_deterministic():
    """Same seed → same pair stream."""
    from smsae.sae.data import cascade_transitions
    a = list(cascade_transitions(n_trajectories=3, seed=42, max_seq=16))
    b = list(cascade_transitions(n_trajectories=3, seed=42, max_seq=16))
    assert len(a) == len(b)
    for (ai, at), (bi, bt) in zip(a, b):
        assert (ai == bi).all()
        assert (at == bt).all()
