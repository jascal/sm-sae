"""Unit tests for scripts/forge_pipeline.py feature selectors.

Run with: pytest -q tests/test_selectors.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))


def _fixture():
    """Build a tiny TopK SAE and a synthetic feed that gives the SAE
    enough signal to fire on distinct features per sample.

    Returns (sae, feed, records).
    """
    from smsae.sae.data import Feed
    from smsae.sae.models import TopKSAE

    torch.manual_seed(0)
    D, F = 8, 6
    sae = TopKSAE(input_dim=D, n_features=F, k=2)

    # Pick decoder columns to be the first F coordinate-aligned axes so each
    # SAE feature is uniquely driven by one input dimension. Then feed in
    # one-hot vectors with different per-feature firing frequencies.
    with torch.no_grad():
        sae.W_dec.zero_()
        for j in range(F):
            sae.W_dec[j, j] = 1.0
        sae.W_enc.zero_()
        for j in range(F):
            sae.W_enc[j, j] = 1.0
        sae.b_enc.zero_()
        sae.b_dec.zero_()

    # Build a feed where feature j fires (j+1)*2 times, so firing-rate
    # ordering is deterministic and reverse to feature_id.
    rows = []
    feats = []
    vocab = [f"feat_{j}" for j in range(F)]
    for j in range(F):
        for _ in range((j + 1) * 2):
            x = torch.zeros(D)
            x[j] = 1.0
            rows.append(x)
            feats.append({vocab[j]})
    X = torch.stack(rows, dim=0)
    feed = Feed(
        name="test", X=X,
        sample_names=[f"s{i}" for i in range(len(rows))],
        sample_features=feats,
        feature_vocab=vocab,
    )

    # Records: a stand-in dict — the selectors only need .keys().
    records = {j: object() for j in range(F)}
    return sae, feed, records


def test_select_by_head_is_identity_order():
    from scripts.forge_pipeline import select_by_head
    sae, feed, records = _fixture()
    out = select_by_head(sae, feed, records)
    assert out == sorted(records.keys())


def test_select_by_firing_rate_orders_by_firing():
    from scripts.forge_pipeline import select_by_firing_rate
    sae, feed, records = _fixture()
    out = select_by_firing_rate(sae, feed, records)
    # By construction feature j fires (j+1)*2 times → highest j wins.
    assert out == sorted(records.keys(), reverse=True)


def test_select_by_firing_rate_is_deterministic():
    from scripts.forge_pipeline import select_by_firing_rate
    sae, feed, records = _fixture()
    a = select_by_firing_rate(sae, feed, records)
    b = select_by_firing_rate(sae, feed, records)
    assert a == b


def test_select_by_firing_rate_tiebreak_on_id_ascending():
    """With identical firing rates, ties resolve to ascending feature id."""
    from scripts.forge_pipeline import select_by_firing_rate
    from smsae.sae.data import Feed
    from smsae.sae.models import TopKSAE

    torch.manual_seed(0)
    D, F = 4, 4
    sae = TopKSAE(input_dim=D, n_features=F, k=1)
    with torch.no_grad():
        sae.W_dec.zero_()
        sae.W_enc.zero_()
        for j in range(F):
            sae.W_dec[j, j] = 1.0
            sae.W_enc[j, j] = 1.0
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    # Every feature fires exactly once.
    X = torch.eye(F, D)
    feed = Feed(
        name="tie", X=X,
        sample_names=[f"s{i}" for i in range(F)],
        sample_features=[{f"f{i}"} for i in range(F)],
        feature_vocab=[f"f{i}" for i in range(F)],
    )
    records = {j: object() for j in range(F)}
    out = select_by_firing_rate(sae, feed, records)
    assert out == sorted(records.keys())  # all ties → ascending id


def test_select_by_gt_alignment_returns_permutation():
    from scripts.forge_pipeline import select_by_gt_alignment
    sae, feed, records = _fixture()
    out = select_by_gt_alignment(sae, feed, records)
    assert sorted(out) == sorted(records.keys())


def test_select_by_gt_alignment_is_deterministic():
    from scripts.forge_pipeline import select_by_gt_alignment
    sae, feed, records = _fixture()
    a = select_by_gt_alignment(sae, feed, records)
    b = select_by_gt_alignment(sae, feed, records)
    assert a == b


def test_resolve_selector_accepts_keys_and_callables():
    from scripts.forge_pipeline import (
        SELECTORS, _resolve_selector, select_by_head,
    )
    assert _resolve_selector("head") is select_by_head
    assert _resolve_selector("firing_rate") is SELECTORS["firing_rate"]

    def custom(sae, feed, records):
        return list(records.keys())
    assert _resolve_selector(custom) is custom


def test_resolve_selector_rejects_unknown_keys():
    import pytest
    from scripts.forge_pipeline import _resolve_selector
    with pytest.raises(ValueError):
        _resolve_selector("not_a_real_selector")
    with pytest.raises(TypeError):
        _resolve_selector(42)
