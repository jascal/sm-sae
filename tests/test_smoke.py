"""Minimal smoke tests for the sm-sae package.

Run with: pytest -q tests/
"""

from __future__ import annotations

import numpy as np


def test_sm_build_61_particles():
    from smsae.sm.embeddings import build_sm
    sm = build_sm()
    assert len(sm) == 61


def test_vertex_catalog_168():
    from smsae.sm.checks import vertex_catalog
    assert len(vertex_catalog()) == 168


def test_vertex_closure_all():
    """Every catalog vertex must close on the conserved-charge subspace."""
    from smsae.sm.checks import vertex_catalog
    from smsae.sm.embeddings import CONSERVED, build_sm, vertex_residual
    sm = build_sm()
    for desc, inc, out in vertex_catalog():
        r = vertex_residual(sm, inc, out)
        assert np.max(np.abs(r)) < 1e-12, f"vertex did not close: {desc}"


def test_conservation_algebra_nullity_7():
    """Signed vertex incidence has nullity exactly 7 (= the SM conserved charges)."""
    import numpy as np
    from smsae.sm.checks import vertex_catalog, signed_incidence_matrix
    from smsae.sm.embeddings import build_sm
    sm = build_sm()
    names, B = signed_incidence_matrix(sm, vertex_catalog())
    rank = int(np.linalg.matrix_rank(B, tol=1e-9))
    nullity = B.shape[1] - rank
    assert nullity == 7, f"expected nullity 7, got {nullity}"


def test_sae_round_trip():
    """Train a tiny TopK SAE on the raw feed; it should reconstruct well."""
    import torch
    from smsae.sae.data import feed_raw
    from smsae.sae.models import make_sae
    from smsae.sae.train import TrainConfig, train
    torch.manual_seed(0)
    feed = feed_raw()
    sae = make_sae("topk", input_dim=feed.D, n_features=16, k=6)
    cfg = TrainConfig(epochs=200, batch_size=32, lr=3e-3, log_every=10_000)
    train(sae, feed, cfg, verbose=False)
    with torch.no_grad():
        out = sae(feed.X)
    var_explained = 1 - float((feed.X - out.x_hat).var() / feed.X.var())
    assert var_explained > 0.5, f"recon var-explained too low: {var_explained}"
