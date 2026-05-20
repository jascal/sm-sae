"""Unit tests for the post-polygram-compression Axis-A / Axis-B scorers."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _fixture():
    """A coordinate-aligned TopK SAE on one-hot inputs, mirroring the
    setup in test_selectors.py so each feature is uniquely driven by
    one input dimension and the AUCs are exactly 1.0 or 0.5.
    """
    from smsae.sae.data import Feed
    from smsae.sae.models import TopKSAE

    torch.manual_seed(0)
    D, F = 8, 6
    sae = TopKSAE(input_dim=D, n_features=F, k=2)
    with torch.no_grad():
        sae.W_dec.zero_()
        sae.W_enc.zero_()
        for j in range(F):
            sae.W_dec[j, j] = 1.0
            sae.W_enc[j, j] = 1.0
        sae.b_enc.zero_()
        sae.b_dec.zero_()

    rows = []
    feats = []
    vocab = [f"feat_{j}" for j in range(F)]
    for j in range(F):
        for _ in range(3):
            x = torch.zeros(D)
            x[j] = 1.0
            rows.append(x)
            feats.append({vocab[j]})
    X = torch.stack(rows, dim=0)
    feed = Feed(
        name="post_compress",
        X=X,
        sample_names=[f"s{i}" for i in range(len(rows))],
        sample_features=feats,
        feature_vocab=vocab,
    )
    return sae, feed, F


def test_post_a_keeping_all_features_matches_baseline_recon():
    from smsae.sae.evaluation import score_post_compression_reconstruction
    sae, feed, F = _fixture()
    out = score_post_compression_reconstruction(sae, feed, list(range(F)))
    # Coordinate-aligned identity SAE reconstructs perfectly.
    assert out["n_kept"] == F
    assert out["n_total"] == F
    assert out["var_explained"] == \
        np.float32(out["var_explained"])  # finite
    assert out["var_explained"] >= 0.99, out
    assert out["recon_loss_mse"] < 1e-6, out


def test_post_a_dropping_all_features_gives_zero_or_negative_var_explained():
    from smsae.sae.evaluation import score_post_compression_reconstruction
    sae, feed, _F = _fixture()
    out = score_post_compression_reconstruction(sae, feed, [])
    # With no kept features the decoder collapses to b_dec (zero in this
    # fixture). Residual variance ≥ input variance → var_explained ≤ 0.
    assert out["n_kept"] == 0
    assert out["var_explained"] <= 0.0 + 1e-6, out


def test_post_a_dropping_half_the_features_degrades_recon():
    from smsae.sae.evaluation import score_post_compression_reconstruction
    sae, feed, F = _fixture()
    full = score_post_compression_reconstruction(sae, feed, list(range(F)))
    half = score_post_compression_reconstruction(sae, feed, list(range(F // 2)))
    assert half["var_explained"] < full["var_explained"]
    assert half["n_kept"] == F // 2


def test_post_b_keeping_all_features_matches_baseline_alignment():
    from smsae.sae.evaluation import (
        auc_matrix,
        build_gt_matrix,
        score_post_compression_gt,
        score_sae,
    )
    sae, feed, F = _fixture()
    Z = score_sae(sae, feed)
    Y = build_gt_matrix(feed)
    A_full = auc_matrix(Z, Y)
    best_full = A_full.max(axis=0)

    out = score_post_compression_gt(sae, feed, list(range(F)))
    assert out["n_kept"] == F
    assert out["n_gt_features"] == Y.shape[1]
    assert abs(out["mean_best_auc"] - float(best_full.mean())) < 1e-6
    assert abs(out["coverage_0.95"]
               - float((best_full >= 0.95).mean())) < 1e-6
    assert abs(out["coverage_0.90"]
               - float((best_full >= 0.90).mean())) < 1e-6


def test_post_b_dropping_all_features_yields_chance_alignment():
    from smsae.sae.evaluation import score_post_compression_gt
    sae, feed, _F = _fixture()
    out = score_post_compression_gt(sae, feed, [])
    assert out["n_kept"] == 0
    assert out["mean_best_auc"] == 0.5
    assert out["coverage_0.95"] == 0.0
    assert out["coverage_0.90"] == 0.0


def test_post_b_dropping_one_feature_drops_one_gt_match():
    """The coordinate-aligned fixture has a 1:1 SAE↔GT pairing, so
    dropping a single SAE feature should drop coverage by exactly 1/F.
    """
    from smsae.sae.evaluation import score_post_compression_gt
    sae, feed, F = _fixture()
    full = score_post_compression_gt(sae, feed, list(range(F)))
    short = score_post_compression_gt(sae, feed, list(range(1, F)))
    assert short["n_kept"] == F - 1
    assert abs((full["coverage_0.95"] - short["coverage_0.95"])
               - 1.0 / F) < 1e-6
