"""Phase B: Ground-truth feature alignment scoring.

Question: when a trained SAE produces N learned features and we know the M
ground-truth physical features (per-charge axes, color, generation, flavor,
particle type, etc.) — how well does the SAE recover them?

For each (sae_feature f, ground_truth_feature g) pair we compute the AUC of
"sample i has feature g" vs "sae feature f activates on sample i".  This is
a chance-corrected, threshold-free measure that works whether f is binary
(top-k) or continuous (L1, JumpReLU).

Outputs per trained SAE:
  - alignment matrix A : (n_sae_features, n_gt_features)  of AUCs in [0, 1]
  - per-GT-feature best-match: which SAE feature aligned best, and at what AUC
  - per-SAE-feature best-match: which GT feature it most resembles
  - coverage: fraction of GT features with at least one SAE feature at AUC >= 0.95
  - monosemanticity: fraction of *active* SAE features that have a clean GT match
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from smsae.sae.data import Feed, feed_cascade, feed_embedded, feed_raw
from smsae.sae.models import make_sae


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC of `scores` against binary `labels`. Returns 0.5 on ties."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(-scores, kind="stable")
    rank_sum = 0.0
    seen_pos = 0
    for i, idx in enumerate(order):
        if labels[idx]:
            rank_sum += i
            seen_pos += 1
    # Mann-Whitney U: AUC = (n_pos*n_neg - U) / (n_pos*n_neg)
    U = rank_sum - n_pos * (n_pos - 1) / 2.0
    auc = 1.0 - U / (n_pos * n_neg)
    return float(auc)


@dataclass
class AlignmentReport:
    run_id: str
    feed_name: str
    variant: str
    sae_feature_count: int
    gt_feature_count: int
    alignment: np.ndarray            # (n_sae, n_gt)
    gt_vocab: list[str]
    sae_features_active: int         # how many SAE features ever activated
    coverage_at_0_95: float          # fraction of GT features with max AUC >= 0.95
    coverage_at_0_90: float
    mean_best_auc: float             # average max-AUC per GT feature
    monosemanticity: float           # frac of active SAE features with clean GT match (>=0.95)
    top_matches: list[dict]          # best (GT, SAE) matches above 0.95


def build_gt_matrix(feed: Feed) -> np.ndarray:
    """(N, M) binary matrix: row i, col j = 1 iff sample i has GT feature j."""
    N = feed.N
    M = len(feed.feature_vocab)
    vocab_idx = {f: j for j, f in enumerate(feed.feature_vocab)}
    Y = np.zeros((N, M), dtype=np.float32)
    for i, feats in enumerate(feed.sample_features):
        for f in feats:
            if f in vocab_idx:
                Y[i, vocab_idx[f]] = 1.0
    return Y


def score_sae(sae, feed: Feed) -> np.ndarray:
    """(N, n_features) activation magnitudes."""
    with torch.no_grad():
        out = sae(feed.X)
    return out.z.detach().cpu().numpy()


def align(Z: np.ndarray, Y: np.ndarray, gt_vocab: list[str]) -> AlignmentReport:
    """Compute the full alignment matrix and aggregate statistics.

    Z: (N, n_sae_features) -- continuous activations
    Y: (N, n_gt_features)  -- binary
    """
    N, n_sae = Z.shape
    M = Y.shape[1]
    active_mask = (Z.max(axis=0) > 1e-9)
    A = np.full((n_sae, M), 0.5, dtype=np.float32)
    for f in range(n_sae):
        if not active_mask[f]:
            continue
        col = Z[:, f]
        for j in range(M):
            A[f, j] = _auc(col, Y[:, j])

    # Per-GT best match
    if A.shape[0] == 0:
        best_auc_per_gt = np.zeros(M)
    else:
        best_auc_per_gt = A.max(axis=0)
    coverage_95 = float((best_auc_per_gt >= 0.95).mean())
    coverage_90 = float((best_auc_per_gt >= 0.90).mean())
    mean_best = float(best_auc_per_gt.mean())

    # Per-SAE-feature best GT match (only for active features)
    monosem_count = 0
    n_active = int(active_mask.sum())
    if n_active > 0:
        for f in range(n_sae):
            if not active_mask[f]:
                continue
            if A[f].max() >= 0.95:
                monosem_count += 1
    monosem = monosem_count / max(n_active, 1)

    top_matches = []
    for j in range(M):
        if best_auc_per_gt[j] >= 0.95:
            f_best = int(A[:, j].argmax())
            top_matches.append({
                "gt_feature": gt_vocab[j],
                "sae_feature_idx": f_best,
                "auc": float(A[f_best, j]),
            })
    top_matches.sort(key=lambda m: -m["auc"])

    return AlignmentReport(
        run_id="", feed_name="", variant="",
        sae_feature_count=n_sae, gt_feature_count=M,
        alignment=A, gt_vocab=gt_vocab,
        sae_features_active=n_active,
        coverage_at_0_95=coverage_95,
        coverage_at_0_90=coverage_90,
        mean_best_auc=mean_best,
        monosemanticity=monosem,
        top_matches=top_matches,
    )


def load_sae(ckpt_path: str):
    obj = torch.load(ckpt_path, weights_only=False)
    kind, input_dim, n_features = obj["kind"], obj["input_dim"], obj["n_features"]
    if kind == "topk":
        sae = make_sae("topk", input_dim, n_features, k=6)  # k from training; not stored
    elif kind == "l1":
        sae = make_sae("l1", input_dim, n_features, l1_coeff=1e-2)
    else:
        sae = make_sae("jumprelu", input_dim, n_features, l0_coeff=3e-3)
    sae.load_state_dict(obj["state_dict"])
    sae.eval()
    return sae, obj


def main():
    import os
    runs_dir = "runs"
    feed_builders = {
        "raw":      feed_raw,
        "embedded": lambda: feed_embedded(embed_dim=16, seed=0),
        "cascade":  lambda: feed_cascade(n_events=2000, seed=0),
    }
    # Cache feeds (cascade takes time)
    feeds = {name: builder() for name, builder in feed_builders.items()}
    gt_matrices = {name: build_gt_matrix(f) for name, f in feeds.items()}

    reports = []
    for feed_name in ("raw", "embedded", "cascade"):
        for variant in ("topk", "l1", "jumprelu"):
            run_id = f"{feed_name}__{variant}"
            ckpt = os.path.join(runs_dir, f"{run_id}.pt")
            if not os.path.exists(ckpt):
                print(f"  skip (no ckpt): {ckpt}")
                continue
            sae, obj = load_sae(ckpt)
            feed = feeds[feed_name]
            Y = gt_matrices[feed_name]
            Z = score_sae(sae, feed)
            rep = align(Z, Y, feed.feature_vocab)
            rep.run_id = run_id; rep.feed_name = feed_name; rep.variant = variant
            reports.append(rep)

    # Summary table
    print("=" * 88)
    print("ALIGNMENT REPORT  (AUC of SAE feature vs ground-truth feature, max over SAE)")
    print("=" * 88)
    print(f"{'feed':<10s} {'variant':<10s} {'n_sae':>6s} {'active':>7s} "
          f"{'cov>=.95':>9s} {'cov>=.90':>9s} {'mean_best':>10s} {'monosem':>9s}")
    print("-" * 88)
    for r in reports:
        print(f"{r.feed_name:<10s} {r.variant:<10s} {r.sae_feature_count:>6d} "
              f"{r.sae_features_active:>7d} {r.coverage_at_0_95:>9.1%} "
              f"{r.coverage_at_0_90:>9.1%} {r.mean_best_auc:>10.4f} "
              f"{r.monosemanticity:>9.1%}")

    # Per-run detail: which GT features were recovered cleanly
    for r in reports:
        print(f"\n--- {r.run_id}: top {min(15, len(r.top_matches))} recovered GT features ---")
        if not r.top_matches:
            print("    (none reached AUC >= 0.95)")
            continue
        for m in r.top_matches[:15]:
            print(f"    AUC={m['auc']:.4f}  GT={m['gt_feature']:<35s}  SAE feature #{m['sae_feature_idx']}")

    # Persist
    summary = []
    for r in reports:
        summary.append({
            "run_id": r.run_id,
            "feed": r.feed_name, "variant": r.variant,
            "n_sae_features": r.sae_feature_count,
            "n_active": r.sae_features_active,
            "n_gt_features": r.gt_feature_count,
            "coverage_0.95": r.coverage_at_0_95,
            "coverage_0.90": r.coverage_at_0_90,
            "mean_best_auc": r.mean_best_auc,
            "monosemanticity": r.monosemanticity,
            "n_top_matches": len(r.top_matches),
        })
    with open(os.path.join(runs_dir, "alignment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {runs_dir}/alignment_summary.json")


if __name__ == "__main__":
    main()
