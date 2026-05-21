"""Per-feature discriminative entropy of the cascade rollout vocabulary.

Closes the diagnostic loop opened by the gate-7.3 lineage. The
previous arc (PRs #19 → #27) ruled out every host-side lever and
SAE family as the binding constraint. The remaining hypothesis is:
**the cascade rollout vocabulary itself is information-capped** —
state_t does not contain enough discriminative entropy to predict
per-particle features of state_{t+1} with high reliability,
regardless of host quality.

This script tests that directly. For each of the 110 GT features:

1. Compute marginal P(feature fires in state_{t+1}).
2. Train a logistic-regression classifier from state_t (the
   61-dim bag-of-particles vector — NO LM in between) to the
   feature label. Report AUC.
3. Compare to the probe's per-feature AUC (from LM-pooled
   hidden state).

Three interpretations:

- **state_t-AUC ≈ random (~0.5):** the cascade rollout fundamentally
  cannot disambiguate this feature. No host/SAE combo helps.
- **state_t-AUC ≈ probe-AUC:** host is faithful — the LM is
  preserving the state's information; ceiling is set by state_t
  itself.
- **state_t-AUC > probe-AUC:** the LM is DROPPING info — there's
  room for host improvement (this would contradict the sweep
  findings; we expect this case to be empty).

Run:
    python scripts/cascade_rollout_entropy.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np


# Reuse the spotlight set from the capacity sweep — these were the
# features the gate-7.3 investigation flagged as under-trained.
SPOTLIGHT_FEATURES = [
    "color:r", "color:b", "color:g",
    "flavor:u", "flavor:d", "flavor:mu",
    "particle:mu+", "particle:u_b",
]


def _binary_entropy(p: float) -> float:
    """Shannon binary entropy in bits, with the convention 0*log(0) = 0."""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _build_rollout_dataset(n_trajectories: int, seed: int):
    """Materialise (state_t_bag, gt_label_matrix) for the 110-feature
    GT vocabulary. state_t is the 61-dim bag-of-particles count vector
    (matches the cascade feed encoding); gt labels are the OR over
    per-particle feature sets of state_{t+1}."""
    import random

    from smsae.sae.data import (
        all_ground_truth_features, particle_features,
    )
    from smsae.sm.cascade import build_decay_catalog, cascade as cascade_fn
    from smsae.sm.embeddings import build_sm

    feature_vocab = all_ground_truth_features()
    feat_to_idx = {f: i for i, f in enumerate(feature_vocab)}

    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)
    sm = build_sm()
    catalog = build_decay_catalog(sm)
    particle_names = sorted(sm.keys())
    particle_to_col = {n: i for i, n in enumerate(particle_names)}

    start_distribution = {
        "H": 1.0, "Z": 1.0, "W+": 1.0, "W-": 1.0,
        "t_r": 1.0, "t_g": 1.0, "t_b": 1.0,
        "tau-": 0.5, "tau+": 0.5,
        "mu-": 0.3, "mu+": 0.3,
    }
    starts = list(start_distribution.keys())
    weights = np.array(list(start_distribution.values()), dtype=float)
    weights /= weights.sum()

    state_t_rows: list[np.ndarray] = []
    gt_rows: list[np.ndarray] = []

    for _ in range(n_trajectories):
        parent = rng_np.choice(starts, p=weights)
        history = cascade_fn(sm, {parent: 1}, catalog, max_steps=30, rng=rng_py)
        for t in range(len(history) - 1):
            state_t = history[t][0]
            state_tp1 = history[t + 1][0]
            # state_t as a 61-dim count vector
            x = np.zeros(len(particle_names), dtype=np.float32)
            for particle, count in state_t.items():
                if particle in particle_to_col:
                    x[particle_to_col[particle]] = count
            state_t_rows.append(x)
            # GT labels for state_{t+1}
            row = np.zeros(len(feature_vocab), dtype=np.float32)
            for particle, count in state_tp1.items():
                if count <= 0:
                    continue
                for f in particle_features(particle):
                    if f in feat_to_idx:
                        row[feat_to_idx[f]] = 1.0
            gt_rows.append(row)

    if not state_t_rows:
        raise RuntimeError("no rollout samples generated; check rollout config")
    X = np.stack(state_t_rows, axis=0)
    Y = np.stack(gt_rows, axis=0)
    return X, Y, feature_vocab


def _per_feature_auc(X: np.ndarray, Y: np.ndarray,
                     feature_vocab: list[str]) -> dict[str, float | None]:
    """Standard per-feature LogisticRegression probe + AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    aucs: dict[str, float | None] = {}
    for i, name in enumerate(feature_vocab):
        y = Y[:, i]
        if y.min() == y.max():
            aucs[name] = None
            continue
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=0, stratify=y,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=0,
            )
        clf = LogisticRegression(max_iter=2000, solver="liblinear")
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        try:
            aucs[name] = float(roc_auc_score(y_te, proba))
        except ValueError:
            aucs[name] = None
    return aucs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-trajectories", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--probe-json", default="runs/aux_probe/full_gt_recoverability.json",
        type=Path,
        help="path to the prior probe JSON whose baseline_aucs_residual "
             "we compare against (defaults to PR #22's output).",
    )
    ap.add_argument(
        "--out", default="runs/aux_probe/rollout_entropy.json", type=Path,
    )
    args = ap.parse_args()

    print(f"[entropy] building rollout dataset (n_trajectories={args.n_trajectories})...")
    t0 = time.time()
    X, Y, feature_vocab = _build_rollout_dataset(
        n_trajectories=args.n_trajectories, seed=args.seed,
    )
    print(f"[entropy]   {X.shape[0]} samples, {Y.shape[1]} GT features, "
          f"{X.shape[1]}-dim state vector ({time.time() - t0:.1f}s)")

    # Marginal probabilities + Shannon entropy
    marginals = Y.mean(axis=0)
    marginal_entropies = np.array([_binary_entropy(float(p)) for p in marginals])

    # state_t-direct AUC (no LM)
    print("[entropy] training per-feature classifiers from state_t directly...")
    t0 = time.time()
    state_t_aucs = _per_feature_auc(X, Y, feature_vocab)
    print(f"[entropy]   done ({time.time() - t0:.1f}s)")

    # Load probe AUCs for comparison
    probe_path = Path(args.probe_json)
    if not probe_path.is_absolute():
        probe_path = Path(REPO_ROOT) / probe_path
    probe_aucs: dict[str, float | None] = {}
    if probe_path.exists():
        with open(probe_path) as f:
            probe_data = json.load(f)
        probe_aucs = probe_data.get("baseline_aucs_residual") or {}
    else:
        print(f"[entropy]   warning: no probe data at {probe_path}; skipping comparison")

    # Per-feature analysis
    rows = []
    for i, name in enumerate(feature_vocab):
        s = state_t_aucs.get(name)
        p = probe_aucs.get(name)
        rows.append({
            "feature":              name,
            "marginal_p":           float(marginals[i]),
            "marginal_entropy_bits": float(marginal_entropies[i]),
            "state_t_auc":          s,
            "probe_auc_lm":         p,
            "lm_drop":              (s - p) if (s is not None and p is not None) else None,
        })

    # Bucket analysis
    measured = [r for r in rows if r["state_t_auc"] is not None]
    state_t_ceiling = [r["state_t_auc"] for r in measured]
    spotlight_rows = [r for r in rows if r["feature"] in SPOTLIGHT_FEATURES]

    summary = {
        "n_trajectories":   args.n_trajectories,
        "n_samples":        int(X.shape[0]),
        "n_features":       int(Y.shape[1]),
        "n_features_measurable": len(measured),
        "state_t_auc_summary": {
            "mean":   round(float(np.mean(state_t_ceiling)), 4),
            "median": round(float(np.median(state_t_ceiling)), 4),
            "pct_ge_0.9":  round(sum(1 for v in state_t_ceiling if v >= 0.9) / len(state_t_ceiling), 4),
            "pct_ge_0.95": round(sum(1 for v in state_t_ceiling if v >= 0.95) / len(state_t_ceiling), 4),
            "pct_lt_0.7":  round(sum(1 for v in state_t_ceiling if v < 0.7) / len(state_t_ceiling), 4),
        },
        "spotlight": [
            {
                "feature":        r["feature"],
                "state_t_auc":    r["state_t_auc"],
                "probe_auc_lm":   r["probe_auc_lm"],
                "lm_drop":        r["lm_drop"],
                "marginal_p":     r["marginal_p"],
            }
            for r in spotlight_rows
        ],
        "all_rows":  rows,
    }

    # Headline
    print()
    print("=" * 72)
    print("CASCADE ROLLOUT INFORMATION CONTENT")
    print("=" * 72)
    print(f"  state_t AUC (74 measurable features):")
    print(f"    mean:   {summary['state_t_auc_summary']['mean']:.3f}")
    print(f"    median: {summary['state_t_auc_summary']['median']:.3f}")
    print(f"    pct ≥ 0.9:  {summary['state_t_auc_summary']['pct_ge_0.9']:.0%}")
    print(f"    pct ≥ 0.95: {summary['state_t_auc_summary']['pct_ge_0.95']:.0%}")
    print(f"    pct < 0.7:  {summary['state_t_auc_summary']['pct_lt_0.7']:.0%}")

    print()
    print("  Spotlight (the previously-weak per-particle features):")
    print(f"  {'feature':<28} {'state_t':>10} {'probe_LM':>10} {'lm_drop':>10}")
    for r in summary["spotlight"]:
        s_str = f"{r['state_t_auc']:.3f}" if r['state_t_auc'] is not None else "—"
        p_str = f"{r['probe_auc_lm']:.3f}" if r['probe_auc_lm'] is not None else "—"
        d_str = f"{r['lm_drop']:+.3f}" if r['lm_drop'] is not None else "—"
        print(f"  {r['feature']:<28} {s_str:>10} {p_str:>10} {d_str:>10}")

    out_path = args.out
    if not out_path.is_absolute():
        out_path = Path(REPO_ROOT) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[entropy] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
