"""Probe how recoverable each sm-sae GT feature is from the cascade-trained
host — at two layer depths.

Diagnostic for the gate-7.3 follow-up
(`probe-full-gt-recoverability-cascade-host`). Extends
`probe_host_aux_recoverability.py`'s pattern to the **full 110-feature
GT vocabulary** (not just the 5 aux labels) and adds an optional
`--from-projected` flag that ALSO probes the post-SAE-encode feature
space, not just the raw host residual.

The two-layer measurement is load-bearing: it distinguishes:

  A. host AND SAE both carry the GT signal → bottleneck is the
     decoder/forge layer; file `investigate-cascade-jumprelu-sparsity-loss`.
  B. host carries it but SAE drops it → bottleneck is the SAE encode;
     file `investigate-projection-bottleneck-cascade`.
  C. host doesn't carry per-GT-feature signal → host objective is
     leaving info out; file `richer-cascade-host-supervision-v2`.

Run:
    python scripts/probe_full_gt_recoverability.py \\
        --baseline-host runs/cascade_host/61/host \\
        --aux-host      runs/cascade_host/61_aux/host \\
        --from-projected \\
        --n-trajectories 5000

Cheap (~5 minutes on Intel CPU). Output:
`runs/aux_probe/full_gt_recoverability.json`.
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
import torch


# Bucket-interpretation thresholds. Heuristic per the proposal; the
# per-feature AUC table is the load-bearing artefact. The bucket label
# is just a routing hint.
RESIDUAL_HIGH_AUC = 0.9      # "carries the signal"
RESIDUAL_LOW_AUC = 0.7       # "doesn't carry the signal"
HIGH_FRAC_THRESHOLD = 0.8    # ≥ 80% of features
LOW_FRAC_THRESHOLD = 0.5     # ≥ 50% of features for the "drops" / "missing" calls


def _load_host(host_dir: Path):
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(str(host_dir))
    model.eval()
    return model


def _build_probe_dataset(
    n_trajectories: int, seed: int, max_seq: int,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Materialise (input_ids, gt_label_matrix, feature_vocab).

    `gt_label_matrix` is shape (n_samples, n_gt_features); each row's
    bit-i is 1 iff the cascade state at this sample contains a particle
    that carries GT feature i.
    """
    from smsae.sae.data import (
        all_ground_truth_features, cascade_transitions, particle_features,
    )
    from smsae.sm.cascade import build_decay_catalog, cascade as cascade_fn
    from smsae.sm.embeddings import build_sm
    from smsae.sae.data import _particle_to_id, encode_state_as_ids

    feature_vocab = all_ground_truth_features()
    feat_to_idx = {f: i for i, f in enumerate(feature_vocab)}

    # cascade_transitions yields (input_ids, target_ids) — but we need
    # state_{t+1} to derive GT labels. Re-roll the cascades and produce
    # both the input_ids and the per-state GT label vector.
    import random

    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)
    sm = build_sm()
    catalog = build_decay_catalog(sm)
    name_to_id = _particle_to_id()

    start_distribution = {
        "H": 1.0, "Z": 1.0, "W+": 1.0, "W-": 1.0,
        "t_r": 1.0, "t_g": 1.0, "t_b": 1.0,
        "tau-": 0.5, "tau+": 0.5,
        "mu-": 0.3, "mu+": 0.3,
    }
    starts = list(start_distribution.keys())
    weights = np.array(list(start_distribution.values()), dtype=float)
    weights /= weights.sum()

    inputs: list[np.ndarray] = []
    gt_rows: list[np.ndarray] = []
    for _ in range(n_trajectories):
        parent = rng_np.choice(starts, p=weights)
        history = cascade_fn(sm, {parent: 1}, catalog, max_steps=30, rng=rng_py)
        for t in range(len(history) - 1):
            state_t = history[t][0]
            state_tp1 = history[t + 1][0]
            ids = encode_state_as_ids(state_t, max_seq=max_seq, name_to_id=name_to_id)
            inputs.append(ids)
            # Build the GT label row: OR over the per-particle feature sets
            # for every particle in state_{t+1}.
            row = np.zeros(len(feature_vocab), dtype=np.float32)
            for particle, count in state_tp1.items():
                if count <= 0:
                    continue
                for f in particle_features(particle):
                    if f in feat_to_idx:
                        row[feat_to_idx[f]] = 1.0
            gt_rows.append(row)

    if not inputs:
        raise RuntimeError("no probe samples generated; check rollout config")
    input_ids = torch.from_numpy(np.stack(inputs, axis=0))
    gt_matrix = np.stack(gt_rows, axis=0)
    return input_ids, gt_matrix, feature_vocab


def _pooled_hidden(model, input_ids: torch.Tensor, batch_size: int = 32) -> np.ndarray:
    """Mean-pool the final hidden state across the sequence axis."""
    model.eval()
    rows: list[np.ndarray] = []
    n = input_ids.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            x = input_ids[start:start + batch_size]
            out = model(input_ids=x, output_hidden_states=True)
            h = out.hidden_states[-1]
            rows.append(h.mean(dim=1).cpu().numpy())
    return np.concatenate(rows, axis=0)


def _project_via_sae(
    pooled: np.ndarray, sae_ckpt_path: Path,
) -> np.ndarray:
    """Encode pooled hidden states through the SAE's encoder.
    Returns shape (n_samples, n_features) — the SAE feature space."""
    from smsae.sae.evaluation import load_sae

    sae, _ = load_sae(str(sae_ckpt_path))
    sae.eval()
    with torch.no_grad():
        # Pooled is (n_samples, d_model). SAE.encode produces feature
        # activations.
        h = torch.from_numpy(pooled).to(torch.float32)
        encoded = sae.encode(h)  # (n_samples, n_features)
    return encoded.cpu().numpy()


def _per_feature_auc(
    X: np.ndarray, Y: np.ndarray, feature_vocab: list[str],
) -> dict[str, float | None]:
    """LogisticRegression probe per GT feature. AUC on a held-out split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    aucs: dict[str, float | None] = {}
    for i, name in enumerate(feature_vocab):
        y = Y[:, i]
        # Degenerate per-label: no positive or no negative samples → AUC undefined
        if y.min() == y.max():
            aucs[name] = None
            continue
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=0, stratify=y,
            )
        except ValueError:
            # Stratify fails if a class has < 2 examples → fall back to no stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=0,
            )
        clf = LogisticRegression(max_iter=2000, solver="liblinear")
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        try:
            aucs[name] = float(roc_auc_score(y_test, proba))
        except ValueError:
            aucs[name] = None
    return aucs


def _bucket_summary(
    baseline_residual: dict[str, float | None],
    baseline_projected: dict[str, float | None] | None,
) -> dict:
    """Apply the proposal's bucket A/B/C/ambiguous rules."""
    measured_residual = [v for v in baseline_residual.values() if v is not None]
    n = len(measured_residual)
    pct_resid_ge_high = (
        sum(1 for v in measured_residual if v >= RESIDUAL_HIGH_AUC) / n if n else 0.0
    )
    pct_resid_lt_low = (
        sum(1 for v in measured_residual if v < RESIDUAL_LOW_AUC) / n if n else 0.0
    )

    if baseline_projected is not None:
        measured_projected = [v for v in baseline_projected.values() if v is not None]
        np_ = len(measured_projected)
        pct_proj_ge_high = (
            sum(1 for v in measured_projected if v >= RESIDUAL_HIGH_AUC) / np_
            if np_ else 0.0
        )
    else:
        pct_proj_ge_high = None

    # Bucket assignment
    bucket = "ambiguous"
    if pct_resid_ge_high >= HIGH_FRAC_THRESHOLD:
        if pct_proj_ge_high is None:
            # Without the projected measurement we can't distinguish A vs B;
            # the rules require both. Surface "ambiguous_need_projected".
            bucket = "ambiguous_need_projected"
        elif pct_proj_ge_high >= HIGH_FRAC_THRESHOLD:
            bucket = "A"
        elif pct_proj_ge_high < LOW_FRAC_THRESHOLD:
            bucket = "B"
        else:
            bucket = "ambiguous_borderline_projected"
    elif pct_resid_lt_low >= LOW_FRAC_THRESHOLD:
        bucket = "C"

    return {
        "bucket": bucket,
        "n_features_measured_residual": n,
        "pct_residual_ge_0.9": pct_resid_ge_high,
        "pct_residual_lt_0.7": pct_resid_lt_low,
        "pct_projected_ge_0.9": pct_proj_ge_high,
        "interpretation": {
            "A": "host + SAE both carry signal; bottleneck downstream (forge / decoder).",
            "B": "host carries it but SAE drops it; bottleneck is the SAE encode.",
            "C": "host doesn't carry per-GT-feature signal; richer aux supervision needed.",
            "ambiguous": "neither A/B/C threshold met; manual triage required.",
            "ambiguous_need_projected": "--from-projected required to disambiguate A vs B.",
            "ambiguous_borderline_projected": "projected dropoff between 50%-80%; borderline A/B.",
        }.get(bucket, "<unknown>"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-host", required=True, type=Path)
    ap.add_argument("--aux-host", default=None, type=Path)
    ap.add_argument(
        "--from-projected", action="store_true",
        help="ALSO probe from the SAE feature space (post-encode). "
             "Required to distinguish bucket A from B.",
    )
    ap.add_argument(
        "--sae-checkpoint", default="runs/cascade__jumprelu.pt", type=Path,
        help="Path to the cascade SAE checkpoint for the projected probe.",
    )
    ap.add_argument("--n-trajectories", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=32)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()

    print(f"[probe] building probe dataset (n_trajectories={args.n_trajectories})")
    t0 = time.time()
    input_ids, gt_matrix, feature_vocab = _build_probe_dataset(
        n_trajectories=args.n_trajectories, seed=args.seed, max_seq=args.max_seq,
    )
    print(f"[probe]   {input_ids.shape[0]} samples, "
          f"{len(feature_vocab)} GT features ({time.time() - t0:.1f}s)")

    def _probe_host(host_path: Path, tag: str) -> dict:
        print(f"[probe] {tag} host: {host_path}")
        model = _load_host(host_path)
        pooled = _pooled_hidden(model, input_ids)
        residual_aucs = _per_feature_auc(pooled, gt_matrix, feature_vocab)
        n_meas = sum(1 for v in residual_aucs.values() if v is not None)
        print(f"[probe]   residual probe: {n_meas}/{len(feature_vocab)} features measured")

        projected_aucs: dict[str, float | None] | None = None
        if args.from_projected:
            print(f"[probe]   projecting via SAE: {args.sae_checkpoint}")
            projected = _project_via_sae(pooled, args.sae_checkpoint)
            projected_aucs = _per_feature_auc(projected, gt_matrix, feature_vocab)
            np_meas = sum(1 for v in projected_aucs.values() if v is not None)
            print(f"[probe]   projected probe: {np_meas}/{len(feature_vocab)} features measured")
        return {"residual": residual_aucs, "projected": projected_aucs}

    baseline = _probe_host(args.baseline_host, "baseline")
    aux = _probe_host(args.aux_host, "aux") if args.aux_host else None

    summary = _bucket_summary(baseline["residual"], baseline["projected"])
    print()
    print(f"[probe] bucket={summary['bucket']}; "
          f"residual ≥ 0.9: {summary['pct_residual_ge_0.9']:.0%}; "
          f"projected ≥ 0.9: "
          f"{f'{summary['pct_projected_ge_0.9']:.0%}' if summary['pct_projected_ge_0.9'] is not None else '<no --from-projected>'}")
    print(f"[probe] {summary['interpretation']}")

    out_path = args.out
    if out_path is None:
        out_path = Path(REPO_ROOT) / "runs" / "aux_probe" / "full_gt_recoverability.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "baseline_host":           str(args.baseline_host),
        "aux_host":                str(args.aux_host) if args.aux_host else None,
        "sae_checkpoint":          str(args.sae_checkpoint) if args.from_projected else None,
        "n_trajectories":          int(args.n_trajectories),
        "n_samples":               int(input_ids.shape[0]),
        "seed":                    int(args.seed),
        "gt_features":             feature_vocab,
        "baseline_aucs_residual":  baseline["residual"],
        "baseline_aucs_projected": baseline["projected"],
        "aux_aucs_residual":       aux["residual"] if aux else None,
        "aux_aucs_projected":      aux["projected"] if aux else None,
        "bucket_summary":          summary,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[probe] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
