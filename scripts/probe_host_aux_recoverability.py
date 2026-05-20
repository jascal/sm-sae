"""Probe how recoverable each of the 5 aux labels is from a frozen host's
pooled hidden state.

Diagnostic (recommended, not gating) from `aux-supervise-cascade-host` task
§9.1. Trains a tiny linear classifier from `mean-pool(model.transformer(x).
last_hidden_state)` → each aux label, with the host's weights frozen, and
reports per-label AUC.

Two questions this answers:

  (a) Were the aux labels already recoverable from the un-aux-trained host's
      hidden state? High baseline AUC predicts the v1 aux-supervised head
      will struggle to add information (since the labels are already linearly
      encoded). That'd predict a flat gate-7.3 outcome.

  (b) On the labels where supervision DID add information, what's the
      magnitude (`ΔAUC = aux-trained − baseline`)?

Usage:

    python scripts/probe_host_aux_recoverability.py \
        --baseline-host runs/cascade_host/61/host \
        --aux-host      runs/cascade_host/61_aux/host

Cheap (~5 minutes on CPU). Runs before the more-expensive forge_pipeline.py
re-run; gives a per-label predictor of the gate-7.3 outcome before paying
the forge cost.
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


def _load_host(host_dir: Path):
    """Load a saved cascade host (GPT2LMHeadModel.from_pretrained)."""
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(str(host_dir))
    model.eval()
    return model


def _build_probe_data(
    n_trajectories: int, seed: int, max_seq: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Materialize (input_ids, aux_labels) tensors. Same dataset both
    hosts probe against for a fair AUC comparison."""
    from smsae.sae.data import cascade_transitions

    rows = list(cascade_transitions(
        n_trajectories=n_trajectories, seed=seed, max_seq=max_seq,
        with_aux=True,
    ))
    if not rows:
        raise RuntimeError("cascade_transitions produced no rows; check config")
    input_ids = torch.from_numpy(np.stack([r[0] for r in rows], axis=0))
    aux_labels = np.stack([r[2] for r in rows], axis=0)
    return input_ids, aux_labels


def _pooled_hidden(model, input_ids: torch.Tensor, batch_size: int = 32) -> np.ndarray:
    """Forward `input_ids` through `model` and return mean-pool of the
    final hidden state per row. Frozen model, no grad."""
    model.eval()
    pooled_rows: list[np.ndarray] = []
    n = input_ids.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            x = input_ids[start:start + batch_size]
            out = model(input_ids=x, output_hidden_states=True)
            h = out.hidden_states[-1]  # (B, T, n_embd)
            p = h.mean(dim=1).cpu().numpy()  # (B, n_embd)
            pooled_rows.append(p)
    return np.concatenate(pooled_rows, axis=0)


def _per_label_auc(hidden: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Train a tiny logistic-regression probe per aux label; report AUC
    on a held-out split. Uses sklearn (already a polygram[regrow] dep
    and likely already available)."""
    from smsae.host.aux_labels import aux_label_names

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(
            "probe_host_aux_recoverability requires scikit-learn. "
            "Install via `pip install scikit-learn` or the polygram[regrow] "
            "extra (also pulls scikit-learn).") from e

    names = aux_label_names()
    aucs: dict[str, float] = {}
    for i, name in enumerate(names):
        y = labels[:, i]
        # Skip degenerate per-label: AUC is undefined if all labels match
        if y.min() == y.max():
            aucs[name] = float("nan")
            continue
        X_train, X_test, y_train, y_test = train_test_split(
            hidden, y, test_size=0.3, random_state=0, stratify=y,
        )
        clf = LogisticRegression(max_iter=2000, solver="liblinear")
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        aucs[name] = float(roc_auc_score(y_test, proba))
    return aucs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-host", required=True, type=Path,
                    help="Directory containing a non-aux-trained cascade host.")
    ap.add_argument("--aux-host", default=None, type=Path,
                    help="(Optional) Directory containing an aux-trained host. "
                         "If omitted, only baseline AUCs are reported.")
    ap.add_argument("--n-trajectories", type=int, default=500,
                    help="Trajectories to sample for the probe dataset.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=32)
    ap.add_argument("--out", default=None, type=Path,
                    help="Optional JSON output path. Defaults to "
                         "<aux-host or baseline-host>/aux_recoverability.json.")
    args = ap.parse_args()

    print(f"[probe] building probe dataset...")
    t0 = time.time()
    input_ids, aux_labels = _build_probe_data(
        n_trajectories=args.n_trajectories, seed=args.seed, max_seq=args.max_seq,
    )
    print(f"[probe]   {input_ids.shape[0]} samples ({time.time() - t0:.1f}s)")

    print(f"[probe] baseline host: {args.baseline_host}")
    baseline = _load_host(args.baseline_host)
    baseline_h = _pooled_hidden(baseline, input_ids)
    baseline_aucs = _per_label_auc(baseline_h, aux_labels)
    print(f"[probe]   baseline AUCs:")
    for name, auc in baseline_aucs.items():
        print(f"     {name:<24s}  {auc:.4f}")

    delta_aucs: dict[str, float | None] = {}
    aux_aucs: dict[str, float | None] = {}
    if args.aux_host is not None:
        print(f"[probe] aux-trained host: {args.aux_host}")
        aux = _load_host(args.aux_host)
        aux_h = _pooled_hidden(aux, input_ids)
        aux_aucs = _per_label_auc(aux_h, aux_labels)
        print(f"[probe]   aux-trained AUCs (ΔAUC vs baseline):")
        for name in baseline_aucs:
            base = baseline_aucs[name]
            this = aux_aucs[name]
            delta = this - base if (np.isfinite(base) and np.isfinite(this)) else float("nan")
            delta_aucs[name] = delta
            sign = "+" if delta > 0 else ""
            print(f"     {name:<24s}  {this:.4f}  ({sign}{delta:+.4f})")
    else:
        print("[probe]   no --aux-host supplied; skipping ΔAUC computation")

    out_path = args.out
    if out_path is None:
        anchor = args.aux_host if args.aux_host is not None else args.baseline_host
        out_path = Path(anchor) / "aux_recoverability.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline_host":     str(args.baseline_host),
        "aux_host":          str(args.aux_host) if args.aux_host else None,
        "n_trajectories":    int(args.n_trajectories),
        "n_samples":         int(input_ids.shape[0]),
        "seed":              int(args.seed),
        "baseline_aucs":     baseline_aucs,
        "aux_aucs":          aux_aucs or None,
        "delta_aucs":        delta_aucs or None,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[probe] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
