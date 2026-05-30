"""Salience-headroom probe on sm-sae — testing the concise-via-routing heuristic.

Imports the recipe-agnostic diagnostic from sae-forge (``saeforge.isf``) and
applies it to sm-sae's raw substrate. The concise-via-routing methodology
(sae-forge ``docs/concise-via-routing.md``) leans on a **rule of thumb, not a
law**: a specialist tends to pay off only where the host substrate can't
already read the concept — i.e. where ``salience_headroom = 1 − host_auc`` is
high. This script measures that headroom on sm-sae's factorial physical
features.

sm-sae is the natural stress test for the *low-headroom* end of the heuristic:
the substrate's coordinates **are** the conserved charges (Q, B, L_e, L_mu,
L_tau, C3, C8, spin, mass), so a feature that *is* a conserved quantity should
be read off the raw substrate at AUC ≈ 1 (≈ zero headroom) — the heuristic then
predicts a specialist would add little there.

Honest caveats (the heuristic is rough): headroom is a *cheap prior* for where
to look, not a predictor of lift. On econ-sae the conjunctive tier had LOW
headroom yet got the biggest routing lift (an objective-aligned specialist beat
the substrate even where it wasn't weak), so a low headroom here does **not**
prove a specialist would be useless — only that the substrate already encodes
the concept linearly. N is also tiny (61 particles), so per-feature AUCs are
noisy; we filter to features with ≥2 positives/negatives and read namespaces,
not individual features.

Outputs ``docs/salience_headroom_summary.json``.

Usage::

    python scripts/salience_headroom_validation.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from saeforge.isf import best_auc_per_label, salience_headroom
from smsae.sae.data import feed_raw
from smsae.sae.evaluation import build_gt_matrix


def main() -> None:
    feed = feed_raw()
    X = np.asarray(feed.X, dtype=np.float64)          # raw substrate (conserved charges)
    Y = build_gt_matrix(feed).astype(np.uint8)
    vocab = list(feed.feature_vocab)
    print(f"sm-sae feed_raw: substrate X={X.shape} (coords={X.shape[1]}), labels={Y.shape}")

    host_auc = best_auc_per_label(X, Y)               # best-of-coord AUC per feature
    head = salience_headroom(host_auc)
    n_pos = Y.sum(axis=0)
    n_neg = Y.shape[0] - n_pos
    keep = (n_pos >= 2) & (n_neg >= 2) & ~np.isnan(host_auc)

    by_ns: dict[str, list[float]] = defaultdict(list)
    per_feature = []
    for v, a, h, k in zip(vocab, host_auc, head, keep):
        if not k:
            continue
        ns = v.split(":")[0]
        by_ns[ns].append(float(h))
        per_feature.append({"feature": v, "namespace": ns,
                            "host_auc": float(a), "headroom": float(h)})

    print(f"\nkept {int(keep.sum())}/{len(vocab)} features (n_pos>=2, n_neg>=2)")
    print(f"{'namespace':16s} {'n':>3s} {'mean_host_auc':>13s} {'mean_headroom':>13s}")
    ns_rows = {}
    for ns in sorted(by_ns, key=lambda p: float(np.mean(by_ns[p]))):
        hr = float(np.mean(by_ns[ns]))
        # mean host AUC for this namespace
        aucs = [pf["host_auc"] for pf in per_feature if pf["namespace"] == ns]
        ns_rows[ns] = {"n": len(by_ns[ns]), "mean_host_auc": float(np.mean(aucs)),
                       "mean_headroom": hr}
        print(f"{ns:16s} {len(by_ns[ns]):3d} {np.mean(aucs):13.3f} {hr:13.3f}")

    overall_head = float(np.mean([h for v in by_ns.values() for h in v]))
    overall_auc = float(np.nanmean(host_auc[keep]))
    print(f"\nOVERALL  mean host AUC = {overall_auc:.3f}  mean headroom = {overall_head:.3f}")

    summary = {
        "fixture": "sm-sae (feed_raw — conserved-charge substrate)",
        "primitive": "saeforge.isf (best_auc_per_label + salience_headroom)",
        "thesis": "docs/concise-via-routing.md (sae-forge) — salience HEURISTIC (rule of thumb)",
        "substrate_dim": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "n_features_kept": int(keep.sum()),
        "overall_mean_host_auc": overall_auc,
        "overall_mean_headroom": overall_head,
        "per_namespace": ns_rows,
        "reading": (
            "sm-sae's factorial physical features mostly sit at LOW headroom on "
            "the conserved-charge substrate (the coords ARE the conserved "
            "quantities), so the salience heuristic predicts a specialist would "
            "add little for the charge/color/spin-derived features. This probes "
            "the low-headroom end of the rule of thumb — it is NOT proof a "
            "specialist is useless (econ-sae's conjunctive tier was low-headroom "
            "yet gained from a dedicated specialist), only that the substrate "
            "already encodes these concepts linearly. N=61 is tiny; read "
            "namespaces, not individual features."
        ),
    }
    out = REPO_ROOT / "docs" / "salience_headroom_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
