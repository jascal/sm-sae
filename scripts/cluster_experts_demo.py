"""Test polygram 0.10.0's cluster_experts on the sm-sae cascade SAE.

The bio-sae 2026-05-19 report flagged cluster_experts as the genuinely
useful polygram entry point on real (non-toy) decoder geometry. They
validated it via after-the-fact biological plausibility checking on a
substrate without an explicit answer key.

sm-sae has 110 ground-truth particle-physics features (charge, color,
generation, family, flavour, …) per cascade sample — an actual
answer key. This script runs cluster_experts on the cascade__jumprelu
decoder and scores the recovered clusters against the GT labels.

Outputs:

  runs/cluster_experts/<run_id>/results.json
    Per-cluster best-GT-match (label, cluster-mean AUC, member count)
    plus aggregate purity / coverage numbers.

Usage:
    python scripts/cluster_experts_demo.py
    python scripts/cluster_experts_demo.py \\
        --run-id cascade__jumprelu --encoding rung5 \\
        --thresholds 0.2 0.3 0.4 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch


def _decoder_vectors_for_dictionary(sae, dictionary) -> np.ndarray:
    """(n_features, d_model) in dictionary-feature order.

    cluster_experts requires the decoder rows aligned with
    `dictionary.features`. The polygram feature names are `feat_<int>`
    where the int is the SAE's original feature_id (column of W_dec).
    """
    W_dec = sae.W_dec.detach().cpu().numpy()      # (input_dim, n_features)
    out = np.zeros((len(dictionary.features), W_dec.shape[0]),
                   dtype=np.float32)
    for i, f in enumerate(dictionary.features):
        fid = int(f.name.replace("feat_", ""))
        out[i] = W_dec[:, fid]
    return out


def _feature_id_from_name(name: str) -> int:
    return int(name.replace("feat_", ""))


def _score_cluster_against_gt(member_ids: list[int], Z: np.ndarray,
                              Y: np.ndarray, meta_idx: list[int]) -> dict:
    """For a single cluster, compute the best-matching GT label.

    Two views per cluster:
    - **best GT** (any label): the cluster-mean-AUC argmax across all
      GT columns. Cluster_mean_auc is that argmax's value.
    - **best META label** (non-`particle:` / non-`flavor:`): the same
      computation restricted to meta-axis labels (`kind:`, `origin:`,
      `is_*`, `charge_sign:`, `color:`, `generation:`). This is the
      bio-sae-style "cluster captures meta-structure no single
      per-feature label encodes" view.
    """
    from smsae.sae.evaluation import auc_matrix
    if not member_ids:
        return {"members": 0, "best_gt": -1, "cluster_mean_auc": 0.5,
                "member_best_auc_mean": 0.5, "monosem": 0.0,
                "best_meta": -1, "cluster_mean_auc_meta": 0.5}
    Z_members = Z[:, member_ids]
    A = auc_matrix(Z_members, Y)        # (n_members, n_gt)
    if A.size == 0:
        return {"members": len(member_ids), "best_gt": -1,
                "cluster_mean_auc": 0.5, "member_best_auc_mean": 0.5,
                "monosem": 0.0, "best_meta": -1,
                "cluster_mean_auc_meta": 0.5}
    cluster_mean = A.mean(axis=0)        # (n_gt,)
    best_gt = int(cluster_mean.argmax())
    cluster_mean_auc = float(cluster_mean[best_gt])
    member_best = A.max(axis=1)
    member_argmax = A.argmax(axis=1)
    monosem = float((member_argmax == best_gt).mean())
    # Best META: restrict the argmax to meta-axis label indices.
    if meta_idx:
        best_meta = int(max(meta_idx, key=lambda j: cluster_mean[j]))
        best_meta_auc = float(cluster_mean[best_meta])
    else:
        best_meta = -1
        best_meta_auc = 0.5
    return {
        "members":               len(member_ids),
        "member_ids":            [int(i) for i in member_ids],
        "best_gt":               best_gt,
        "cluster_mean_auc":      cluster_mean_auc,
        "member_best_auc_mean":  float(member_best.mean()),
        "monosem":               monosem,
        "best_meta":             best_meta,
        "cluster_mean_auc_meta": best_meta_auc,
    }


def _run_one_threshold(dictionary, sae, feed, decoder_vectors: np.ndarray,
                       gt_vocab: list[str], coherence_threshold: float,
                       max_features_per_expert: int | None) -> dict:
    from polygram.experts import cluster_experts
    from smsae.sae.evaluation import build_gt_matrix

    Y = build_gt_matrix(feed)             # (N, M)
    with torch.no_grad():
        Z = sae(feed.X).z.detach().cpu().numpy().astype(np.float32)
        # (N, F_full)

    # Meta labels: anything that isn't a per-particle or per-flavor
    # identity. Those are the bio-sae-style "cluster captures
    # higher-level structure" axes (kind, origin, is_*, color, etc.).
    meta_idx = [j for j, l in enumerate(gt_vocab)
                if not l.startswith("particle:")
                and not l.startswith("flavor:")]

    ed = cluster_experts(
        dictionary, decoder_vectors,
        method="cosine",
        coherence_threshold=coherence_threshold,
        max_features_per_expert=max_features_per_expert,
    )

    clusters: list[dict] = []
    for ei, expert in enumerate(ed.experts):
        member_ids = [_feature_id_from_name(f.name) for f in expert.features]
        s = _score_cluster_against_gt(member_ids, Z, Y, meta_idx)
        s["cluster_id"] = ei
        s["best_gt_label"] = (gt_vocab[s["best_gt"]]
                              if s["best_gt"] >= 0 else None)
        s["best_meta_label"] = (gt_vocab[s["best_meta"]]
                                if s["best_meta"] >= 0 else None)
        clusters.append(s)

    sizes = [c["members"] for c in clusters]
    mean_aucs = [c["cluster_mean_auc"] for c in clusters]
    n_pure_090 = sum(1 for c in clusters if c["cluster_mean_auc"] >= 0.90)
    n_pure_095 = sum(1 for c in clusters if c["cluster_mean_auc"] >= 0.95)
    # GT coverage: how many distinct GT features are claimed by a cluster
    # with cluster_mean_auc >= threshold?
    covered_090 = {c["best_gt"] for c in clusters
                   if c["cluster_mean_auc"] >= 0.90}
    covered_095 = {c["best_gt"] for c in clusters
                   if c["cluster_mean_auc"] >= 0.95}
    # Multi-member META coherence: of multi-member clusters, how many
    # find a meta-axis label that the whole cluster fires on at >= 0.80?
    # This is the bio-sae-style "cluster recovered structure no single
    # particle label encodes" number.
    multi = [c for c in clusters if c["members"] > 1]
    n_multi_meta_080 = sum(1 for c in multi
                            if c["cluster_mean_auc_meta"] >= 0.80)
    n_multi_meta_070 = sum(1 for c in multi
                            if c["cluster_mean_auc_meta"] >= 0.70)

    return {
        "coherence_threshold":      coherence_threshold,
        "max_features_per_expert":  max_features_per_expert,
        "n_experts":                ed.n_experts,
        "n_features":               ed.n_features,
        "n_singletons":             sum(1 for s in sizes if s == 1),
        "n_multi_member":           sum(1 for s in sizes if s > 1),
        "cluster_size_max":         max(sizes),
        "cluster_size_mean":        float(np.mean(sizes)),
        "cluster_mean_auc_mean":    float(np.mean(mean_aucs)),
        "cluster_mean_auc_max":     float(np.max(mean_aucs)),
        "n_clusters_pure_at_0.90":  n_pure_090,
        "n_clusters_pure_at_0.95":  n_pure_095,
        "n_gt_covered_at_0.90":     len(covered_090),
        "n_gt_covered_at_0.95":     len(covered_095),
        "n_multi_meta_at_0.80":     n_multi_meta_080,
        "n_multi_meta_at_0.70":     n_multi_meta_070,
        "n_gt_total":               int(Y.shape[1]),
        "clusters":                 clusters,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default="cascade__jumprelu",
                    dest="run_id",
                    help="SAE checkpoint to use; reads runs/<run_id>.pt")
    ap.add_argument("--encoding", default="rung5",
                    choices=["mps_rung1", "rung3", "rung4", "rung5"],
                    help="polygram encoding for the source Dictionary. "
                         "Defaults to rung5 (cap=128) so the cascade SAE's "
                         "full feature set goes into cluster_experts.")
    ap.add_argument("--thresholds", nargs="*", type=float,
                    default=[0.2, 0.3, 0.4, 0.5],
                    help="coherence_threshold values to sweep "
                         "(forwarded to cluster_experts).")
    ap.add_argument("--max-per-expert", type=int, default=None,
                    dest="max_per_expert",
                    help="optional max_features_per_expert cap.")
    ap.add_argument("--n-cascade-events", type=int, default=2000,
                    dest="n_cascade_events")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "runs",
                                                   "cluster_experts"))
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.run_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"cluster_experts_demo: run_id={args.run_id} encoding={args.encoding}")

    from smsae.sae.evaluation import load_sae
    from smsae.sae.data import feed_cascade, feed_embedded, feed_raw
    from scripts.forge_pipeline import build_records, build_dictionary, convert_to_safetensors

    feed_name = args.run_id.split("__")[0]
    builders = {
        "raw":      feed_raw,
        "embedded": lambda: feed_embedded(embed_dim=16, seed=0),
        "cascade":  lambda: feed_cascade(n_events=args.n_cascade_events,
                                          seed=0),
    }
    if feed_name not in builders:
        raise SystemExit(f"unsupported feed inferred from run_id: {feed_name}")
    feed = builders[feed_name]()

    sae, _meta = load_sae(os.path.join(REPO_ROOT, "runs", f"{args.run_id}.pt"))
    print(f"  loaded SAE: input_dim={sae.input_dim} n_features={sae.n_features}")

    # Re-use the existing safetensors→records→Dictionary plumbing so
    # cluster_experts sees exactly the same Dictionary the forge pipeline
    # would have constructed.
    st_path = os.path.join(out_dir, "sae.safetensors")
    convert_to_safetensors(sae, st_path)
    records = build_records(st_path)
    dictionary, _, _ = build_dictionary(
        records, args.encoding, selector="firing_rate",
        sae=sae, feed=feed, n_amp_qubits=4,
    )
    print(f"  built dictionary: {len(dictionary.features)} features "
          f"({args.encoding})")

    decoder_vectors = _decoder_vectors_for_dictionary(sae, dictionary)
    print(f"  decoder vectors: shape={decoder_vectors.shape}")

    runs = []
    print(f"\n  sweeping coherence_threshold ∈ {args.thresholds}")
    print(f"  {'thresh':>7}  {'k':>4}  {'kmulti':>6}  {'sizeMx':>6}  "
          f"{'µAUC':>5}  {'GT≥.95':>7}  "
          f"{'multi-meta≥.80':>15}  {'multi-meta≥.70':>15}")
    for t in args.thresholds:
        r = _run_one_threshold(dictionary, sae, feed, decoder_vectors,
                               feed.feature_vocab, t, args.max_per_expert)
        runs.append(r)
        print(f"  {t:>7.2f}  {r['n_experts']:>4}  "
              f"{r['n_multi_member']:>6}  {r['cluster_size_max']:>6}  "
              f"{r['cluster_mean_auc_mean']:>5.3f}  "
              f"{r['n_gt_covered_at_0.95']:>7}  "
              f"{r['n_multi_meta_at_0.80']:>15}  "
              f"{r['n_multi_meta_at_0.70']:>15}")

    runs.sort(key=lambda r: r["coherence_threshold"])
    # Pick the threshold that gives the strongest multi-member meta
    # signal — that's the bio-sae validation question.
    best = max(runs, key=lambda r: (r["n_multi_meta_at_0.80"],
                                     r["n_multi_meta_at_0.70"]))
    print(f"\n  top multi-member clusters at threshold="
          f"{best['coherence_threshold']} (ranked by best-META cluster-µAUC):")
    print(f"    {'id':>3}  {'n':>2}  {'best META':<24}  {'metaµAUC':>8}  "
          f"{'best GT':<24}  {'µAUC':>5}  monosem")
    multi = sorted([c for c in best["clusters"] if c["members"] > 1],
                   key=lambda c: -c["cluster_mean_auc_meta"])
    for c in multi[:12]:
        print(f"    {c['cluster_id']:>3}  {c['members']:>2}  "
              f"{str(c['best_meta_label'])[:24]:<24}  "
              f"{c['cluster_mean_auc_meta']:>8.3f}  "
              f"{str(c['best_gt_label'])[:24]:<24}  "
              f"{c['cluster_mean_auc']:>5.3f}  "
              f"{c['monosem']:.2f}")

    payload = {
        "run_id":     args.run_id,
        "encoding":   args.encoding,
        "n_features_full":       int(sae.n_features),
        "n_features_dictionary": len(dictionary.features),
        "n_gt_features":         int(len(feed.feature_vocab)),
        "thresholds":            args.thresholds,
        "runs":                  runs,
        "best_threshold":        best["coherence_threshold"],
    }
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
