"""Phase A.3 runner: train all 3 SAE variants on all 3 feeds.

Saves each trained SAE to runs/{feed}__{variant}.pt under the repo root,
plus a runs/summary.json.

Usage:
    python scripts/train_all.py                 # auto: cuda if available, else cpu
    python scripts/train_all.py --device cpu     # force cpu
    python scripts/train_all.py --device cuda    # force cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make repo root importable and force CWD to repo root so relative paths land in repo/runs/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch

from smsae.sae.data import Feed, feed_cascade, feed_embedded, feed_raw
from smsae.sae.models import make_sae
from smsae.sae.train import TrainConfig, train


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device string. 'auto' -> cuda if available, else cpu."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    return requested


FEED_CONFIGS = {
    "raw": dict(
        n_features=32, epochs=2000, batch_size=32,
        topk_k=6, l1_coeff=1e-2, l0_coeff=3e-3, init_theta=0.05,
    ),
    "embedded": dict(
        n_features=64, epochs=3000, batch_size=32,
        topk_k=6, l1_coeff=8e-3, l0_coeff=3e-3, init_theta=0.05,
    ),
    "cascade": dict(
        n_features=128, epochs=200, batch_size=256,
        topk_k=12, l1_coeff=5e-3, l0_coeff=2e-3, init_theta=0.05,
    ),
}


def build_feed(name: str) -> Feed:
    if name == "raw":      return feed_raw()
    if name == "embedded": return feed_embedded(embed_dim=16, seed=0)
    if name == "cascade":  return feed_cascade(n_events=2000, seed=0)
    raise ValueError(name)


def build_sae(variant: str, input_dim: int, feed_name: str):
    cfg = FEED_CONFIGS[feed_name]
    n_features = cfg["n_features"]
    if variant == "topk":
        return make_sae("topk", input_dim, n_features, k=cfg["topk_k"])
    if variant == "l1":
        return make_sae("l1", input_dim, n_features, l1_coeff=cfg["l1_coeff"])
    if variant == "jumprelu":
        return make_sae("jumprelu", input_dim, n_features,
                        l0_coeff=cfg["l0_coeff"], init_theta=cfg["init_theta"])
    raise ValueError(variant)


def main():
    parser = argparse.ArgumentParser(description="Train 3 SAE variants x 3 feeds.")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto = cuda if available, else cpu).",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    torch.manual_seed(0)
    rows = []
    summary = {}

    for feed_name in ("raw", "embedded", "cascade"):
        print("=" * 78)
        print(f"FEED: {feed_name}")
        print("=" * 78)
        feed = build_feed(feed_name)
        print(f"  X: {tuple(feed.X.shape)}    vocab: {len(feed.feature_vocab)} GT features")

        fcfg = FEED_CONFIGS[feed_name]
        for variant in ("topk", "l1", "jumprelu"):
            print(f"\n--- {variant} on {feed_name} ---")
            torch.manual_seed(0)
            sae = build_sae(variant, feed.D, feed_name)
            tcfg = TrainConfig(
                epochs=fcfg["epochs"], batch_size=fcfg["batch_size"],
                lr=1e-3, warmup_steps=50,
                resample_every=max(100, fcfg["epochs"] // 5),
                log_every=max(50, fcfg["epochs"] // 5),
                device=device,
            )
            t0 = time.time()
            hist = train(sae, feed, tcfg, verbose=False)
            elapsed = time.time() - t0

            with torch.no_grad():
                X_dev = feed.X.to(device)
                out = sae(X_dev)
                final_recon = float(out.recon_loss)
                final_l0 = float((out.z.abs() > 1e-9).float().sum(dim=-1).mean())
                dead = float((sae.steps_dead >= tcfg.resample_threshold).float().mean())
                var_total = float(X_dev.var())
                var_resid = float((X_dev - out.x_hat).var())
                ve = 1.0 - var_resid / max(var_total, 1e-12)

            # Save checkpoints on CPU so they load anywhere regardless of train device.
            sae = sae.to("cpu")

            run_id = f"{feed_name}__{variant}"
            ckpt_path = os.path.join(RUNS_DIR, f"{run_id}.pt")
            torch.save({
                "state_dict": sae.state_dict(),
                "kind": variant,
                "feed_name": feed_name,
                "input_dim": sae.input_dim,
                "n_features": sae.n_features,
            }, ckpt_path)

            print(f"  recon={final_recon:.4f}  L0={final_l0:.2f}  "
                  f"dead={dead:.2%}  var_explained={ve:.4f}  "
                  f"resamples={len(hist.resamples)}  time={elapsed:.1f}s")
            print(f"  saved to runs/{run_id}.pt")

            rows.append({
                "feed": feed_name, "variant": variant, "run_id": run_id,
                "input_dim": sae.input_dim, "n_features": sae.n_features,
                "recon_loss": final_recon, "l0": final_l0,
                "dead_fraction": dead, "var_explained": ve,
                "resamples_count": len(hist.resamples),
                "n_train_steps": hist.step[-1] if hist.step else 0,
                "wall_time_s": elapsed,
            })
            summary[run_id] = rows[-1]

    print("\n" + "=" * 78)
    print("SUMMARY: 3 variants x 3 feeds")
    print("=" * 78)
    print(f"{'feed':<10s} {'variant':<10s} {'recon':>8s} {'L0':>6s} "
          f"{'dead':>6s} {'var_expl':>9s} {'time(s)':>8s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['feed']:<10s} {r['variant']:<10s} {r['recon_loss']:>8.4f} "
              f"{r['l0']:>6.2f} {r['dead_fraction']:>6.1%} {r['var_explained']:>9.4f} "
              f"{r['wall_time_s']:>8.1f}")

    with open(os.path.join(RUNS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote runs/summary.json")


if __name__ == "__main__":
    main()
