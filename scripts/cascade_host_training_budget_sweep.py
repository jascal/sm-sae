"""Training-budget sweep at fixed L6 cascade host.

Closes the gate-7.3 saga's converged hypothesis from PR #28: the
cascade rollout is information-rich (state_t-direct AUC 0.92 mean,
0% < 0.7), but the LM at the previous sweep's training budget (500
steps × 5 epochs × 2000 trajectories) drops 0.09-0.16 absolute AUC
on the per-particle features. The host's *training regime*, not its
*capacity*, is the binding constraint.

This sweep tests that directly: fix the host at n_embd=61 / n_layer=6
(the empirical depth peak from PR #26) and scale the training budget
along {n_trajectories, epochs}. Measure forge gate 7.3 + probe at
each cell. Predicted: color:r LM-probe AUC lifts from 0.74 toward
0.90 (the state_t ceiling) as gradient steps scale.

Acceptance: any cell achieves `forge_delta_vs_random >= +0.05` on
`cascade__jumprelu` rung5 — closing gate 7.3 positively after 28
PRs of iterative diagnosis.

Run:
    python scripts/cascade_host_training_budget_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# Fixed host (the empirical depth peak from PR #26).
HOST_N_EMBD = 61
HOST_N_LAYER = 6

# 4-cell budget grid. (n_trajectories, epochs). Each cell's
# n_total_steps ≈ ceil(n_pairs / batch_size) * epochs, where
# n_pairs ≈ 1.5 × n_trajectories.
BUDGET_GRID: list[tuple[int, int]] = [
    (2000, 5),    # PR #26 baseline reference (~500 steps)
    (5000, 10),   # ~5x compute (~1562 steps)
    (10000, 20),  # ~25x compute (~6250 steps)
]

SPOTLIGHT_FEATURES = [
    "color:r", "color:b", "color:g",
    "flavor:u", "flavor:d", "flavor:mu",
    "particle:mu+", "particle:u_b",
]


def _train_one(n_trajectories: int, epochs: int, out_dir: Path,
               seed: int = 0) -> dict:
    """Train L6 host at the given budget. No aux supervision."""
    from scripts.train_cascade_host import train as train_cascade_host
    t0 = time.time()
    meta = train_cascade_host(
        n_embd=HOST_N_EMBD, n_layer=HOST_N_LAYER,
        n_trajectories=n_trajectories, epochs=epochs,
        seed=seed, out=str(out_dir),
        aux_supervision="off",
    )
    meta["train_wall_s"] = round(time.time() - t0, 1)
    return meta


def _forge_one(host_dir: Path, run_out: Path) -> dict:
    """forge_pipeline.py cascade__jumprelu rung5 with --host-dir."""
    t0 = time.time()
    result = subprocess.run(
        ["python", "scripts/forge_pipeline.py", "cascade__jumprelu",
         "--encoding", "rung5",
         "--out", str(run_out),
         "--host-dir", str(host_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    wall = round(time.time() - t0, 1)
    if result.returncode != 0:
        return {"status": "failed",
                "stderr": result.stderr[-400:],
                "wall_s": wall}
    forge_json = run_out / "cascade__jumprelu" / "forge_results.json"
    if not forge_json.exists():
        return {"status": "failed",
                "reason": "forge_results.json not written",
                "wall_s": wall}
    with open(forge_json) as f:
        d = json.load(f)
    return {
        "status":      "ok",
        "forge_score": float(d["forge_score"]),
        "wall_s":      wall,
    }


def _probe_one(host_dir: Path, run_out: Path,
               n_trajectories: int = 5000) -> dict:
    """probe_full_gt_recoverability.py against the host."""
    t0 = time.time()
    out_path = run_out / "probe.json"
    probe_host_path = host_dir / "host"
    result = subprocess.run(
        ["python", "scripts/probe_full_gt_recoverability.py",
         "--baseline-host", str(probe_host_path),
         "--n-trajectories", str(n_trajectories),
         "--out", str(out_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    wall = round(time.time() - t0, 1)
    if result.returncode != 0 or not out_path.exists():
        return {"status": "failed",
                "stderr": result.stderr[-400:],
                "wall_s": wall}
    with open(out_path) as f:
        d = json.load(f)
    residual = d["baseline_aucs_residual"]
    measured = [v for v in residual.values() if v is not None]
    spotlight_aucs = {n: residual.get(n) for n in SPOTLIGHT_FEATURES}
    measured_spot = [v for v in spotlight_aucs.values() if v is not None]
    return {
        "status":                  "ok",
        "mean_residual_auc":       round(sum(measured) / max(1, len(measured)), 4),
        "median_residual_auc":     round(sorted(measured)[len(measured) // 2], 4) if measured else None,
        "pct_residual_ge_0.9":     round(sum(1 for v in measured if v >= 0.9) / max(1, len(measured)), 4),
        "spotlight_aucs":          spotlight_aucs,
        "spotlight_median_auc":    round(sorted(measured_spot)[len(measured_spot) // 2], 4) if measured_spot else None,
        "color_r_auc":             spotlight_aucs.get("color:r"),
        "color_g_auc":             spotlight_aucs.get("color:g"),
        "color_b_auc":             spotlight_aucs.get("color:b"),
        "wall_s":                  wall,
    }


def _measure_random_baseline(run_out: Path) -> dict:
    print("[budget-sweep] measuring random-init forge baseline...")
    result = _forge_one(Path("/tmp/__nonexistent_random_baseline__"),
                       run_out / "random_baseline")
    if result.get("status") == "ok":
        print(f"[budget-sweep]   random forge_score: {result['forge_score']:.4f}")
    return result


def run_sweep(out_dir: Path, configs: list[tuple[int, int]],
              probe_n_trajectories: int = 5000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_root = Path(REPO_ROOT) / "runs" / "cascade_host"

    random_baseline = _measure_random_baseline(out_dir)
    random_score = random_baseline.get("forge_score", 0.0)

    rows: list[dict] = []
    grid_t0 = time.time()
    for i, (n_traj, epochs) in enumerate(configs, start=1):
        config_name = f"budget_T{n_traj}_E{epochs}"
        host_dir = sweep_root / config_name
        print(f"\n[budget-sweep] [{i}/{len(configs)}] config "
              f"n_trajectories={n_traj}, epochs={epochs}")

        # Train
        train_meta = _train_one(n_traj, epochs, host_dir)
        n_steps = int(train_meta.get("n_train_steps", 0))
        print(f"[budget-sweep]   trained {n_steps} steps in "
              f"{train_meta['train_wall_s']}s "
              f"(final_loss={train_meta['train_loss_final']:.3f})")

        # Forge
        forge_out = out_dir / f"forge_{config_name}"
        forge_result = _forge_one(host_dir, forge_out)
        if forge_result.get("status") != "ok":
            print(f"[budget-sweep]   FORGE FAILED: {forge_result}")
        else:
            delta = forge_result["forge_score"] - random_score
            print(f"[budget-sweep]   forge_score={forge_result['forge_score']:.4f}  "
                  f"Δ_random={delta:+.4f}  ({forge_result['wall_s']}s)")

        # Probe
        probe_out = out_dir / f"probe_{config_name}"
        probe_out.mkdir(parents=True, exist_ok=True)
        probe_result = _probe_one(host_dir, probe_out,
                                  n_trajectories=probe_n_trajectories)
        if probe_result.get("status") != "ok":
            print(f"[budget-sweep]   PROBE FAILED: {probe_result}")
        else:
            cr = probe_result.get("color_r_auc")
            cr_str = f"{cr:.3f}" if cr is not None else "—"
            print(f"[budget-sweep]   probe: mean={probe_result['mean_residual_auc']:.3f}  "
                  f"spotlight_med={probe_result['spotlight_median_auc']:.3f}  "
                  f"color:r={cr_str}  ({probe_result['wall_s']}s)")

        rows.append({
            "n_trajectories":       n_traj,
            "epochs":               epochs,
            "n_train_steps":        n_steps,
            "n_params":             train_meta.get("n_params"),
            "train_loss_final":     round(float(train_meta.get("train_loss_final", 0.0)), 4),
            "train_wall_s":         train_meta.get("train_wall_s"),
            "forge_score":          forge_result.get("forge_score") if forge_result.get("status") == "ok" else None,
            "forge_delta_random":   (forge_result.get("forge_score", 0.0) - random_score) if forge_result.get("status") == "ok" else None,
            "probe_mean_auc":       probe_result.get("mean_residual_auc"),
            "probe_spotlight_med":  probe_result.get("spotlight_median_auc"),
            "probe_color_r":        probe_result.get("color_r_auc"),
            "probe_color_g":        probe_result.get("color_g_auc"),
            "probe_color_b":        probe_result.get("color_b_auc"),
        })

    total_wall = round(time.time() - grid_t0, 1)
    print(f"\n[budget-sweep] total wall: {total_wall}s ({total_wall/60:.1f} min)")

    # Gate verdict
    valid = [r for r in rows if r["forge_delta_random"] is not None]
    hits = [r for r in valid if r["forge_delta_random"] >= 0.05]

    summary = {
        "host":              {"n_embd": HOST_N_EMBD, "n_layer": HOST_N_LAYER},
        "configs":           configs,
        "random_baseline":   random_baseline,
        "rows":              rows,
        "total_wall_s":      total_wall,
        "gate_7_3_closed":   len(hits) > 0,
        "hits":              [(r["n_trajectories"], r["epochs"], r["forge_delta_random"])
                              for r in hits],
        "best_delta":        max((r["forge_delta_random"] for r in valid), default=None),
        "best_color_r":      max((r["probe_color_r"] for r in rows if r["probe_color_r"] is not None), default=None),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if rows:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    print()
    print("=" * 72)
    print("GATE VERDICT")
    print("=" * 72)
    print(f"  Total wall: {total_wall}s ({total_wall/60:.1f} min)")
    best_delta_str = (
        f"{summary['best_delta']:+.4f}" if summary['best_delta'] is not None else "N/A"
    )
    if summary["gate_7_3_closed"]:
        print(f"  ✅ GATE 7.3 CLOSED — best Δ_random = {best_delta_str}, "
              f"threshold +0.05, {len(hits)}/{len(valid)} cells hit")
    else:
        print(f"  ❌ Gate 7.3 still open. Best Δ_random = {best_delta_str}, "
              f"threshold +0.05, 0/{len(valid)} cells hit")
    best_cr = summary["best_color_r"]
    best_cr_str = f"{best_cr:.3f}" if best_cr is not None else "N/A"
    print(f"  Best color:r LM-probe AUC: {best_cr_str} (state_t ceiling: 0.904)")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/budget_sweep", type=Path)
    ap.add_argument("--probe-n-trajectories", type=int, default=5000)
    ap.add_argument("--smoke", action="store_true",
                    help="run only the smallest (2000, 5) baseline cell as a "
                         "smoke test.")
    args = ap.parse_args()

    configs = BUDGET_GRID
    if args.smoke:
        configs = [(2000, 5)]
        print("[budget-sweep] --smoke: baseline cell only")

    out_dir = Path(args.out) if args.out.is_absolute() else (Path(REPO_ROOT) / args.out)
    summary = run_sweep(out_dir, configs=configs,
                        probe_n_trajectories=args.probe_n_trajectories)
    return 0 if summary["gate_7_3_closed"] else 1


if __name__ == "__main__":
    sys.exit(main())
