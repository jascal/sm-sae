"""Cascade-host capacity sweep — gate-7.3-by-capacity falsifiable test.

Trains a small grid of cascade hosts at varying ``(n_embd, n_layer)``,
then for each config measures both gate 7.3 (forge faithfulness on
``cascade__jumprelu`` rung5) and the per-feature GT probe. Surfaces
whether host capacity scales away the gate-7.3 gap that supervised aux
training (v1/v2) failed to close.

See ``openspec/changes/investigate-cascade-host-capacity-sweep/proposal.md``
for the full hypothesis + gates.

Run:
    python scripts/cascade_host_capacity_sweep.py
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


# Capacity grid. NB: the cascade SAE (`cascade__jumprelu`) has
# input_dim=61, so the forge measurement (gate C.2) is only valid for
# n_embd=61 hosts. Configs with n_embd > 61 still train + get probed
# (the probe runs on pooled hidden state at any n_embd) but their
# forge row is left unmeasured — they inform gate C.3 (per-particle
# AUC scaling) but not gate C.2 directly.
#
# n_layer is the within-n_embd=61 capacity dimension that IS forge-
# measurable. So the grid is:
#   - Forge + probe: vary n_layer in {2, 4, 6} at fixed n_embd=61.
#   - Probe only: vary n_embd in {96, 128, 192} at n_layer=2.
SWEEP_CONFIGS: list[tuple[int, int]] = [
    # Forge-measurable (n_embd == cascade SAE input_dim):
    (61, 2),    # baseline / reference
    (61, 4),
    (61, 6),
    # Probe-only (n_embd > 61):
    (96, 2),
    (128, 2),
    (192, 2),
]


# Depth-only grid for cascade-host-depth-sweep. Direct follow-up to
# the capacity sweep's finding that depth at fixed n_embd=61 is the
# binding axis for gate 7.3. Extrapolating PR #25's L4 -> L6 rate,
# L10 is the predicted transition point for Δ_random >= +0.05.
DEPTH_GRID: list[tuple[int, int]] = [
    (61, 6),    # re-baseline at the highest depth from PR #25
    (61, 8),
    (61, 10),
    (61, 12),
]

# Spotlight features for gate C.3 — the weakest features in
# PRs #22/#23 that the capacity hypothesis predicts will lift by
# capacity alone.
SPOTLIGHT_FEATURES: list[str] = [
    "color:r", "color:b", "color:g",
    "flavor:u", "flavor:d", "flavor:mu",
    "particle:mu+", "particle:u_b",
]


def _train_one(n_embd: int, n_layer: int, out_dir: Path,
               n_trajectories: int = 2000, epochs: int = 5,
               seed: int = 0) -> dict:
    """Train a single cascade host. No aux supervision (the load-bearing
    hypothesis of the sweep)."""
    from scripts.train_cascade_host import train as train_cascade_host
    t0 = time.time()
    meta = train_cascade_host(
        n_embd=n_embd, n_layer=n_layer,
        n_trajectories=n_trajectories, epochs=epochs,
        seed=seed, out=str(out_dir),
        aux_supervision="off",
    )
    meta["train_wall_s"] = round(time.time() - t0, 1)
    return meta


def _forge_one(host_dir: Path, run_out: Path) -> dict:
    """Run forge_pipeline.py on the cascade__jumprelu cell with the
    given host. Returns the parsed forge_results.json plus wall time."""
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
    def _safe_float(v):
        return float(v) if isinstance(v, (int, float)) else None
    return {
        "status":              "ok",
        "forge_score":         float(d["forge_score"]),
        "baseline_score":      _safe_float(d.get("baseline_score")),
        "post_compress_score": _safe_float(d.get("post_compress_score")),
        "host_kind":           d.get("forge", {}).get("host", {}).get("kind"),
        "wall_s":              wall,
    }


def _probe_one(host_dir: Path, run_out: Path,
               n_trajectories: int = 5000) -> dict:
    """Run probe_full_gt_recoverability.py against the given host.
    Returns {feature_name: auc} + spotlight subsummary."""
    t0 = time.time()
    out_path = run_out / "probe.json"
    # The trained host's safetensors live under `<host_dir>/host/` (the
    # train_cascade_host save convention). The probe script expects the
    # inner directory; forge_pipeline does the join itself.
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
    spotlight_aucs = {
        name: residual.get(name) for name in SPOTLIGHT_FEATURES
    }
    measured_spotlight = [v for v in spotlight_aucs.values() if v is not None]
    return {
        "status":                 "ok",
        "n_features_measured":    len(measured),
        "mean_residual_auc":      round(sum(measured) / max(1, len(measured)), 4),
        "median_residual_auc":    round(sorted(measured)[len(measured) // 2], 4) if measured else None,
        "pct_residual_ge_0.9":    round(sum(1 for v in measured if v >= 0.9) / max(1, len(measured)), 4),
        "pct_residual_ge_0.92":   round(sum(1 for v in measured if v >= 0.92) / max(1, len(measured)), 4),
        "spotlight_aucs":         spotlight_aucs,
        "spotlight_median_auc":   round(sorted(measured_spotlight)[len(measured_spotlight) // 2], 4) if measured_spotlight else None,
        "spotlight_n_ge_0.92":    sum(1 for v in measured_spotlight if v >= 0.92),
        "wall_s":                 wall,
    }


def _measure_random_baseline(run_out: Path) -> dict:
    """Forge measurement with NO host present — the random-init fallback
    score. Run once at the start; gate C.2 compares per-config forge
    scores against this number."""
    print("[sweep] measuring random-init baseline forge_score (no host)...")
    # Use a non-existent host dir so forge_pipeline falls back to random init.
    random_host_dir = "/tmp/__nonexistent_for_random_baseline__"
    result = _forge_one(Path(random_host_dir), run_out / "random_baseline")
    if result.get("status") == "ok":
        print(f"[sweep]   random-init forge_score: {result['forge_score']:.4f} "
              f"(wall {result['wall_s']}s)")
    return result


def _config_dir_name(n_embd: int, n_layer: int) -> str:
    return f"sweep_NE{n_embd}_L{n_layer}"


def run_sweep(out_dir: Path, configs: list[tuple[int, int]],
              n_trajectories: int = 2000, epochs: int = 5,
              probe_n_trajectories: int = 5000) -> dict:
    """Run train -> forge -> probe across the grid. Returns the summary
    payload (also written to out_dir / summary.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_root = Path(REPO_ROOT) / "runs" / "cascade_host"

    random_baseline = _measure_random_baseline(out_dir)

    rows: list[dict] = []
    grid_t0 = time.time()
    for i, (n_embd, n_layer) in enumerate(configs, start=1):
        config_name = _config_dir_name(n_embd, n_layer)
        host_dir = sweep_root / config_name
        print(f"\n[sweep] [{i}/{len(configs)}] config {config_name}")

        # Train
        train_meta = _train_one(n_embd, n_layer, host_dir,
                                n_trajectories=n_trajectories, epochs=epochs)
        print(f"[sweep]   trained in {train_meta['train_wall_s']}s "
              f"(final_loss={train_meta['train_loss_final']:.3f}, "
              f"n_params={train_meta['n_params']})")

        # Forge — only run when host n_embd matches the SAE's input_dim
        # (cascade__jumprelu fixed at 61). Probe-only configs (n_embd > 61)
        # report `forge_status="skipped_dim_mismatch"`.
        sae_input_dim = 61
        if n_embd != sae_input_dim:
            print(f"[sweep]   skipping forge (host n_embd={n_embd} != "
                  f"sae input_dim={sae_input_dim}; cascade SAE can't ingest)")
            forge_result = {
                "status": "skipped_dim_mismatch",
                "forge_score": None,
                "wall_s": 0.0,
            }
        else:
            forge_out = out_dir / f"forge_{config_name}"
            forge_result = _forge_one(host_dir, forge_out)
            if forge_result.get("status") != "ok":
                print(f"[sweep]   FORGE FAILED: {forge_result}")
            else:
                forge_score = forge_result["forge_score"]
                random_score = random_baseline.get("forge_score", 0.0)
                delta = forge_score - random_score
                print(f"[sweep]   forge_score={forge_score:.4f}  Δ_vs_random={delta:+.4f}  "
                      f"({forge_result['wall_s']}s)")

        # Probe — runs on the pooled hidden state at any n_embd.
        # Use a per-config probe output dir so forge-skipped configs
        # still have somewhere to land the probe JSON.
        probe_out = out_dir / f"probe_{config_name}"
        probe_out.mkdir(parents=True, exist_ok=True)
        probe_result = _probe_one(host_dir, probe_out,
                                  n_trajectories=probe_n_trajectories)
        if probe_result.get("status") != "ok":
            print(f"[sweep]   PROBE FAILED: {probe_result}")
        else:
            print(f"[sweep]   probe: mean_auc={probe_result['mean_residual_auc']:.3f}  "
                  f"pct≥0.92={probe_result['pct_residual_ge_0.92']:.0%}  "
                  f"spotlight_median={probe_result['spotlight_median_auc']:.3f}  "
                  f"({probe_result['wall_s']}s)")

        rows.append({
            "n_embd":               n_embd,
            "n_layer":              n_layer,
            "n_params":             train_meta.get("n_params"),
            "train_loss_final":     round(float(train_meta.get("train_loss_final", 0.0)), 4),
            "train_wall_s":         train_meta.get("train_wall_s"),
            "forge_score":          forge_result.get("forge_score") if forge_result.get("status") == "ok" else None,
            "forge_delta_vs_random": (
                forge_result.get("forge_score", 0.0) - random_baseline.get("forge_score", 0.0)
                if forge_result.get("status") == "ok"
                and random_baseline.get("status") == "ok"
                else None
            ),
            "forge_wall_s":         forge_result.get("wall_s"),
            "probe_mean_residual_auc":  probe_result.get("mean_residual_auc"),
            "probe_median_residual_auc": probe_result.get("median_residual_auc"),
            "probe_pct_residual_ge_0.9": probe_result.get("pct_residual_ge_0.9"),
            "probe_pct_residual_ge_0.92": probe_result.get("pct_residual_ge_0.92"),
            "probe_spotlight_median_auc": probe_result.get("spotlight_median_auc"),
            "probe_spotlight_n_ge_0.92":  probe_result.get("spotlight_n_ge_0.92"),
            "probe_n_features_measured":  probe_result.get("n_features_measured"),
            "probe_color_r_auc":    (probe_result.get("spotlight_aucs") or {}).get("color:r"),
            "probe_color_b_auc":    (probe_result.get("spotlight_aucs") or {}).get("color:b"),
            "probe_color_g_auc":    (probe_result.get("spotlight_aucs") or {}).get("color:g"),
            "probe_flavor_mu_auc":  (probe_result.get("spotlight_aucs") or {}).get("flavor:mu"),
            "probe_wall_s":         probe_result.get("wall_s"),
        })

    total_wall = round(time.time() - grid_t0, 1)
    print(f"\n[sweep] total sweep wall: {total_wall}s ({total_wall / 60:.1f} min)")

    # Gate assessment
    forge_rows = [r for r in rows if r["forge_delta_vs_random"] is not None]
    c2_hits = [r for r in forge_rows if r["forge_delta_vs_random"] >= 0.05]
    # C.3 looks at the largest-capacity config that was PROBED — both
    # forge-measurable and probe-only configs count, since the probe runs
    # at any n_embd. Pick by (n_embd, n_layer) tuple ordering.
    probe_rows = [r for r in rows if r.get("probe_spotlight_median_auc") is not None]
    largest_probe = max(
        probe_rows, key=lambda r: (r["n_embd"], r["n_layer"])
    ) if probe_rows else None
    c3_hit = (
        largest_probe is not None
        and largest_probe["probe_spotlight_median_auc"] >= 0.92
    )

    # C.1 = the script finished AND every probe-only config got a probe
    #       row AND every forge-measurable config got a forge row.
    c1_pass = all(
        (r["forge_delta_vs_random"] is not None or r["n_embd"] != 61)
        and r["probe_mean_residual_auc"] is not None
        for r in rows
    )

    gate_summary = {
        "C.1_mechanical_pass":      c1_pass,
        "C.1_total_wall_s":         total_wall,
        "C.2_gate_7_3_by_capacity": {
            "passed":               len(c2_hits) > 0,
            "n_configs_meeting":    len(c2_hits),
            "n_forge_measurable_configs": len(forge_rows),
            "configs":              [(r["n_embd"], r["n_layer"], r["forge_delta_vs_random"])
                                     for r in c2_hits],
            "best_delta":           max((r["forge_delta_vs_random"] for r in forge_rows), default=None),
            "best_config":          (
                (max(forge_rows, key=lambda r: r["forge_delta_vs_random"])["n_embd"],
                 max(forge_rows, key=lambda r: r["forge_delta_vs_random"])["n_layer"])
                if forge_rows else None
            ),
        },
        "C.3_per_particle_scaling": {
            "passed":               c3_hit,
            "largest_config":       (largest_probe["n_embd"], largest_probe["n_layer"]) if largest_probe else None,
            "largest_spotlight_median_auc": largest_probe["probe_spotlight_median_auc"] if largest_probe else None,
        },
    }

    summary = {
        "configs_run":      [(n, l) for (n, l) in configs],
        "n_trajectories":   n_trajectories,
        "epochs":           epochs,
        "probe_n_trajectories": probe_n_trajectories,
        "random_baseline":  random_baseline,
        "rows":             rows,
        "gate_summary":     gate_summary,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # CSV roll-up
    if rows:
        csv_path = out_dir / "summary.csv"
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[sweep] wrote {csv_path}")
    print(f"[sweep] wrote {out_dir / 'summary.json'}")

    # Print gate verdict
    print()
    print("=" * 70)
    print("GATE VERDICT")
    print("=" * 70)
    c2 = gate_summary["C.2_gate_7_3_by_capacity"]
    c3 = gate_summary["C.3_per_particle_scaling"]
    print(f"  C.1 mechanical: {'PASS' if gate_summary['C.1_mechanical_pass'] else 'FAIL'} "
          f"(total wall {total_wall}s)")
    best_delta_str = f"{c2['best_delta']:+.4f}" if c2["best_delta"] is not None else "N/A"
    print(f"  C.2 gate 7.3 by capacity: {'PASS' if c2['passed'] else 'FAIL'} "
          f"(best Δ={best_delta_str}; threshold +0.05; "
          f"{c2['n_configs_meeting']}/{c2['n_forge_measurable_configs']} forge-measurable configs meet)")
    spotlight_auc = c3.get("largest_spotlight_median_auc")
    spotlight_str = f"{spotlight_auc:.3f}" if spotlight_auc is not None else "N/A"
    print(f"  C.3 per-particle scaling: {'PASS' if c3['passed'] else 'FAIL'} "
          f"(largest-config spotlight median AUC = {spotlight_str}; threshold 0.92)")

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/capacity_sweep", type=Path,
                    help="output directory for summary.json / summary.csv "
                         "and per-config probe/forge artefacts")
    ap.add_argument("--n-trajectories", type=int, default=2000,
                    help="trajectories per host training run")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--probe-n-trajectories", type=int, default=5000,
                    help="trajectories for the probe dataset (per config)")
    ap.add_argument("--smoke", action="store_true",
                    help="run only the smallest 2 configs (61x2, 96x2) as a "
                         "smoke test; useful for verifying the script wiring "
                         "without burning ~16 min on the full grid.")
    ap.add_argument("--grid", default="capacity",
                    choices=("capacity", "depth", "custom"),
                    help="config preset. 'capacity' (default) — the 6-config "
                         "(n_embd, n_layer) grid from PR #25. 'depth' — the "
                         "4-config follow-up at fixed n_embd=61, varying "
                         "n_layer in {6,8,10,12} (cascade-host-depth-sweep). "
                         "'custom' — supply --config NE_L flags.")
    ap.add_argument("--config", action="append", default=[],
                    help="repeatable NE_L specifier when --grid=custom "
                         "(e.g. --config 61_8 --config 96_4).")
    args = ap.parse_args()

    if args.grid == "depth":
        configs = DEPTH_GRID
        print(f"[sweep] --grid=depth: 4-config depth follow-up "
              f"{configs}")
    elif args.grid == "custom":
        if not args.config:
            ap.error("--grid=custom requires at least one --config NE_L")
        configs = []
        for spec in args.config:
            try:
                ne_str, l_str = spec.split("_")
                configs.append((int(ne_str), int(l_str)))
            except (ValueError, AttributeError):
                ap.error(f"--config must be 'NE_L' (e.g. 61_8); got {spec!r}")
        print(f"[sweep] --grid=custom: {configs}")
    else:
        configs = SWEEP_CONFIGS
    if args.smoke:
        # Smoke: one forge-measurable + one probe-only to verify both paths
        configs = [(61, 2), (96, 2)]
        print("[sweep] --smoke: limiting to (61,2) + (96,2) — exercises "
              "both forge-measurable and probe-only paths")

    summary = run_sweep(
        Path(args.out) if not args.out.is_absolute() else args.out,
        configs=configs,
        n_trajectories=args.n_trajectories,
        epochs=args.epochs,
        probe_n_trajectories=args.probe_n_trajectories,
    )
    return 0 if summary["gate_summary"]["C.1_mechanical_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
