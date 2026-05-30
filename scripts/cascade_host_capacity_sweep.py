"""Cascade-host capacity sweep — gate-7.3-by-capacity falsifiable test.

Trains a small grid of cascade hosts at varying ``(n_embd, n_layer)``,
then for each config measures both gate 7.3 (forge faithfulness on
``cascade__jumprelu`` rung5) and the per-feature GT probe. Surfaces
whether host capacity scales away the gate-7.3 gap that supervised aux
training (v1/v2) failed to close.

See ``openspec/changes/archive/investigate-cascade-host-capacity-sweep/proposal.md``
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

# Training-budget grid for cascade-host-training-budget-sweep. Direct
# follow-up to PR #28 (rollout entropy), which found the cascade
# vocabulary is information-rich (color:r sits at AUC 0.904 in state_t)
# but the L6 host only surfaces it at 0.740 — i.e. the LM training
# regime, not the architecture, is the binding constraint. This grid
# holds the architecture fixed at the L6 depth peak and scales only the
# training budget. Each entry carries its own n_trajectories/epochs;
# omitted knobs (lr) inherit the global defaults.
#
# Factorial design:
#   - rows 1->2->3 isolate gradient-step count (corpus fixed at 2000).
#   - row 4 vs row 2 isolates corpus size (same epochs, 2.5x data).
BUDGET_GRID: list[dict] = [
    {"n_embd": 61, "n_layer": 6, "n_trajectories": 2000, "epochs": 5},   # 1x baseline (== depth-sweep L6)
    {"n_embd": 61, "n_layer": 6, "n_trajectories": 2000, "epochs": 20},  # 4x gradient steps, same corpus
    {"n_embd": 61, "n_layer": 6, "n_trajectories": 2000, "epochs": 40},  # 8x gradient steps, same corpus
    {"n_embd": 61, "n_layer": 6, "n_trajectories": 5000, "epochs": 20},  # corpus-scaling control vs row 2
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
               lr: float = 5e-3, seed: int = 0) -> dict:
    """Train a single cascade host. No aux supervision (the load-bearing
    hypothesis of the sweep)."""
    from scripts.train_cascade_host import train as train_cascade_host
    t0 = time.time()
    meta = train_cascade_host(
        n_embd=n_embd, n_layer=n_layer,
        n_trajectories=n_trajectories, epochs=epochs, lr=lr,
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


def _normalize_configs(raw_configs: list,
                       *, default_n_trajectories: int,
                       default_epochs: int,
                       default_lr: float) -> list[dict]:
    """Turn the grid representations into a uniform list of config dicts,
    each carrying its own training budget + a stable host-dir name.

    Accepts either 2-tuples ``(n_embd, n_layer)`` (capacity/depth grids,
    which inherit the global ``--n-trajectories``/``--epochs``/``--lr``)
    or dicts (the budget grid, whose per-row ``n_trajectories``/``epochs``
    override the globals). Capacity/depth configs keep the legacy
    ``sweep_NE{ne}_L{nl}`` dir name so re-runs reuse the same hosts;
    budget configs append ``_T{traj}_E{ep}`` so same-architecture rows
    don't collide.
    """
    norm: list[dict] = []
    for c in raw_configs:
        if isinstance(c, dict):
            n_embd = c["n_embd"]
            n_layer = c["n_layer"]
            n_traj = c.get("n_trajectories", default_n_trajectories)
            epochs = c.get("epochs", default_epochs)
            lr = c.get("lr", default_lr)
            base = _config_dir_name(n_embd, n_layer)
            # Disambiguate only when the budget differs from the global
            # default, so a dict that happens to match the defaults still
            # lands on the legacy dir name.
            if n_traj == default_n_trajectories and epochs == default_epochs:
                dir_name = base
            else:
                dir_name = f"{base}_T{n_traj}_E{epochs}"
        else:
            n_embd, n_layer = c
            n_traj = default_n_trajectories
            epochs = default_epochs
            lr = default_lr
            dir_name = _config_dir_name(n_embd, n_layer)
        norm.append({
            "n_embd": n_embd, "n_layer": n_layer,
            "n_trajectories": n_traj, "epochs": epochs, "lr": lr,
            "dir_name": dir_name,
        })
    return norm


# color:r anchors. PR #28 (rollout entropy) cited the host surfacing
# color:r at 0.74 and the state_t direct classifier at 0.90, and
# predicted training budget would lift the host toward that ceiling.
# IMPORTANT: the 0.74 figure was the L2 baseline host; this sweep fixes
# the host at the L6 depth peak, where DEPTH ALONE already lifts color:r
# to ~0.877 (see runs/depth_sweep). So the residual headroom this sweep
# probes (≈0.877 -> 0.90) is far smaller than the 0.74 -> 0.90 PR #28
# dramatised. The prediction is therefore operationalised
# host-agnostically: color:r must rise monotonically along the
# gradient-step axis AND clear its OWN lowest-budget baseline by the
# lift threshold below.
_COLOR_R_L2_REFERENCE = 0.74     # PR #28's cited figure (the L2 host)
_COLOR_R_CEILING = 0.90          # state_t direct-classifier ceiling
_COLOR_R_LIFT_THRESHOLD = 0.02   # min lift over the observed baseline to count as "lifting"


def _budget_trend(rows: list[dict]) -> dict:
    """Gate-B.3 summary for the training-budget sweep: does color:r
    residual-probe AUC lift toward its state_t ceiling as gradient-step
    count grows (PR #28's falsifiable prediction)?

    Separates the gradient-step axis (rows sharing the smallest corpus)
    from corpus-scaling controls (larger n_trajectories) so the dominant
    lever — more passes vs more data — is identifiable. The prediction is
    judged host-agnostically (lift over the lowest-budget baseline)
    because depth has already moved the L6 host's color:r well above the
    0.74 figure PR #28 anchored on.
    """
    trail = [
        {
            "n_trajectories":         r["n_trajectories"],
            "epochs":                 r["epochs"],
            "n_train_steps":          r["n_train_steps"],
            "train_loss_final":       r["train_loss_final"],
            "forge_delta_vs_random":  r["forge_delta_vs_random"],
            "color_r_auc":            r["probe_color_r_auc"],
            "spotlight_median_auc":   r["probe_spotlight_median_auc"],
        }
        for r in sorted(rows, key=lambda r: (r.get("n_train_steps") or 0))
    ]
    min_traj = min((r["n_trajectories"] for r in rows), default=None)
    grad_axis = [t for t in trail if t["n_trajectories"] == min_traj]
    grad_color_r = [t["color_r_auc"] for t in grad_axis if t["color_r_auc"] is not None]
    all_color_r = [t["color_r_auc"] for t in trail if t["color_r_auc"] is not None]
    forge_deltas = [t["forge_delta_vs_random"] for t in trail
                    if t["forge_delta_vs_random"] is not None]

    monotonic = all(b >= a - 1e-9 for a, b in zip(grad_color_r, grad_color_r[1:]))
    baseline = grad_color_r[0] if grad_color_r else (all_color_r[0] if all_color_r else None)
    best = max(all_color_r) if all_color_r else None
    lift = (round(best - baseline, 4)
            if best is not None and baseline is not None else None)

    return {
        "gradient_step_axis":         grad_axis,
        "corpus_scaling_controls":    [t for t in trail if t["n_trajectories"] != min_traj],
        "color_r_baseline_observed":  baseline,
        "color_r_best":               best,
        "color_r_lift":               lift,
        "color_r_l2_reference":       _COLOR_R_L2_REFERENCE,
        "color_r_state_t_ceiling":    _COLOR_R_CEILING,
        "color_r_lift_threshold":     _COLOR_R_LIFT_THRESHOLD,
        "gradient_step_axis_monotonic": monotonic,
        # gate B.3: prediction holds iff color:r rises monotonically along
        # the gradient-step axis AND clears its own baseline by the lift
        # threshold (host-agnostic; depth already moved L6 above 0.74).
        "prediction_holds":           bool(lift is not None
                                           and lift >= _COLOR_R_LIFT_THRESHOLD
                                           and monotonic),
        # gate B.2: gate 7.3 closes iff any config beats random by >= 0.05.
        "gate_7_3_closed":            any(d >= 0.05 for d in forge_deltas),
        "best_forge_delta_vs_random": max(forge_deltas, default=None),
    }


def run_sweep(out_dir: Path, configs: list,
              n_trajectories: int = 2000, epochs: int = 5,
              lr: float = 5e-3, grid: str = "capacity",
              probe_n_trajectories: int = 5000) -> dict:
    """Run train -> forge -> probe across the grid. Returns the summary
    payload (also written to out_dir / summary.json).

    ``configs`` may be 2-tuples ``(n_embd, n_layer)`` (capacity/depth)
    or dicts carrying per-row ``n_trajectories``/``epochs`` (budget).
    The global ``n_trajectories``/``epochs``/``lr`` are the defaults each
    config inherits when it doesn't override them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_root = Path(REPO_ROOT) / "runs" / "cascade_host"

    norm_configs = _normalize_configs(
        configs, default_n_trajectories=n_trajectories,
        default_epochs=epochs, default_lr=lr,
    )

    random_baseline = _measure_random_baseline(out_dir)

    rows: list[dict] = []
    grid_t0 = time.time()
    for i, cfg in enumerate(norm_configs, start=1):
        n_embd = cfg["n_embd"]
        n_layer = cfg["n_layer"]
        cfg_n_traj = cfg["n_trajectories"]
        cfg_epochs = cfg["epochs"]
        cfg_lr = cfg["lr"]
        config_name = cfg["dir_name"]
        host_dir = sweep_root / config_name
        print(f"\n[sweep] [{i}/{len(norm_configs)}] config {config_name} "
              f"(NE{n_embd} L{n_layer} T{cfg_n_traj} E{cfg_epochs})")

        # Train
        train_meta = _train_one(n_embd, n_layer, host_dir,
                                n_trajectories=cfg_n_traj, epochs=cfg_epochs,
                                lr=cfg_lr)
        print(f"[sweep]   trained in {train_meta['train_wall_s']}s "
              f"(final_loss={train_meta['train_loss_final']:.3f}, "
              f"n_params={train_meta['n_params']}, "
              f"steps={train_meta.get('n_train_steps')})")

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
            "n_trajectories":       cfg_n_traj,
            "epochs":               cfg_epochs,
            "lr":                   cfg_lr,
            "n_params":             train_meta.get("n_params"),
            "n_train_steps":        train_meta.get("n_train_steps"),
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

    # Budget-sweep gate B.3 (color:r lift vs gradient-step count). Only
    # meaningful when the architecture is held fixed and the budget
    # varies, so attach it for the budget grid.
    if grid == "budget":
        gate_summary["B.3_budget_trend"] = _budget_trend(rows)

    summary = {
        "grid":             grid,
        "configs_run":      [
            {"n_embd": c["n_embd"], "n_layer": c["n_layer"],
             "n_trajectories": c["n_trajectories"], "epochs": c["epochs"]}
            for c in norm_configs
        ],
        "n_trajectories":   n_trajectories,
        "epochs":           epochs,
        "lr":               lr,
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

    bt = gate_summary.get("B.3_budget_trend")
    if bt is not None:
        print()
        print("  --- budget trend (gate B.2 / B.3) ---")
        print(f"  {'n_traj':>7} {'epochs':>6} {'steps':>6} "
              f"{'Δ_rand':>8} {'color:r':>8} {'spot_med':>8}")
        for t in bt["gradient_step_axis"] + bt["corpus_scaling_controls"]:
            d = t["forge_delta_vs_random"]
            cr = t["color_r_auc"]
            sm = t["spotlight_median_auc"]
            print(f"  {t['n_trajectories']:>7} {t['epochs']:>6} "
                  f"{(t['n_train_steps'] or 0):>6} "
                  f"{(f'{d:+.4f}' if d is not None else 'N/A'):>8} "
                  f"{(f'{cr:.3f}' if cr is not None else 'N/A'):>8} "
                  f"{(f'{sm:.3f}' if sm is not None else 'N/A'):>8}")
        lift = bt["color_r_lift"]
        lift_str = f"{lift:+.4f}" if lift is not None else "N/A"
        print(f"  B.2 gate 7.3 (Δ≥+0.05): {'PASS' if bt['gate_7_3_closed'] else 'FAIL'} "
              f"(best Δ={(f'{bt['best_forge_delta_vs_random']:+.4f}' if bt['best_forge_delta_vs_random'] is not None else 'N/A')})")
        base_obs = bt["color_r_baseline_observed"]
        base_str = f"{base_obs:.3f}" if base_obs is not None else "N/A"
        print(f"  B.3 color:r lift (PR #28 prediction): "
              f"{'PASS' if bt['prediction_holds'] else 'FAIL'} "
              f"(baseline={base_str}, best={bt['color_r_best']}, lift={lift_str}, "
              f"monotonic={bt['gradient_step_axis_monotonic']}; "
              f"need lift ≥ {bt['color_r_lift_threshold']} toward ceiling "
              f"{bt['color_r_state_t_ceiling']}; L2 ref {bt['color_r_l2_reference']})")

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Default --out depends on the grid; resolved below once --grid is known.
    ap.add_argument("--out", default=None, type=Path,
                    help="output directory for summary.json / summary.csv "
                         "and per-config probe/forge artefacts. Default: "
                         "runs/capacity_sweep (capacity), runs/depth_sweep "
                         "(depth), runs/budget_sweep (budget).")
    ap.add_argument("--n-trajectories", type=int, default=2000,
                    help="trajectories per host training run (the global "
                         "default; budget configs override it per-row)")
    ap.add_argument("--epochs", type=int, default=5,
                    help="global default epochs (budget configs override "
                         "per-row)")
    ap.add_argument("--lr", type=float, default=5e-3,
                    help="global default peak learning rate (budget configs "
                         "may override per-row)")
    ap.add_argument("--probe-n-trajectories", type=int, default=5000,
                    help="trajectories for the probe dataset (per config)")
    ap.add_argument("--smoke", action="store_true",
                    help="run a 2-config smoke test to verify wiring. For "
                         "'capacity' this is (61,2)+(96,2) (exercises both "
                         "forge-measurable and probe-only paths); for "
                         "'depth'/'budget' it's the first 2 configs of the "
                         "selected grid.")
    ap.add_argument("--grid", default="capacity",
                    choices=("capacity", "depth", "budget", "custom"),
                    help="config preset. 'capacity' (default) — the 6-config "
                         "(n_embd, n_layer) grid from PR #25. 'depth' — the "
                         "4-config follow-up at fixed n_embd=61, varying "
                         "n_layer in {6,8,10,12} (cascade-host-depth-sweep). "
                         "'budget' — the 4-config training-budget grid at "
                         "fixed (61,6), varying n_trajectories/epochs "
                         "(cascade-host-training-budget-sweep). "
                         "'custom' — supply --config NE_L flags.")
    ap.add_argument("--config", action="append", default=[],
                    help="repeatable NE_L specifier when --grid=custom "
                         "(e.g. --config 61_8 --config 96_4).")
    args = ap.parse_args()

    if args.grid == "depth":
        configs = DEPTH_GRID
        print(f"[sweep] --grid=depth: 4-config depth follow-up "
              f"{configs}")
    elif args.grid == "budget":
        configs = BUDGET_GRID
        print(f"[sweep] --grid=budget: 4-config training-budget sweep at "
              f"fixed (61,6); "
              f"{[(c['n_trajectories'], c['epochs']) for c in configs]} (T,E)")
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
        if args.grid == "capacity":
            # one forge-measurable + one probe-only to verify both paths
            configs = [(61, 2), (96, 2)]
            print("[sweep] --smoke: limiting to (61,2) + (96,2) — exercises "
                  "both forge-measurable and probe-only paths")
        else:
            configs = configs[:2]
            print(f"[sweep] --smoke: limiting to the first 2 {args.grid} "
                  f"configs")

    default_out = {
        "capacity": "runs/capacity_sweep",
        "depth":    "runs/depth_sweep",
        "budget":   "runs/budget_sweep",
        "custom":   "runs/custom_sweep",
    }[args.grid]
    out = args.out if args.out is not None else Path(default_out)

    summary = run_sweep(
        out if out.is_absolute() else Path(out),
        configs=configs,
        n_trajectories=args.n_trajectories,
        epochs=args.epochs,
        lr=args.lr,
        grid=args.grid,
        probe_n_trajectories=args.probe_n_trajectories,
    )
    return 0 if summary["gate_summary"]["C.1_mechanical_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
