"""Benchmark sweep: run the same 4 cancellation pairs against multiple
polygram encoding configurations to see which can crack the structural
floors that the baseline (MPSRung1, phase-only) cannot.

Configurations tested:
  - MPSRung1 phase-only       (2 knobs: a.phi, b.phi)               -- baseline
  - Rung3 with amplitude     (4 knobs: + b.theta_amp, b.psi_aux)
  - Rung4 with amplitude     (6 knobs: + b.theta_amp_b, b.psi_amp_b)

Writes per-config / per-pair summaries under runs/polygram/sweep/<config>/...
and aggregates a sweep_results.json the visualize.py scoreboard reads.

Run:
    python scripts/polygram_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from polygram import (Cancellation, Dictionary, Feature, MPSRung1, Rung3, Rung4,
                       Rung5)

from smsae.polygram_bridge import SM_SLICE
from smsae.sm.embeddings import build_sm


OUT_DIR = os.path.join(REPO_ROOT, "runs", "polygram", "sweep")
os.makedirs(OUT_DIR, exist_ok=True)


CONFIGS = [
    {
        "name":     "MPSRung1_phase",
        "encoding": MPSRung1(bond_dim=2, phase_knobs=True),
        "cancellation_encoding": None,        # default 2-knob phase-only
        "optimize": {"method": "grid", "max_steps": 40},
        "desc":     "Baseline. 2 knobs (a.phi, b.phi); phase-only.",
    },
    {
        "name":     "Rung3_amp",
        "encoding": Rung3(bond_dim=2),
        "cancellation_encoding": "rung3",     # 4 knobs incl. theta_amp, psi_aux
        "optimize": {"method": "grid"},
        "desc":     "Rung3 amplitude branch (4 knobs: +theta_amp, +psi_aux).",
    },
    {
        "name":     "Rung4_amp_default",
        "encoding": Rung4(bond_dim=2),
        "cancellation_encoding": "rung4",
        "optimize": {"method": "scipy"},        # max_steps defaults to 50
        "desc":     "Rung4 (6 knobs) at polygram's default scipy budget (max_steps=50).",
        # Don't run missing pairs — the unlimited-budget run takes ~80 min per
        # pair. Only re-use what is already on disk (e_e was completed).
        "skip_if_missing": True,
    },
    {
        "name":     "Rung4_amp_budget",
        "encoding": Rung4(bond_dim=2),
        "cancellation_encoding": "rung4",
        # max_steps=10 is 5x cheaper than the default; differential_evolution
        # may still converge early via its tol parameter. Goal: ~15 min/pair.
        "optimize": {"method": "scipy", "max_steps": 10, "seed": 0},
        "desc":     "Rung4 (6 knobs) at limited budget (max_steps=10).",
    },
    {
        "name":     "Rung5_amp_budget",
        # n_amp_qubits=2 → 2 + 2*2 = 6 knobs; matches Rung4's count for an
        # apples-to-apples comparison of the encoding family.
        "encoding": Rung5(bond_dim=2, n_amp_qubits=2),
        "cancellation_encoding": "rung5",
        "optimize": {"method": "scipy", "max_steps": 10, "seed": 0},
        "desc":     "Rung5 (n_amp_qubits=2, 6 knobs) at limited budget.",
    },
]


PAIRS = [
    ("e_minus",  "e_plus",   "e_e"),
    ("u_r",      "anti_u_r", "u_pair"),
    ("e_minus",  "u_r",      "lepton_quark"),
    ("photon",   "W_plus",   "boson_pair"),
]


def build_dictionary(name: str, encoding) -> Dictionary:
    sm = build_sm()
    features: list[Feature] = []
    hierarchy: dict[str, list[str]] = {}
    for pname, cluster in SM_SLICE:
        beta = float(sm[pname].vec[0])
        feat_name = (pname.replace("~", "anti_").replace("+", "_plus")
                          .replace("-", "_minus"))
        f = Feature(name=feat_name, cluster=cluster, beta=beta)
        # Rung5 (and any encoding with n_amp_qubits > 0) requires per-feature
        # amp_knobs to be populated; call the helper polygram provides.
        if isinstance(encoding, Rung5):
            f = f.with_default_amp_knobs(encoding)
        features.append(f)
        hierarchy.setdefault(cluster, []).append(feat_name)
    return Dictionary(
        name=name,
        features=features,
        hierarchy=hierarchy,
        encoding=encoding,
    )


def run_one(config: dict, pair_a: str, pair_b: str, label: str) -> dict:
    dictionary = build_dictionary(f"SM_{config['name']}", config["encoding"])
    kwargs = dict(
        dictionary=dictionary,
        target_pair=(pair_a, pair_b),
        tolerance=0.05,
        preserve_tiers=True,
        optimize=config["optimize"],
    )
    if config["cancellation_encoding"] is not None:
        kwargs["encoding"] = config["cancellation_encoding"]
    cancel = Cancellation(**kwargs)
    t0 = time.time()
    result = cancel.run()
    elapsed = time.time() - t0
    out_path = os.path.join(OUT_DIR, config["name"], f"cancellation_{label}")
    os.makedirs(out_path, exist_ok=True)
    try:
        result.materialize(out_path)
    except Exception:
        pass  # not all encodings may support materialize
    return {
        "config":            config["name"],
        "config_desc":       config["desc"],
        "pair":              f"{pair_a}/{pair_b}",
        "label":             label,
        "n_knobs":           len(cancel.knobs),
        "before":            float(result.before_overlap),
        "after":             float(result.after_overlap),
        "delta":             float(result.after_overlap - result.before_overlap),
        "structural_floor":  float(result.structural_floor),
        "tolerance":         0.05,
        "tolerance_met":     bool(result.tolerance_met),
        "wall_time_s":       elapsed,
    }


def _existing_result(config_name: str, label: str) -> dict | None:
    """If this (config, label) has already been run and a summary.md exists,
    parse it back into the same dict shape run_one() emits — so we can skip
    already-completed work after a long Rung4 search."""
    import re as _re
    md = os.path.join(OUT_DIR, config_name, f"cancellation_{label}",
                      f"SM_{config_name}_at_optimum_summary.md")
    if not os.path.exists(md):
        return None
    with open(md) as f:
        text = f.read()
    def _pull(pat: str) -> str | None:
        m = _re.search(pat, text)
        return m.group(1) if m else None
    pair_a = _pull(r"target pair:\s*`([^`]+)`")
    pair_b = None
    m = _re.search(r"target pair:\s*`[^`]+`\s*×\s*`([^`]+)`", text)
    if m:
        pair_b = m.group(1)
    before = _pull(r"before:\s*([\-\d.]+)")
    after = _pull(r"after:\s*([\-\d.]+)")
    floor = _pull(r"structural_floor:\s*([\-\d.]+)")
    tol = _pull(r"tolerance_met:\s*(\w+)")
    if before is None or after is None or floor is None:
        return None
    knobs_line = _re.search(r"knobs:\s*`([^`]+)`", text)
    n_knobs = len(knobs_line.group(1).split(",")) if knobs_line else None
    return {
        "config":           config_name,
        "pair":             f"{pair_a}/{pair_b}",
        "label":            label,
        "n_knobs":          n_knobs,
        "before":           float(before),
        "after":            float(after),
        "delta":            float(after) - float(before),
        "structural_floor": float(floor),
        "tolerance_met":    (tol or "").lower() == "true",
        "wall_time_s":      0.0,
        "resumed":          True,
    }


def _write_summary(all_results: list[dict]) -> None:
    summary = {
        "n_total": len(all_results),
        "n_pass":  sum(1 for r in all_results if r.get("tolerance_met")),
        "n_error": sum(1 for r in all_results if "error" in r),
        "by_config": {},
        "runs":     all_results,
    }
    for config in CONFIGS:
        rows = [r for r in all_results if r.get("config") == config["name"]]
        summary["by_config"][config["name"]] = {
            "desc":   config["desc"],
            "n_runs": len(rows),
            "n_pass": sum(1 for r in rows if r.get("tolerance_met")),
            "n_error": sum(1 for r in rows if "error" in r),
            "best_delta": min((r.get("delta", 0) for r in rows
                              if "delta" in r), default=0.0),
        }
    sweep_path = os.path.join(OUT_DIR, "sweep_results.json")
    with open(sweep_path, "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    print(f"Polygram sweep — {len(CONFIGS)} configs × {len(PAIRS)} pairs "
          f"= {len(CONFIGS) * len(PAIRS)} runs")
    print(f"Output → {OUT_DIR}\n")

    all_results: list[dict] = []
    for config in CONFIGS:
        print(f"--- {config['name']} ({config['desc']}) ---")
        skip_missing = config.get("skip_if_missing", False)
        for pair_a, pair_b, label in PAIRS:
            cached = _existing_result(config["name"], label)
            if cached is not None:
                print(f"  {label:<14s}  [resumed from disk]  "
                      f"Δ={cached['delta']:+.4f}  "
                      f"met={cached['tolerance_met']}")
                all_results.append(cached)
                _write_summary(all_results)
                continue
            if skip_missing:
                print(f"  {label:<14s}  [skipped: skip_if_missing=True]")
                continue
            try:
                r = run_one(config, pair_a, pair_b, label)
                status = "PASS" if r["tolerance_met"] else "fail"
                print(f"  {label:<14s}  {r['n_knobs']}-knob  "
                      f"before={r['before']:.4f}  after={r['after']:.4f}  "
                      f"Δ={r['delta']:+.4f}  floor={r['structural_floor']:.4f}  "
                      f"[{status}]  ({r['wall_time_s']:.2f}s)")
                all_results.append(r)
            except Exception as e:
                print(f"  {label:<14s}  ERROR: {type(e).__name__}: {e}")
                all_results.append({
                    "config": config["name"], "pair": f"{pair_a}/{pair_b}",
                    "label": label, "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
            _write_summary(all_results)
        print()

    _write_summary(all_results)
    sweep_path = os.path.join(OUT_DIR, "sweep_results.json")
    with open(sweep_path) as f:
        summary = json.load(f)
    print("=" * 70)
    print(f"SWEEP SUMMARY  ({summary['n_pass']}/{summary['n_total']} "
          f"cancellations passed; {summary['n_error']} errors)")
    print("=" * 70)
    for name, agg in summary["by_config"].items():
        marker = "✓" if agg["n_pass"] == agg["n_runs"] else (
            "·" if agg["n_pass"] > 0 else "✗")
        print(f"  [{marker}]  {name:<22s} "
              f"pass {agg['n_pass']}/{agg['n_runs']}  "
              f"errors {agg['n_error']}  "
              f"best Δ {agg['best_delta']:+.4f}")
    print(f"\nWrote {sweep_path}")


if __name__ == "__main__":
    main()
