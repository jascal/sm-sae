"""Run scripts/forge_pipeline.py across the (run_id × encoding) matrix.

Initial sweep: {embedded__topk, cascade__jumprelu} × {mps_rung1, rung3,
rung4, rung5} = 8 cells. One forge_results.json per cell under
runs/sae_forge/<run_id>__<encoding>/.

- Skips cells whose SAE checkpoint runs/<run_id>.pt doesn't exist.
- Resumable: skips cells whose forge_results.json already exists,
  unless --force is passed.

Usage:
    python scripts/forge_pipeline_matrix.py
    python scripts/forge_pipeline_matrix.py --force
    python scripts/forge_pipeline_matrix.py --runs embedded__topk \\
        --encodings rung3 rung5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RUNS      = ["embedded__topk", "cascade__jumprelu"]
DEFAULT_ENCODINGS = ["mps_rung1", "rung3", "rung4", "rung5"]


def _cell_dir(out_root: str, run_id: str, encoding: str) -> str:
    return os.path.join(out_root, f"{run_id}__{encoding}")


def _run_one(run_id: str, encoding: str, out_root: str, force: bool,
             extra_args: list[str]) -> dict:
    cell_id = f"{run_id}__{encoding}"
    cell_dir = _cell_dir(out_root, run_id, encoding)
    result_path = os.path.join(cell_dir, "forge_results.json")
    ckpt_path = os.path.join(REPO_ROOT, "runs", f"{run_id}.pt")

    if not os.path.exists(ckpt_path):
        return {"cell": cell_id, "status": "skip-no-ckpt", "ckpt": ckpt_path}
    if not force and os.path.exists(result_path):
        return {"cell": cell_id, "status": "skip-existing", "path": result_path}

    os.makedirs(cell_dir, exist_ok=True)
    # forge_pipeline writes to <args.out>/<run_id>/, but we want
    # one directory per (run_id, encoding). Run with --out=<out_root>
    # then move <out_root>/<run_id>/ → <out_root>/<run_id>__<encoding>/.
    tmp_dir = os.path.join(out_root, run_id)
    tmp_result = os.path.join(tmp_dir, "forge_results.json")
    if os.path.exists(tmp_result):
        os.remove(tmp_result)

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "forge_pipeline.py"),
        run_id,
        "--encoding", encoding,
        "--out", out_root,
        *extra_args,
    ]

    print(f"\n=== {cell_id} ({run_id} @ {encoding}) ===")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        return {"cell": cell_id, "status": "error",
                "returncode": proc.returncode}
    if not os.path.exists(tmp_result):
        return {"cell": cell_id, "status": "error", "reason": "no result file"}

    # Move the per-cell artifacts from <out>/<run_id>/ to
    # <out>/<run_id>__<encoding>/ so encodings don't clobber each other.
    import shutil
    if os.path.exists(cell_dir):
        shutil.rmtree(cell_dir)
    shutil.move(tmp_dir, cell_dir)
    return {"cell": cell_id, "status": "ok", "path": result_path}


def _summarize(out_root: str, runs: list[str], encodings: list[str]) -> None:
    print("\n=== matrix summary ===")
    hdr = ("cell", "post-A", "Δ A", "post-B cov", "Δ B", "forge_score")
    print("  {:<32}  {:>6}  {:>6}  {:>10}  {:>6}  {:>8}".format(*hdr))
    for run_id in runs:
        for enc in encodings:
            cell_id = f"{run_id}__{enc}"
            result_path = os.path.join(_cell_dir(out_root, run_id, enc),
                                       "forge_results.json")
            if not os.path.exists(result_path):
                print(f"  {cell_id:<32}  {'—':>6}  {'—':>6}  {'—':>10}  "
                      f"{'—':>6}  {'—':>8}")
                continue
            with open(result_path) as f:
                r = json.load(f)
            pcs = r.get("post_compress_score") or {}
            pa = pcs.get("var_explained")
            dA = pcs.get("var_explained_delta")
            pb = pcs.get("coverage_0.95")
            dB = pcs.get("coverage_0.95_delta")
            fs = r.get("forge_score")

            def _fmt(v, spec):
                return spec.format(v) if isinstance(v, (int, float)) else "—"

            print(f"  {cell_id:<32}  "
                  f"{_fmt(pa, '{:>6.3f}'):>6}  "
                  f"{_fmt(dA, '{:>+6.3f}'):>6}  "
                  f"{_fmt(pb, '{:>9.1%}'):>10}  "
                  f"{_fmt(dB, '{:>+5.1%}'):>6}  "
                  f"{_fmt(fs, '{:>8.4f}'):>8}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS,
                    help=f"run_ids to sweep. Default: {DEFAULT_RUNS}")
    ap.add_argument("--encodings", nargs="*", default=DEFAULT_ENCODINGS,
                    choices=DEFAULT_ENCODINGS,
                    help=f"encodings to sweep. Default: {DEFAULT_ENCODINGS}")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "sae_forge"))
    ap.add_argument("--force", action="store_true",
                    help="re-run even if forge_results.json already exists")
    args, extra = ap.parse_known_args()

    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

    cells = [(r, e) for r in args.runs for e in args.encodings]
    print(f"forge_pipeline_matrix: {len(cells)} cells, out={out_root}, "
          f"force={args.force}")

    results = []
    for run_id, enc in cells:
        results.append(_run_one(run_id, enc, out_root, args.force, extra))

    _summarize(out_root, args.runs, args.encodings)

    print("\n=== status ===")
    for r in results:
        print(f"  {r['cell']}: {r['status']}")


if __name__ == "__main__":
    main()
