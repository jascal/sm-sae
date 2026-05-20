"""Phase C: Convert the SM bundle into a polygram Dictionary and run
interference/cancellation experiments against it.

Goal: prove the SM bundle works as a meaningful polygram test fixture.
polygram is a sparse-feature-dictionary analysis lab — features encoded into
a quantum register, then phase-interference sweeps and Cancellation
optimization over the encoding.

Constraints:
  - MPSRung1 caps at 8 features (3-qubit Hilbert dim).
  - Rung3 caps at 16 features.
  - We pick 8 SM particles whose electric charges span a useful range, with
    4 physical clusters (charged_lepton / up_quark / down_quark / boson)
    that produce a meaningful hierarchy for polygram's `preserve_tiers`.

Experiments:
  1. Build a polygram.Dictionary from an 8-particle slice of the SM.
  2. InterferenceSweep over a chosen phi knob; record target-pair overlap
     across the sweep + the structural floor.
  3. Cancellation on a particle-antiparticle pair: can polygram drive their
     overlap to zero by phase alone? (Physically: e- and e+ are
     CPT-conjugate; their inner product on a phase-invariant encoding is
     constrained by their beta and amplitude structure.)
"""

from __future__ import annotations

import os

import numpy as np

from polygram import Cancellation, Dictionary, Experiment, Feature, MPSRung1

from smsae.sm.embeddings import build_sm


OUT_DIR = "runs/polygram"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Choice of 8 particles (fits MPSRung1's 8-feature cap)
# ---------------------------------------------------------------------------
SM_SLICE = [
    # name,    cluster
    ("e-",     "charged_lepton"),
    ("e+",     "charged_lepton"),
    ("u_r",    "up_quark"),
    ("~u_r",   "up_quark"),
    ("d_r",    "down_quark"),
    ("~d_r",   "down_quark"),
    ("photon", "boson"),
    ("W+",     "boson"),
]


def sm_to_polygram_dictionary(name: str = "SM_Rung1_Slice") -> Dictionary:
    """Build a polygram Dictionary from the SM slice.

    beta = electric charge (gives a meaningful spread: -1, +1, 2/3, -2/3,
    -1/3, +1/3, 0, +1).
    """
    sm = build_sm()
    features: list[Feature] = []
    hierarchy: dict[str, list[str]] = {}
    for pname, cluster in SM_SLICE:
        beta = float(sm[pname].vec[0])  # electric charge from particle_vectors
        # polygram's Feature names should be valid identifiers; remap a few.
        feat_name = (pname.replace("~", "anti_").replace("+", "_plus")
                            .replace("-", "_minus"))
        features.append(Feature(name=feat_name, cluster=cluster, beta=beta))
        hierarchy.setdefault(cluster, []).append(feat_name)

    return Dictionary(
        name=name,
        features=features,
        hierarchy=hierarchy,
        encoding=MPSRung1(bond_dim=2, phase_knobs=True),
    )


# ---------------------------------------------------------------------------
def run_interference_sweep(dictionary: Dictionary):
    """Sweep the e+ phase from 0 to 2π and watch the (e-, e+) overlap."""
    print("\n--- Interference sweep:  phi(e+) from 0 to 2*pi ---")
    target = ("e_minus", "e_plus")
    experiment = Experiment(
        name="SM_e_pair_phase_sweep",
        dictionary=dictionary,
        target_pair=target,
        sweep={"e_plus.phi": np.linspace(0.0, 2 * np.pi, 60)},
        measures=["overlap", "gram_matrix", "schmidt_rank"],
        assertions=["hierarchical_ordering_preserved"],
    )
    out_path = os.path.join(OUT_DIR, "interference_e_pair")
    os.makedirs(out_path, exist_ok=True)
    experiment.materialize(out_path)
    result = experiment.run()
    overlaps = list(result.overlaps)
    print(f"  swept {len(overlaps)} phi values")
    print(f"  target overlap: min={min(overlaps):.4f}  "
          f"max={max(overlaps):.4f}  mean={float(np.mean(overlaps)):.4f}")
    csv_path = os.path.join(OUT_DIR, "interference_e_pair.csv")
    result.to_csv(csv_path)
    print(f"  wrote {csv_path}  and {out_path}/  (.q.orca.md artifact)")
    return result


# ---------------------------------------------------------------------------
def run_cancellation(dictionary: Dictionary, pair: tuple[str, str], label: str):
    """Search for phase values that drive a target pair's overlap to ~0."""
    print(f"\n--- Cancellation:  drive |<{pair[0]}|{pair[1]}>|^2 to ~0 ---")
    cancel = Cancellation(
        dictionary=dictionary,
        target_pair=pair,
        tolerance=0.05,
        preserve_tiers=True,
        optimize={"method": "grid", "max_steps": 40},
    )
    result = cancel.run()
    print(f"  before: |<A|B>|^2 = {result.before_overlap:.4f}")
    print(f"  after:  |<A|B>|^2 = {result.after_overlap:.4f}")
    print(f"  structural floor:  {result.structural_floor:.4f}  "
          f"(phase alone cannot go below this)")
    # polygram v0.11+ splits "at floor" (efficiency=0.0, at_structural_floor=True)
    # from "floor undefined" (efficiency=None). Pre-0.11 results lack the new
    # field; `getattr(..., False)` makes this work against either version.
    # Other v0.11 additions (not yet read by this bridge but available on
    # `result` and on `CompressionReport` / `EpochReport` / `RegrowReport`):
    # the convergence-test diagnostics `rank_ratio`, `post_A`, `forge_mse`,
    # `informative_metric`. Natural columns to add if a future summary
    # surfaces forge-quality at the encoding level.
    at_floor = getattr(result, "at_structural_floor", False)
    if result.cancellation_efficiency is None or at_floor:
        print(f"  cancellation efficiency: N/A (no gap to begin with)")
    else:
        print(f"  cancellation efficiency: {result.cancellation_efficiency:.2%}  "
              f"(1.0 = exhausted available gap; gap is amplitude-bound, not phase-bound)")
    print(f"  tolerance met: {result.tolerance_met}")
    print(f"  optimized knobs: {dict(result.optimized_knobs)}")
    print(f"  trajectory length: {len(result.trajectory)} evaluations")
    out_path = os.path.join(OUT_DIR, f"cancellation_{label}")
    os.makedirs(out_path, exist_ok=True)
    result.materialize(out_path)
    print(f"  wrote {out_path}/")
    return result


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("Build polygram Dictionary from SM slice")
    print("=" * 78)
    dictionary = sm_to_polygram_dictionary()
    print(f"  Dictionary: {dictionary.name}")
    print(f"  Features ({len(dictionary.features)}):")
    for f in dictionary.features:
        print(f"    {f.name:<14s}  cluster={f.cluster:<14s}  beta={f.beta:+.4f}")
    print(f"  Hierarchy: {dictionary.hierarchy}")
    print(f"  Encoding: MPSRung1(bond_dim=2, phase_knobs=True)")

    sweep_result = run_interference_sweep(dictionary)

    # Pair experiments
    pairs = [
        ("e_minus",  "e_plus",  "e_e"),       # particle / antiparticle (same cluster)
        ("u_r",      "anti_u_r", "u_pair"),    # quark / antiquark (same cluster)
        ("e_minus",  "u_r",      "lepton_quark"),  # cross-cluster
        ("photon",   "W_plus",   "boson_pair"),     # same cluster, different beta
    ]
    cancellations = []
    for a, b, label in pairs:
        try:
            cancellations.append(run_cancellation(dictionary, (a, b), label))
        except Exception as e:
            print(f"  cancellation {label} failed: {type(e).__name__}: {e}")

    # Summary table
    print("\n" + "=" * 78)
    print("CANCELLATION SUMMARY")
    print("=" * 78)
    print(f"{'pair':<22s} {'before':>8s} {'after':>8s} {'floor':>8s} "
          f"{'efficiency':>11s} {'met?':>5s}")
    print("-" * 70)
    for (a, b, label), res in zip(pairs, cancellations):
        at_floor = getattr(res, "at_structural_floor", False)
        eff = (
            "N/A"
            if (res.cancellation_efficiency is None or at_floor)
            else f"{res.cancellation_efficiency:.2%}"
        )
        print(f"{a + ' / ' + b:<22s} {res.before_overlap:>8.4f} "
              f"{res.after_overlap:>8.4f} {res.structural_floor:>8.4f} "
              f"{eff:>11s} {str(res.tolerance_met):>5s}")


if __name__ == "__main__":
    main()
