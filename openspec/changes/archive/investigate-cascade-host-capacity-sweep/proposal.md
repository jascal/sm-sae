# investigate-cascade-host-capacity-sweep

## Why

This change is the empirically-supported follow-up to the three-PR
arc that just concluded:

| PR | Result | What we learned |
|---|---|---|
| **sm-sae #19** `aux-supervise-cascade-host` (v1 — pooled, 5 labels) | gate 7.3 Δ = +0.0072 vs random | Aux supervision against ceiling-AUC labels is cosmetic |
| **sm-sae #22** `probe-full-gt-recoverability-cascade-host` | 57% baseline ≥ 0.9; 43% in 0.74-0.85 | LM-only host encodes per-particle identity at MEDIUM strength |
| **sm-sae #23** `richer-cascade-host-supervision-v2` (per_channel, 110 labels, focal-BCE) | gate 7.3 Δ = **−0.0053** vs random (WORSE than v1) | More aux labels degrade LM accuracy at this host capacity; aux head steals gradient bandwidth |

**The binding constraint is host capacity, not the aux supervision
objective.** Per-particle identity at AUC 0.74-0.85 isn't a labels
problem — it's a residual-stream-bandwidth problem. The cascade host
at n_embd=61 / n_layer=2 can't carry per-particle identity at ceiling
*alongside* next-state token prediction, and any aux pressure trades
off against LM accuracy.

This change tests that hypothesis directly: **sweep host capacity
(`n_embd`, `n_layer`) without any aux supervision; measure both gate
7.3 (forge faithfulness) and the per-feature probe; see whether the
forge score scales with capacity and the per-particle AUC gap
closes by capacity alone**. If yes, the capacity sweep is the
canonical lever for closing gate 7.3 and the aux-supervision arc
(v1, v2, probe) stays as diagnostic tooling. If no, the diagnosis
pivots again — but in a new direction we haven't yet explored.

P1 — gate 7.3 has been the longest-open gate in the project. The
v1+probe+v2 arc converged on this hypothesis through three
independent measurements. The capacity sweep is the falsifiable test
that should either close gate 7.3 directly OR rule out the
"capacity is binding" hypothesis decisively.

## What Changes

### One new script + a sweep manifest

`scripts/cascade_host_capacity_sweep.py` — trains a small grid of
cascade hosts at varying `(n_embd, n_layer)` configurations, then for
each:

1. Measures gate 7.3 (forge score on `cascade__jumprelu` rung5).
2. Runs `scripts/probe_full_gt_recoverability.py` against it.
3. Writes a per-config summary row.
4. Builds a roll-up table (CSV + JSON).

The grid:

```python
SWEEP_GRID = {
    "n_embd": [61, 96, 128, 192],
    "n_layer": [2, 4],
}  # 4 x 2 = 8 configurations
```

Plus the existing `runs/cascade_host/61/` LM-only baseline (n_embd=61,
n_layer=2) for reference. 9 configurations total.

n_embd=61 is the cascade SAE's `input_dim`. Configs with n_embd>61
require the `SubspaceProjector` to project the host's residual stream
into the 61-dim basis at forge time — which it already does for any
mismatched-dim host. The forge measurement is comparable across
configurations because the SAE's feature space is fixed.

### Per-config training settings

To keep compute manageable on Intel CPU (the gate-running platform):

| config | training time est | rationale |
|---|---|---|
| n_embd=61, n_layer=2 | ~27s (existing baseline) | reference |
| n_embd=96, n_layer=2 | ~40s | midpoint |
| n_embd=128, n_layer=2 | ~50s | matches Phase 6.2 econ-sae scale |
| n_embd=192, n_layer=2 | ~75s | upper bound on Intel CPU comfort |
| n_embd=61, n_layer=4 | ~50s | depth-only vs n_embd-only contrast |
| n_embd=96, n_layer=4 | ~75s | combined-up |
| n_embd=128, n_layer=4 | ~95s | larger combined |
| n_embd=192, n_layer=4 | ~140s | upper-bound combined |

Total wall: ~9 min for training + ~6 min for probes + ~80s for forges
= **~16 min total** on Intel CPU.

### Output: `runs/capacity_sweep/summary.json` + `summary.csv`

Per-row schema:

```json
{
  "n_embd": 96,
  "n_layer": 4,
  "n_params": ...,
  "train_loss_final": ...,
  "train_wall_s": ...,
  "forge_score_cascade__jumprelu_rung5": 0.7423,
  "forge_score_random_baseline": 0.7310,
  "forge_delta_vs_random": 0.0113,
  "probe_mean_residual_auc": 0.918,
  "probe_pct_residual_ge_0.9": 0.74,
  "probe_pct_residual_ge_0.92": 0.68,
  "probe_color_r_auc": 0.821,
  "probe_color_b_auc": 0.851,
  "probe_color_g_auc": 0.838,
  "probe_n_features_measured": 74
}
```

The `probe_color_*_auc` fields are spotlights — these were the
weakest features in PRs #22/#23 and the test of the
"capacity-not-labels" hypothesis is whether capacity alone lifts
them above 0.92.

## Acceptance gates

This change ships when:

### Gate C.1 (mechanical)

Total sweep wall ≤ 20 minutes on Intel CPU; all 8 configurations
train to a finite final loss.

### Gate C.2 (gate-7.3-by-capacity, the actual hypothesis)

**At least ONE configuration in the sweep achieves `forge_delta_vs_random ≥ 0.05`**
(the original gate 7.3 target).

If yes — capacity alone closes gate 7.3, vindicating the diagnosis from
PR #23. Archive the aux-supervision arc (v1, v2) as diagnostic
tooling; recommend the smallest-capacity-that-passes as the canonical
production host.

If no — the "capacity is binding" hypothesis is FALSIFIED. The
diagnosis pivots to: maybe the cascade-rollout vocabulary doesn't
have enough discriminative entropy, OR the `cascade__jumprelu` SAE
itself has a structural issue. File `investigate-cascade-vocabulary-entropy`
or `investigate-cascade-jumprelu-sparsity` as the next experiment.

### Gate C.3 (per-particle AUC scaling)

Looking at the spotlight features (`color:r`, `color:b`, `color:g`,
`flavor:u`, `flavor:d`, `flavor:mu`, `particle:mu+`,
`particle:u_b`): **the largest-capacity configuration SHALL have
median AUC ≥ 0.92** on these features (matching v2's gate v2.2 target,
but achieved by capacity not supervision).

If yes — the per-particle gap closes by capacity alone, definitively
confirming the diagnosis.

If no — capacity alone isn't enough; the per-particle identity
problem persists across the sweep. The hypothesis pivots to "the
LM objective can't reward per-particle identity beyond a certain
point regardless of capacity" — file a separate
`investigate-cascade-lm-objective-saturation` if this case lands.

## Capabilities

### New Capabilities

- `cascade-host-capacity-sweep`: a sweep script that trains a grid
  of cascade hosts at varying capacities, measures both gate 7.3
  forge score AND per-feature probe AUCs at each, and produces a
  rollup table. Surfaces whether capacity is the binding constraint
  for gate 7.3.

## Impact

- `scripts/cascade_host_capacity_sweep.py` (new) — the sweep driver.
- `runs/capacity_sweep/` (gitignored; the script writes summary.csv
  and summary.json plus per-config artefacts).
- Tests: lightweight integration test that the script runs end-to-end
  on a tiny 2-config grid (the smallest two cells) and produces a
  well-formed summary; the full 9-config measurement is the
  acceptance artefact.

No source-tree changes outside `scripts/` + `tests/`. The script
re-uses `train_cascade_host.train(...)`, `forge_pipeline.py`, and
`probe_full_gt_recoverability.py` as library functions.

## Risks

- **Compute headroom on Intel CPU**: the largest config (n_embd=192,
  n_layer=4) takes ~140s to train; the full sweep is ~16 min. Acceptable
  on this MBP but the user should expect to leave it running.
- **Mismatched-dim forge path**: configs with `n_embd > 61` rely on
  `SubspaceProjector`'s projection-down behaviour, which has been
  exercised by every previous forge run; no new code path needed.
- **The probe takes ~30s per host**: 9 hosts × 30s = ~5 min just for
  probes. The script can be made parallel later; for v1 ship it
  sequentially.
- **Probe `--from-projected` is required** for the per-particle
  spotlights (the SAE-encoded measurement), adding ~10s per host.

## Out of scope

- **Tuning** `n_inner`, `n_head`, or any other GPT-2 hyperparameter
  beyond `n_embd` and `n_layer`. The minimal 2D grid is the question
  this change is asking.
- **Comparing JumpReLU vs L1 vs TopK SAEs** — the forge measurement
  is fixed at `cascade__jumprelu` rung5 (the gate 7.3 cell).
- **Running on GPU/MPS** — Intel CPU is the target. If
  `n_embd=192, n_layer=4` becomes the binding configuration, a
  follow-up sweep at larger sizes on a GPU box would be the natural
  extension.
- **Sweep parallelism** — sequential train→forge→probe per config
  is fine for v1.

## Acceptance summary

This change ships when the sweep completes (gate C.1) AND either:

- Gate C.2 hits → `forge_delta_vs_random ≥ 0.05` at SOME capacity,
  closing gate 7.3 by capacity alone, OR
- Gate C.2 misses across the full 9-config grid AND Gate C.3 misses
  (per-particle AUC ≤ 0.92 even at the largest capacity), in which
  case the diagnosis pivots and the matching follow-up change is
  filed (`investigate-cascade-vocabulary-entropy` or
  `investigate-cascade-jumprelu-sparsity`).

Either outcome is informative. The capacity sweep is the
falsifiable test of the v1+probe+v2 arc's converged hypothesis.
