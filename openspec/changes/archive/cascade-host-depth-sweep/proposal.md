# cascade-host-depth-sweep

## Why

This is the direct follow-up to `investigate-cascade-host-capacity-sweep`
(PR #25, merged this session). That sweep ran live on 6 configs and
surfaced TWO crystallized findings about gate-7.3's binding axis:

1. **DEPTH at fixed n_embd=61 scales the forge faithfulness gate
   monotonically.** Each doubling of depth roughly doubles the
   trained-vs-random Δ:

   | n_embd | n_layer | forge Δ_random |
   |---|---|---|
   | 61 | 2 | −0.0048 |
   | 61 | 4 | +0.0048 (+0.0096 from L2) |
   | 61 | 6 | +0.0239 (+0.0191 from L4) |

2. **WIDTH saturates by n_embd ≈ 96.** Going from n_embd=96 to 192
   only buys +0.003 spotlight AUC for 4× the parameter count;
   forge measurement isn't even possible at width != 61 because
   the cascade SAE has input_dim=61.

Extrapolating the depth trajectory: L8 ≈ +0.035, L10 ≈ +0.045-0.055.
**L10 is the predicted transition point** for closing gate 7.3
(Δ ≥ +0.05).

This change runs the falsifiable test directly: sweep
`n_layer ∈ {6, 8, 10, 12}` at fixed `n_embd=61`, no aux supervision.
Either gate 7.3 hits at some depth — closing the lineage that's been
open across PRs #19/#20/#22/#23/#24/#25 — OR the depth trend saturates
between L6 and L12 and the diagnosis pivots one more time.

P1 — gate 7.3 has been the longest-open gate in the project. This
change's outcome (within a single ~15-min compute window) either ends
the saga or definitively reroutes it.

## What Changes

### CLI extension to existing sweep driver

Add `--grid {capacity,depth,custom}` to `scripts/cascade_host_capacity_sweep.py`:

- `capacity` (default; preserves existing behaviour) — the 6-config grid
  from PR #25.
- `depth` (new) — the 4-config grid for this experiment:
  ```python
  DEPTH_GRID = [(61, 6), (61, 8), (61, 10), (61, 12)]
  ```
- `custom` — accepts an explicit list via repeated `--config NE_L` flags
  (e.g. `--config 61_8 --config 96_4`). For exploratory follow-ups
  without script edits.

The new grid reuses the existing sweep machinery byte-for-byte (train +
forge + probe + summary JSON). Only the configuration list changes.

### Output: `runs/depth_sweep/summary.{json,csv}`

The driver writes to `--out runs/depth_sweep` by default when
`--grid=depth` is selected. Same schema as `runs/capacity_sweep/summary.json`
so downstream analysis tooling is identical.

### Acceptance gates

**Gate D.1 (mechanical)**: full 4-config grid trains + forges + probes
in ≤ 25 minutes on Intel CPU. Predicted: ~15-20 min based on L6 ≈ 50s
and each additional layer adding ~10s + forge ~12s + probe ~14s.

**Gate D.2 (gate-7.3 closer)**: at least ONE configuration achieves
`forge_delta_vs_random ≥ +0.05`. If yes, gate 7.3 closes and the
proposal recommends that configuration as the canonical production
cascade host.

**Gate D.3 (depth-trajectory characterisation)**: the per-doubling
slope (Δ at L6, L8, L10, L12) is reported. If the trajectory is
sub-linear (Δ saturates), the diagnosis pivots; if it's super-linear,
the canonical host is the smallest config that meets gate D.2.

## Capabilities

### Modified Capabilities

- `cascade-host-capacity-sweep`: gains `--grid {capacity, depth, custom}`
  CLI flag. Default `capacity` preserves existing behaviour.

## Impact

- `scripts/cascade_host_capacity_sweep.py`: add `--grid` arg + the
  `DEPTH_GRID` constant + custom-list parsing for `--config NE_L`
  repeated flag. ~30 lines.
- `runs/depth_sweep/` — output dir (force-add the summary.{json,csv}).
- No source-tree changes outside `scripts/`.

## Risks / Trade-offs

- **L12 training time**: linear extrapolation from L6 (50s) suggests
  L12 ≈ 100s. With probe + forge overhead, the L12 config alone is
  ~125s. The full 4-config grid is ~5-6 min training + ~4-5 min
  forge + ~3 min probe ≈ **13-14 min total**. Well inside the 25-min
  budget.
- **Probe sample size**: 5000 trajectories per config gives 7932
  probe samples. Adequate for the 74 measurable GT features. No
  change from PR #25.
- **L12 overfitting**: 5 epochs at 500 steps on 3171 transition pairs
  may overfit at L12 (n_params ≈ 1.2M for ~3K samples). The probe
  AUC tracks held-out test set so this won't bias the gate D.2
  measurement; if it bites, L8/L10 are the actual targets anyway.

## Out of scope

- **Training-data scaling** alongside depth. The 2000-trajectory
  budget is held fixed across all configs for clean A/B against the
  PR #25 baseline.
- **Aux supervision** at any depth — the prior arc proved it's not
  the binding lever. Pure LM-CE training only.
- **Width × depth combined** sweeps. The PR #25 result showed width
  saturates by n_embd=96; combining doesn't add information until
  the depth axis is fully characterised first.
- **Retraining cascade SAE** at different input_dim. Out of scope;
  the existing SAE is the forge measurement substrate.

## Acceptance summary

This change ships when:

1. The 4-config sweep completes (gate D.1) AND
2. The summary.json reports per-config forge_delta_vs_random + probe
   spotlight medians AND
3. The matching follow-up is filed based on gate D.2's outcome:
   - **If D.2 hits**: archive the gate-7.3 lineage; document the
     canonical cascade host configuration in the README.
   - **If D.2 misses across all 4 configs**: file
     `investigate-cascade-jumprelu-sparsity` or
     `investigate-cascade-vocabulary-entropy` per the diagnostic
     framework already in `richer-cascade-host-supervision-v2`'s
     archive.

Either outcome closes the empirical loop opened by PR #19.
