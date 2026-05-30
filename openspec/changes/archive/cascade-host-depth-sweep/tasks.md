# tasks — cascade-host-depth-sweep

## 1. Implementation

- [x] 1.1 Add `--grid {capacity,depth,custom}` flag to
      `scripts/cascade_host_capacity_sweep.py`. `capacity` preserves
      existing behaviour; `depth` selects the 4-config follow-up grid;
      `custom` accepts repeated `--config NE_L` flags.
- [x] 1.2 Add `DEPTH_GRID = [(61,6), (61,8), (61,10), (61,12)]` constant.

## 2. Live sweep (this machine, 2026-05-20)

- [x] 2.1 Full depth sweep ran in 321.2s (5.4 min) on Intel CPU. Wall
      well under the proposal's 25-min budget. Gate D.1 PASS.

### Depth-sweep results

| config | n_params | train_loss | forge_score | Δ_random | probe_mean | spotlight |
|---|---|---|---|---|---|---|
| NE61_L6 | 279k | 1.729 | 0.7549 | **+0.0239** ← peak | 0.915 | 0.859 |
| NE61_L8 | 369k | 1.798 | 0.7375 | +0.0065 | 0.915 | 0.840 |
| NE61_L10 | 460k | 1.730 | 0.7292 | −0.0018 | 0.914 | 0.826 |
| NE61_L12 | 551k | 1.796 | 0.7163 | −0.0147 | 0.911 | 0.817 |

### Combined trajectory (capacity sweep + depth sweep)

| n_layer | forge Δ_random | spotlight |
|---|---|---|
| 2 | −0.0048 | 0.799 |
| 4 | +0.0048 | 0.809 |
| **6** | **+0.0239** ← peak | **0.859** ← peak |
| 8 | +0.0065 | 0.840 |
| 10 | −0.0018 | 0.826 |
| 12 | −0.0147 | 0.817 |

## 3. Gate verdicts

- [x] **Gate D.1 (mechanical)**: **PASS** — 4 configs trained + forged +
      probed in 5.4 min.

- [x] **Gate D.2 (gate-7.3 closer)**: **FAIL** — best Δ across the full
      n_layer ∈ {2, 4, 6, 8, 10, 12} trajectory is still **+0.0239 at
      L6**, well short of the +0.05 target. Depth scaling
      **OVERSHOOTS** — L8/L10/L12 regress monotonically.

- [x] **Gate D.3 (trajectory characterisation)**: trajectory is
      **CONCAVE with peak at L6**, not monotonic as PR #25 had
      extrapolated. The L4→L6 doubling rate was 0.019; the L6→L8
      doubling rate was −0.017 (reversal). The PR #25 extrapolation
      assumed monotonic doubling and was empirically wrong.

## 4. Diagnosis (updated; load-bearing)

The full session-arc evidence now includes 4 negative results:

| lever | result |
|---|---|
| Aux supervision v1 (5 labels) | Δ +0.0072 (#19) |
| Aux supervision v2 (110 labels, focal-BCE) | Δ −0.0053 (#23) |
| Width scaling (n_embd ∈ {61, 96, 128, 192}) | spotlight saturates at 0.87 (#25) |
| **Depth scaling (n_layer ∈ {2, 4, 6, 8, 10, 12})** | **Δ peaks at L6 = +0.0239; L8-12 regress (this PR)** |

**The host is NOT the binding constraint for gate 7.3.** Every
host-side knob has been tried and capped. Training losses across all
12 configurations cluster in the **1.66-1.80 range** — the model is
hitting a similar loss floor regardless of capacity, suggesting the
bottleneck is OUTSIDE the host:

- The cascade SAE (`cascade__jumprelu`) may have limited capacity
  to carry per-particle / per-color signal regardless of host
  quality.
- The cascade rollout vocabulary may not have enough discriminative
  entropy to disambiguate the per-particle features at all.
- The `gt_alignment` faithfulness target may be measuring the wrong
  thing for this setup.

These are the unmeasured hypotheses; one or more is now the
prime suspect.

## 5. Recommendation for follow-up

**File `investigate-cascade-jumprelu-sparsity-and-vocab-entropy`**
as the joint follow-up:

A. **JumpReLU SAE replacement test**: re-run the L6 host
   (the empirical peak) through `cascade__topk` and
   `cascade__l1` SAEs (already present in `runs/`). Same host,
   different SAE family. If forge_delta varies materially
   (≥ ±0.02) across SAE families, the SAE is binding. If not,
   the SAE family is irrelevant and the diagnosis pivots to
   vocabulary entropy.

B. **Vocabulary entropy measurement**: compute per-feature
   discriminative entropy of the 110-feature GT vocabulary
   against the cascade rollout distribution. If the per-particle
   features have very low marginal entropy in the rollout (i.e.
   ~always-on or ~always-off across samples), the LM objective
   physically cannot encode them differently — and no host
   capacity sweep will help.

Compute: A is ~5 min (one host + 2 forge runs); B is ~30s
(probe-set summary statistic).

If A shows SAE-family dependence, file `add-cascade-sae-family-comparison`
proper. If B shows low marginal entropy, file
`enrich-cascade-rollout-vocabulary`.

**Stop trying host-side levers.** The empirical evidence ruled them
all out across PRs #19 / #23 / #25 / THIS.

## 6. Archive trigger

This change SHALL be archived once
`investigate-cascade-jumprelu-sparsity-and-vocab-entropy` is filed.

## 7. Notes for the historical record

The session-arc that produced this result:

  PR #19 → v1 aux (5 labels) ships, gate 7.3 missed at +0.0072
  PR #20 → probe-full-gt-recoverability filed
  PR #22 → probe shipped; diagnoses LM-only host carries per-
           particle at 0.74-0.85, SAE preserves but doesn't sharpen
  PR #23 → v2 aux (110 labels) ships; gate 7.3 regresses to −0.0053
           — more labels make it worse
  PR #24 → capacity-sweep proposal filed
  PR #25 → capacity sweep ships; depth scales monotonically L2→L4→L6
           — but only 3 data points
  PR #26 → THIS — depth sweep at L6,8,10,12 reveals the trajectory
           is concave with peak at L6; gate 7.3 confirmed
           UNREACHABLE via host-side levers

Each PR's conclusion was empirically grounded; the conclusion
shifted with new data. The capacity-sweep extrapolation was the
honest conclusion at 3 data points; this PR's 4 additional data
points falsified it. **This is how the project should investigate
load-bearing questions: iteratively, with falsifiable gates at
each step.**
