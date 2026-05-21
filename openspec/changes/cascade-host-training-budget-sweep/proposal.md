# cascade-host-training-budget-sweep

## Why

Direct follow-up to `cascade-rollout-entropy-measurement` (PR #28).
That measurement closed the diagnostic loop on the gate-7.3 lineage
by showing the **cascade rollout is information-rich** (state_t-direct
mean AUC 0.923; 0% of features below 0.7) while the **LM at the prior
sweep's training budget drops 0.09-0.16 absolute AUC on per-particle
features** (color:r: state_t 0.904 → LM 0.740).

The previous arc (#19→#27) had ruled out aux supervision, width, depth,
and SAE family. PR #28 added the missing 13th measurement: state_t
directly. The host's *training regime* is the binding constraint —
not its capacity, not the supervision objective, not the SAE family.

This change runs the falsifiable test: fix the host at the empirical
depth peak (n_embd=61 / n_layer=6, from PR #26) and **scale the
training budget**. Predicted: color:r LM-probe AUC lifts from 0.74
toward the 0.90 state_t ceiling as gradient steps scale. Gate 7.3
(Δ_random ≥ +0.05 on cascade__jumprelu rung5) is reachable.

P1 — this is the predicted closer for an 8-PR diagnostic arc. Either
gate 7.3 hits at some cell, ending the saga, or the diagnosis
pivots one more time (e.g., to LM-architecture-level constraints).

## What Changes

### `scripts/cascade_host_training_budget_sweep.py` (new)

Train + forge + probe across a 3-cell budget grid at fixed L6 host:

| n_trajectories | epochs | est. steps | est. wall |
|---|---|---|---|
| 2000 | 5 | 500 | ~40s (baseline from PR #26) |
| 5000 | 10 | 1562 | ~150s |
| 10000 | 20 | 6250 | ~600s |

Plus probe (~30s each) + forge (~12s each) — total budget ≈ 15-18 min.

### Measurement output

`runs/budget_sweep/summary.{json,csv}` with per-cell rows:
forge_score, forge_delta_random, probe_mean_auc, probe_color_r,
probe_color_g, probe_color_b.

The headline metric is `forge_delta_random ≥ +0.05` AND the
per-feature `color:r` AUC trajectory: if the entropy-driven
hypothesis holds, color:r LM-probe AUC should lift toward 0.90
as the budget scales.

## Acceptance gates

### Gate B.1 (mechanical)

Total sweep wall ≤ 25 minutes on Intel CPU; all cells produce
forge + probe rows.

### Gate B.2 (gate-7.3 closer)

At least ONE cell achieves `forge_delta_random ≥ +0.05`. If yes,
**the gate-7.3 saga closes positively** after 8 PRs of diagnosis.
Smallest passing cell becomes the canonical sm-sae host config.

### Gate B.3 (entropy-prediction confirmation)

The largest-budget cell's `color:r` LM-probe AUC lifts ≥ 0.85.
This is the falsifiable consequence of PR #28's entropy diagnosis;
hitting it confirms the LM-training-regime hypothesis even if
gate 7.3 itself doesn't close.

## What this PR explicitly does NOT do

- **Hyperparameter tuning** beyond the (n_trajectories, epochs)
  grid. lr, batch size, n_inner, etc. stay fixed at the
  train_cascade_host defaults.
- **Increase the rollout vocabulary** or change `start_distribution`.
  The substrate is held fixed.
- **Per-cell aux supervision experiments.** All cells run pure LM-CE.
- **A 2D budget × capacity sweep.** Capacity is held at L6.

## Capabilities

### New Capabilities

- `cascade-host-budget-sweep`: a 3-cell training-budget sweep at
  fixed L6 host that produces forge + probe rows per budget cell.

## Acceptance

This change ships when:

1. The sweep completes (gate B.1) AND
2. Gate B.2 hits OR gate B.3 hits OR both miss with documented
   regression (which would falsify PR #28's entropy hypothesis).
3. The matching follow-up is filed per the bucket:
   - **Both B.2 and B.3 hit**: archive the gate-7.3 lineage;
     update README with canonical host config.
   - **B.3 hits but B.2 misses**: file
     `re-frame-gate-7-3-against-absolute-score` (gate 7.3's
     framing is the limitation, not the host training; see PR #27's
     recommendation 1).
   - **Both miss**: file `investigate-lm-architecture-for-cascade`.
     The information is in state_t but no autoregressive LM training
     regime can extract it; need a non-AR architecture or
     architectural priors tuned for cascade composition.

## Note for the historical record

The 8-PR session-arc that led here:

| PR | Finding |
|---|---|
| #19 | v1 aux Δ +0.0072 |
| #22 | probe: per-particle 0.74-0.85 |
| #23 | v2 aux Δ −0.0053 |
| #25 | width saturates 0.87; depth monotonic L2→L6 |
| #26 | depth peaks L6; declared host-side exhausted |
| #27 | SAE family binding; gate framing misaligned |
| #28 | state_t IS information-rich; LM drops signal |
| **THIS** | **falsifiable test of the LM-training-regime hypothesis** |

Iterative falsifiable measurement. Each PR's diagnosis was honest
at the evidence available; each next PR exposed an assumption the
previous one was making. The methodology produced the actual root
cause in one session.
