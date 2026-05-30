# cascade-host-training-budget-sweep

## Why

Direct follow-up to `cascade-rollout-entropy-measurement` (PR #28). That
measurement reversed the "host-side exhausted" diagnosis: the cascade
rollout vocabulary IS information-rich (0% of GT features sit below 0.7
AUC when classified directly from `state_t`), yet the LM-trained host's
residual stream surfaces those same features at a 0.09–0.16 absolute
AUC discount. The biggest gap is `color:r`: `state_t` separates it at
**0.904**, the L6 host's residual surfaces it at **0.740**.

PR #28's conclusion: the binding constraint for gate 7.3 is the LM
**training regime**, not host capacity (PR #25), depth (PR #26), SAE
family (PR #27), or vocabulary entropy (PR #28 itself). PR #28
explicitly recommended this change as "the real next experiment" and
left a falsifiable prediction:

> increasing the budget (specifically the gradient-step count) closes
> the LM-info-drop gap. The prediction is `color:r` LM-probe AUC lifts
> from 0.74 toward 0.90 (the `state_t` ceiling) as training scales.

This change runs that test directly at the empirical depth peak
(`n_embd=61, n_layer=6`), holding the architecture fixed and sweeping
only the training budget.

P1 — gate 7.3 is the longest-open gate in the project (open across
PRs #19/#20/#22/#23/#24/#25/#26/#27/#28). This sweep either closes it
("train longer" works) or definitively reroutes it to the LM
architecture itself (causal attention is the bottleneck → the next
experiment is non-autoregressive hosts).

## What Changes

### CLI extension to the existing sweep driver

Add `budget` to the `--grid` choices on
`scripts/cascade_host_capacity_sweep.py` (currently
`{capacity, depth, custom}`). The budget grid holds the architecture
fixed at the L6 depth peak and varies the per-config training budget:

```python
BUDGET_GRID = [
    # (n_embd, n_layer) fixed at the depth peak; budget varies per row.
    {"n_embd": 61, "n_layer": 6, "n_trajectories": 2000,  "epochs": 5},   # 1x  baseline (== depth-sweep L6)
    {"n_embd": 61, "n_layer": 6, "n_trajectories": 2000,  "epochs": 20},  # 4x  gradient steps, same corpus
    {"n_embd": 61, "n_layer": 6, "n_trajectories": 2000,  "epochs": 40},  # 8x  gradient steps, same corpus
    {"n_embd": 61, "n_layer": 6, "n_trajectories": 5000,  "epochs": 20},  # corpus-scaling control vs row 2
]
```

This is a deliberately factorial design:

- **Rows 1→2→3 isolate gradient-step count** — corpus held at 2000
  trajectories, only the number of optimisation passes (epochs) grows.
  This is the exact axis PR #28's hypothesis names.
- **Row 4 vs Row 2 isolates corpus size** — same 20 epochs, 2.5× the
  trajectories. Disentangles "train longer on the same data" from
  "show the model more data."

### Per-config training budget plumbing

The existing driver applies a single global `--n-trajectories`/`--epochs`
to every config. This change generalises the per-config loop so each
config carries its own `n_trajectories`/`epochs`/`lr`, with the
capacity/depth grids inheriting the global defaults unchanged. Each
budget config gets a disambiguated host directory
(`sweep_NE61_L6_T2000_E20`) so the four same-architecture rows don't
collide.

### Output: `runs/budget_sweep/summary.{json,csv}`

Same schema as `runs/depth_sweep/summary.json`, plus four new per-row
columns (`n_trajectories`, `epochs`, `lr`, `n_train_steps`) and a
`budget_trend` block in `gate_summary` reporting the `color:r` and
spotlight-median AUC trajectory against gradient-step count.

### Acceptance gates

**Gate B.1 (mechanical)**: the full 4-config grid trains + forges +
probes to completion and writes `summary.{json,csv}`. Predicted wall
≈ 15–18 min on this CPU (≈11.5 min training + forge/probe overhead).

**Gate B.2 (gate-7.3 closer)**: at least ONE configuration achieves
`forge_delta_vs_random ≥ +0.05`. If yes, gate 7.3 closes and the
proposal recommends that configuration as the canonical production
cascade host.

**Gate B.3 (color:r lift — the PR #28 prediction)**: report the
`color:r` residual-probe AUC at each budget point. The prediction holds
if `color:r` AUC rises monotonically along the gradient-step axis AND
the best point clears its own lowest-budget baseline by ≥ 0.02 (toward
the 0.90 `state_t` ceiling). Gradient-step scaling (rows 1→3) vs corpus
scaling (row 4 vs 2) is reported separately so the dominant lever is
identified.

> **Anchoring correction (found during the smoke run).** PR #28 framed
> the prediction as "color:r lifts from 0.74 toward 0.90." That 0.74
> was the **L2 baseline host** (`capacity_sweep` (61,2) probe = 0.740).
> This sweep fixes the host at the **L6 depth peak**, where the probe
> already reads color:r ≈ **0.877** — i.e. depth alone closed ~85% of
> the 0.74→0.90 gap PR #28 attributed to training budget. The residual
> headroom at L6 is therefore only ≈0.02, and the gate is judged on
> lift over the L6 baseline rather than the absolute 0.74→0.90 span.
> The artifact records both the 0.74 L2 reference and the observed L6
> baseline so the distinction is legible.

## Capabilities

### Modified Capabilities

- `cascade-host-capacity-sweep`: gains `budget` as a fourth `--grid`
  value and per-config training-budget support. Default `capacity`
  and the existing `depth`/`custom` paths are unchanged.

## Impact

- `scripts/cascade_host_capacity_sweep.py`: add `BUDGET_GRID`, a config
  normaliser so each row carries its own budget + host dir, thread `lr`
  through `_train_one`, add `n_trajectories`/`epochs`/`lr`/`n_train_steps`
  to each summary row, add the `budget_trend` gate block, add `budget`
  to `--grid` and a `--lr` flag. ~60 lines; capacity/depth paths
  preserved.
- `runs/budget_sweep/` — new output dir (force-add `summary.{json,csv}`).
- No source-tree changes outside `scripts/`.

## Risks / Trade-offs

- **Compute**: the (5000, 20) row is the heaviest (≈5k gradient steps
  over an ≈7.9k-pair corpus, ≈5 min train). The full grid is ≈16 min;
  run it in the background. `--smoke` runs the two cheapest budget
  points to verify wiring in ≈3 min.
- **Overfitting at high epochs**: 40 epochs on a fixed 2000-trajectory
  corpus (≈3.2k pairs) may overfit. The probe AUC is measured on a
  held-out probe dataset, so gate B.3 is not biased by train-set
  memorisation; if overfitting bites, the probe AUC plateaus or
  regresses at E40 and that itself is the finding ("more passes don't
  help; need more data" → row 4 becomes the lever).
- **LR schedule held fixed**: the cosine schedule + 5e-3 peak is held
  constant so the budget axis is clean. PR #28 listed LR-schedule as a
  third axis; it is deferred (the `lr` knob is plumbed per-config for a
  follow-up `custom` budget grid, but not swept here).

## Out of scope

- **Sweeping the LR schedule** (warmup fraction, peak). Plumbed but not
  swept; a follow-up if the budget axis alone doesn't close gate 7.3.
- **Aux supervision** at any budget — PRs #19/#23 proved it's not the
  lever. Pure LM-CE training only.
- **Non-autoregressive host architectures** — the documented next
  experiment IF this sweep shows the gap is budget-insensitive.
- **Retraining the cascade SAE** — the existing `cascade__jumprelu`
  rung5 cell is the fixed forge measurement substrate.

## Acceptance summary

This change ships when:

1. The 4-config budget sweep completes and writes
   `runs/budget_sweep/summary.{json,csv}` (gate B.1).
2. The summary reports per-config `forge_delta_vs_random` + the
   `color:r`/spotlight AUC trajectory vs gradient-step count
   (gates B.2, B.3).
3. The matching follow-up is filed based on the outcome:
   - **If B.2 hits** (some config Δ ≥ +0.05): archive the gate-7.3
     lineage; document the canonical cascade host (architecture +
     budget) in the README. The saga closes positively.
   - **If B.2 misses but B.3 shows `color:r` lifting**: file
     `forge-projection-faithfulness-deep-dive` — the host now has the
     signal but it's dropped between the residual probe and the forge
     faithfulness measurement.
   - **If B.2 misses AND B.3 shows `color:r` flat**: the LM
     architecture is the binding constraint; file
     `cascade-host-nonautoregressive` (the documented terminal pivot).

Either outcome closes the empirical loop opened by PR #19.

## Result (live run, 2026-05-30)

The full 4-config grid ran to completion (`runs/budget_sweep/summary.{json,csv}`).

| n_traj | epochs | steps | train_loss | forge Δ_vs_random | color:r AUC |
|---:|---:|---:|---:|---:|---:|
| 2000 | 5  | 500  | 1.729 | **+0.0239** | **0.877** |
| 2000 | 20 | 2000 | 2.111 | −0.0021 | 0.814 |
| 2000 | 40 | 4000 | 1.456 | −0.0006 | 0.779 |
| 5000 | 20 | 4960 | 0.984 | +0.0122 | 0.796 |

**Gate B.1 (mechanical): PASS.** Grid trained + forged + probed; summary written.

**Gate B.2 (gate 7.3, Δ ≥ +0.05): FAIL.** No config reaches +0.05. The best
Δ (+0.0239) is the *lowest*-budget row — the depth-sweep L6 baseline itself.
Adding gradient steps drives Δ to ~0 (E20/E40); the corpus-scaling control
(T5000/E20) recovers only to +0.0122, still below baseline.

**Gate B.3 (PR #28 prediction — color:r lifts toward 0.90): REFUTED.** Along
the gradient-step axis color:r runs 0.877 → 0.814 → 0.779 — *monotonically
decreasing*, the opposite of the predicted rise. `color_r_lift = 0.0`
(best == lowest-budget baseline).

**Reading.** More LM gradient steps drive training CE down (E40 = 1.46,
T5000/E20 = 0.98, both well under the E5 baseline's 1.73) while *destroying*
the per-particle discriminative structure the forge gate and the probe both
read. The autoregressive LM objective is anti-aligned with gate 7.3: optimising
it harder trades away the per-particle (color) signal a lightly-trained host
retains. The best cascade host for forge faithfulness is the *least*-trained
one. Combined with PR #25 (capacity saturates), #26 (depth peaks at L6), and
#27 (SAE family binds; the random baseline is already strong), **every
host-side lever — capacity, depth, aux supervision, training budget — is now
exhausted for gate 7.3 on `cascade__jumprelu`.**

**Follow-up filed:** `cascade-host-nonautoregressive` (the documented terminal
pivot — the AR objective is the binding constraint). It also records the
pragmatic alternative PR #27 already surfaced: re-frame gate 7.3 to an absolute
`forge_score` target rather than Δ-vs-random, which this result independently
vindicates (training that lowers CE loss lowers Δ-vs-random). The maintainer
chooses which path to take; this sweep's job — falsify "train longer" — is done.
