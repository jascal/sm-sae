# tasks — cascade-host-training-budget-sweep

## 1. Implementation

- [x] 1.1 `scripts/cascade_host_training_budget_sweep.py` — sweep driver
      at fixed L6 host, varying (n_trajectories, epochs) along
      [(2000, 5), (5000, 10), (10000, 20)].
- [x] 1.2 Reuses `train_cascade_host.train(...)` + `forge_pipeline.py`
      `--host-dir` + `probe_full_gt_recoverability.py` as library /
      subprocess calls.

## 2. Live sweep (this machine, 2026-05-20)

- [x] 2.1 Full sweep ran in 886.1s (14.8 min) on Intel CPU. Within
      the 25-min budget.

### Live results

| trajectories | epochs | steps | train_loss | forge_score | Δ_random | mean AUC | color:r AUC | wall |
|---|---|---|---|---|---|---|---|---|
| **2000** | **5** | **500** | 1.729 | 0.7549 | **+0.0239** ← peak | 0.915 | **0.877** ← peak | 36s |
| 5000 | 10 | 2480 | 1.164 | 0.7307 | −0.0003 | 0.916 | 0.829 | 138s |
| 10000 | 20 | 9980 | 1.205 | 0.7381 | +0.0071 | 0.909 | 0.759 | 591s |

Random forge baseline: 0.7310. state_t-direct color:r ceiling: 0.904.

## 3. Gate verdicts (live measurement)

- [x] **Gate B.1 (mechanical, ≤25 min)**: PASS — 14.8 min total.

- [x] **Gate B.2 (gate-7.3 closer, Δ_random ≥ +0.05)**: **FAIL** —
      best Δ across the budget sweep is still +0.0239 at the smallest
      budget (matching the empirical peak from PR #26). MORE training
      REGRESSES the forge score.

- [x] **Gate B.3 (entropy prediction — color:r ≥ 0.85)**: PASS at
      the smallest budget (color:r = 0.877). FAIL with budget scaling
      (color:r drops to 0.829 then 0.759 as budget scales).

## 4. Diagnosis (CORRECTS PR #28's diagnosis)

### PR #28's claim was wrong

PR #28 reported state_t-direct color:r = 0.904 vs LM-probe color:r =
0.740 → LM info drop of 0.164. **The 0.740 was the L2-host's probe
AUC** (from PR #22's baseline measurement), not the L6-host's. The
correct comparison:

| measurement | color:r AUC |
|---|---|
| state_t-direct (no LM) | 0.904 |
| **L6 host at (2000, 5)** | **0.877** (this sweep) |
| L2 host at (2000, 5) | 0.740 (PR #22) |
| LM info gap (L6 vs state_t) | **0.027** (not 0.164) |

The L6 host is ALREADY near-optimal at extracting state_t's
information. The 0.164 "LM info drop" was an artefact of mixing
hosts across measurements.

### What this implies

- The L6 host at the baseline budget is the empirical optimum for
  this substrate. More training, more aux supervision, more capacity
  — none of them help.
- The cascade benchmark as currently constructed has an inherent
  ceiling at Δ_random ≈ +0.024 (or absolute forge_score ≈ 0.76 on
  cascade__topk).
- The +0.05 gate target was **unrealistic for this substrate**.

### How the diagnosis compounded across the arc

| PR | Claimed diagnosis | What that claim missed |
|---|---|---|
| #19 v1 aux | "5 labels are at ceiling" | Didn't measure state_t-direct floor |
| #22 probe | "LM-only at 0.74-0.85" | Only L2 host probed |
| #23 v2 aux | "more labels regress" | Didn't yet rule out other levers |
| #25 capacity | "depth monotonic L2→L6" | Only 3 data points |
| #26 depth | "host-side exhausted" | Didn't compare to state_t-direct |
| #27 SAE swap | "vocab entropy next" | Didn't yet measure entropy |
| #28 entropy | "LM drops 0.164" | **Mixed hosts in comparison** |
| **THIS** | **+0.024 is the ceiling; gate target was wrong** | actual measurement closes the loop |

Each step's claim was the best inference at its data. Each next step
revealed a hidden assumption. The arc converged in 8 PRs — but only
because each measurement falsified the previous diagnosis cleanly.

## 5. Recommendation: re-frame gate 7.3

Per PR #27's recommendation 1, **adopt absolute forge_score as the
canonical sm-sae faithfulness metric**, not Δ_random:

- `cascade__topk` rung5 with the random-init host alone reaches
  absolute forge_score = **0.7580**.
- `cascade__jumprelu` rung5 with the L6 host reaches 0.7549.
- These are essentially indistinguishable; whichever absolute number
  downstream consumers actually need from the cascade benchmark
  should set the gate.

**Suggested replacement gate:** `absolute forge_score ≥ 0.80` (or
whatever threshold the downstream consumer requires). The Δ_random
≥ +0.05 target was geometry-of-Δ, not substrate-quality.

## 6. Archive trigger

This change SHALL be archived once a README update lands documenting
the cascade benchmark's actual ceiling + the recommended absolute-
score gate. That README PR is the natural session-closer for the
gate-7.3 lineage.

## 7. Note for the historical record

The 8-PR arc that produced this result:

PR #19 → v1 aux ships (Δ +0.0072)
PR #20 → probe-full-gt-recoverability filed
PR #22 → probe ships; L2 host's per-particle is 0.74-0.85
PR #23 → v2 aux ships (Δ −0.0053)
PR #24 → capacity sweep filed
PR #25 → capacity sweep ships; width saturates, depth monotonic 2→6
PR #26 → depth sweep ships; concave at L6, declared "host-side exhausted"
PR #27 → SAE family swap ships; gate framing misaligned
PR #28 → entropy measurement ships; reported LM info drop = 0.164
**THIS** → budget sweep falsifies #28's mismatch and surfaces the
           +0.024 ceiling. Closes the saga.

**Total session compute on Intel CPU**: ~50 min across 8 PRs to
diagnose a single benchmark gate. The methodology of iterative
falsifiable measurement remains the right shape — but THIS PR's
existence is itself the strongest evidence that "best diagnosis at
the available evidence" is necessary but not sufficient; each new
measurement can falsify the previous, and the process only converges
when no falsifiable test is left.

For future sm-sae substrate investigations: **always measure the
state_t-direct floor at the start, against the SAME host
configuration that you'll be evaluating downstream**. That single
measurement collapses the diagnostic time from 8 PRs to 1.
