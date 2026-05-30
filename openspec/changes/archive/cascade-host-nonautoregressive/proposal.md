# cascade-host-nonautoregressive

> **RESOLVED (2026-05-30): option B was chosen.** Gate 7.3 was reframed to an
> absolute `forge_score ≥ 0.76` (see
> [`reframe-gate-7.3-absolute-forge-score`](../../reframe-gate-7.3-absolute-forge-score/)),
> which `cascade__topk` (0.760) meets — the saga closes positively. **Option A
> (build a non-autoregressive host), below, is therefore deferred — preserved
> as an *optional* future experiment**, not a blocker.

## Why

Terminal pivot for the gate-7.3 lineage. `cascade-host-training-budget-sweep`
(the live run, 2026-05-30) refuted PR #28's "train longer" prediction: holding
the architecture at the L6 depth peak and scaling the training budget drove
`color:r` residual-probe AUC *down* (0.877 → 0.814 → 0.779 along the
gradient-step axis) and forge Δ-vs-random *down* (best +0.0239 at the
lowest budget, ≈0 at E20/E40). Training CE fell the whole time
(1.73 → 1.46 → 0.98); per-particle faithfulness fell with it.

That is the decisive observation: **the autoregressive LM cross-entropy
objective is anti-aligned with gate 7.3.** Optimising it harder trades away the
per-particle (color) discriminative structure that the forge gate and the GT
probe both read. Across the lineage every host-side lever is now exhausted:

| lever | change | verdict |
|---|---|---|
| aux supervision (5 / 110 labels) | #19 / #23 | Δ +0.0072 / −0.0053 |
| host width | #25 | saturates ~0.87 by n_embd≈96 |
| host depth | #26 | peaks at L6; L8–12 regress |
| SAE family | #27 | family binds; random baseline already strong |
| vocabulary entropy | #28 | vocab IS rich; LM drops 0.09–0.16 AUC |
| **training budget** | **budget-sweep** | **more steps → lower Δ AND lower color:r** |

The one variable never changed across the whole arc is the **objective**:
causal, next-token LM cross-entropy. PR #28 already showed a permutation-
invariant logistic regression reads `color:r` from `state_t` at 0.904 in
seconds — a non-sequential classifier extracts exactly the signal the AR host
loses. This change tests whether the *objective/architecture* is the binding
constraint by swapping the causal AR host for a non-autoregressive one.

## What Changes

### A non-autoregressive cascade host

Add a non-AR host alongside the existing causal `tiny_gpt2` host, holding the
forge-facing interface fixed (residual-stream width = the cascade SAE's
`input_dim` = 61, saved under `<host-dir>/host/` exactly as forge reads today).
The variable under test is the **objective**, not the width or the data:

- **Architecture**: a bidirectional (non-causal) encoder over the `state_t`
  multiset — either a DeepSets / mean-pool MLP (permutation-invariant by
  construction, matching the PR #28 logistic-regression result) or a
  transformer encoder with the causal mask removed.
- **Objective**: masked / denoising particle modelling (predict held-out
  particles of `state_t` from the rest) instead of next-token CE — so the host
  is rewarded for representing *which particles are present*, which is what the
  GT probe and forge gate measure.

### Measurement (same harness as the budget sweep)

Reuse `cascade_host_capacity_sweep.py`'s train → forge → probe pipeline with a
`--host-kind {ar,nonar}` switch (default `ar` preserves all existing grids).
Report, against the AR baseline:

- forge Δ-vs-random on `cascade__jumprelu` rung5;
- `color:r` and spotlight-median residual-probe AUC.

## Acceptance gates

**Gate N.1 (mechanical)**: the non-AR host trains + forges + probes to
completion and writes a summary row comparable to the AR baseline.

**Gate N.2 (objective is the binding constraint)**: the non-AR host's forge
Δ-vs-random exceeds the AR host's best (+0.0239) **and** its `color:r` residual
probe AUC holds ≥ ~0.90 (the `state_t` ceiling) rather than degrading under
training. If yes → the AR objective was binding; gate 7.3 closes positively and
the non-AR host becomes the canonical cascade host.

**Gate N.3 (the gate metric itself is the problem)**: if the non-AR host *also*
fails to reach Δ ≥ +0.05, the Δ-vs-random metric — not any host — is the
obstacle. Adopt PR #27's already-surfaced reframe: gate 7.3 becomes an
**absolute** `forge_score` target (e.g. ≥ 0.80), measured on the strongest SAE
family, and the Δ-vs-random gate is retired. This sweep's negative result
(training that lowers CE lowers Δ) independently argues for it.

## Decision teed up for the maintainer

Two viable closes, and the budget-sweep result supports either:

1. **Build the non-AR host** (this proposal) — the principled test of whether
   the AR objective was the culprit. Higher effort.
2. **Re-frame gate 7.3 to absolute `forge_score`** (PR #27) — lower effort;
   stops measuring a quantity the AR objective structurally cannot supply.

Recommendation: do (2) first (a definition change + a README edit closes the
saga immediately), and treat (1) as the optional deeper experiment if a
non-AR host is independently wanted for the sae-forge world-model path.

## Capabilities

### New Capabilities

- `cascade-host-nonautoregressive`: a non-AR cascade host + a `--host-kind`
  switch on the sweep driver. (Proposal only — not implemented in this change.)

## Out of scope

- **Implementing the host.** This change files the experiment + the decision;
  implementation is a follow-up the maintainer authorises.
- **Retraining the cascade SAE.** The `cascade__jumprelu` rung5 cell stays the
  fixed forge substrate.
- **Aux supervision / capacity / depth / budget** — all exhausted upstream.

## Acceptance summary

This change ships when the experiment + the maintainer decision point are
recorded (this file). It is the documented close-out branch of
`cascade-host-training-budget-sweep` task 4.3.
</content>
