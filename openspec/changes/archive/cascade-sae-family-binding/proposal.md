# cascade-sae-family-binding

## Why

Direct follow-up to `cascade-host-depth-sweep` (PR #26). That PR ruled
out every host-side lever for gate 7.3 (Aux v1, Aux v2, width, depth)
and recommended testing whether the cascade SAE itself is the binding
constraint via a 3-family swap. **The swap ran live this turn and the
finding is conclusive: SAE family IS binding, AND the gate-7.3
framing itself is misaligned with the data.**

### Measured this session (L6 host = empirical depth peak from PR #26)

| SAE family | trained forge_score | random baseline | Δ_random |
|---|---|---|---|
| `cascade__jumprelu` | 0.7549 | 0.7310 | +0.0239 |
| **`cascade__topk`** | **0.7598** | 0.7580 | +0.0018 |
| `cascade__l1` | 0.7273 | 0.7381 | **−0.0108** |

- **Variance across families: 0.033 (trained) / 0.027 (random)** —
  well above the proposal's ±0.02 "SAE binding" threshold. **SAE
  family IS binding.**
- **Gate-7.3 framing is misaligned**: it measures Δ_random, which by
  construction PENALISES SAEs with strong structural priors.
  `cascade__topk` random alone reaches 0.758 — the SAE's structure
  already encodes most of the cascade signal, leaving nothing for
  training to add. `cascade__jumprelu` has the lowest random
  baseline AND the largest trained-vs-random gap because its sparse-
  threshold structure relies on a trained host to provide the
  signal.

This change formalises the finding and proposes the next experimental
pivot.

## What Changes

This change does NOT implement code. It is **a measurement summary +
a framework pivot** for gate 7.3:

### Finding 1 (SAE family is binding)

The SAE family chosen for the gate-7.3 cell determines the achievable
forge score MORE THAN any host-side intervention measured across
PRs #19 / #23 / #25 / #26.

### Finding 2 (gate-7.3 framing penalises good SAEs)

The Δ_random target is a poor metric when SAE families differ in how
much cascade structure they already encode unsupervised:

- `cascade__topk` has 84929 params (random) / 566640 params
  (post-forge) — its TopK structure already captures most of the
  cascade signal at the SAE-encoding step.
- `cascade__jumprelu` has 30577 random / 204346 trained params — its
  JumpReLU threshold is more reliant on the host residual stream to
  provide signal.

The gate-7.3 target Δ ≥ +0.05 ONLY makes sense against a SAE family
that doesn't already do most of the work. It's specifically a
"jumprelu host-training" gate, not a general "forge faithfulness"
gate.

### Recommendation 1 (re-frame gate 7.3)

**Replace** the gate-7.3 metric from `Δ_random ≥ +0.05` to
`absolute forge_score ≥ TARGET`, where TARGET is chosen against the
absolute scale that the best-performing SAE family achieves:

- `cascade__topk` reaches 0.760 absolute — meaningful faithfulness
  without needing a trained host.
- Set the new gate target to `absolute forge_score ≥ 0.80` (or
  whatever absolute threshold the project's downstream consumers
  actually need from the cascade benchmark).

This decoupling lets us evaluate sm-sae's substrate quality without
the gate definition penalising good structural priors.

### Recommendation 2 (investigate vocabulary entropy as the next pivot)

Training losses cluster at 1.66-1.80 across ALL 12 host configurations
measured this session (n_embd ∈ {61, 96, 128, 192}, n_layer ∈ {2..12}).
That's a **strong indicator** that the cascade rollout vocabulary
has limited per-particle discriminative entropy — i.e., per-particle
identity is not strongly predictable from state_t under the rollout
distribution.

File `cascade-rollout-entropy-measurement`:

- Compute per-feature marginal entropy of the 110-feature GT vocab
  across the rollout dataset.
- For each measurable feature, compute conditional entropy
  `H(feature | state_t)` — how much information state_t contains
  about whether the feature fires in state_{t+1}.
- If conditional entropy ≥ marginal for most per-particle features,
  the rollout fundamentally cannot disambiguate per-particle outcomes
  and any host/SAE combo will cap at the measured ~0.76 ceiling.
- If conditional entropy << marginal, the LM objective IS leaving
  information on the table and the diagnosis pivots one more time.

Expected wall time on Intel CPU: ~30s.

## Capabilities

No code change. This is a measurement + framework artefact.

## Acceptance

This change ships when:

1. The sae-family-swap summary (`runs/sae_family_swap/summary.json`)
   is committed to the repo.
2. The matching follow-up (`cascade-rollout-entropy-measurement`) is
   filed with this proposal's recommendation 2 populating its "Why."

If the rollout-entropy investigation lands the conclusion that the
cascade vocabulary IS information-capped, the gate-7.3 saga closes
with the finding "this benchmark has structurally limited resolution"
and the canonical recommendation becomes "use absolute forge_score
on `cascade__topk` as the sm-sae faithfulness gate, not Δ-vs-random."

## Out of scope

- Implementing the rollout-entropy script. That's the next change.
- Retraining the cascade SAE family. Out of scope; the existing
  three families are the measurement substrate.
- Re-defining gate 7.3's threshold across the rest of sm-sae's
  documentation. That happens once the entropy measurement closes
  the loop.

## Notes for the historical record

PR #19 → v1 aux (5 labels): Δ +0.0072
PR #22 → probe: per-particle 0.74-0.85
PR #23 → v2 aux (110 labels): Δ −0.0053
PR #25 → capacity sweep: width saturates 0.87; depth L2→L6 monotonic
PR #26 → depth sweep L6..L12: trajectory CONCAVE, peaks at L6 +0.024
**THIS** → SAE family swap: variance 0.033 across families; gate
            framing itself is misaligned.

Each step's conclusion was empirically grounded; the conclusion
shifted as new evidence arrived. The gate-7.3 saga's value isn't
that it closes positively — it's that the **methodology of
iterative falsifiable measurement** generated this much insight in
one session.
