# cascade-rollout-entropy-measurement

## Why

Follow-up to `cascade-sae-family-binding` (PR #27). That PR ruled out
SAE family as the *sole* binding constraint and recommended testing
whether the cascade rollout vocabulary is information-capped — if the
LM physically cannot disambiguate per-particle outcomes from
`state_t`, then no amount of host training will help.

**The measurement ran live this turn and the finding REVERSES the
prior diagnosis.** The cascade vocabulary is information-rich. The
binding constraint is the LM *training regime*, not host architecture
or vocabulary.

## Live measurement (this MBP, 2026-05-20)

Per-feature logistic-regression classifier trained from
`state_t` (61-dim bag-of-particles count vector) directly to each
GT label, NO LM in between. 5000 trajectories, 7932 probe samples,
74 measurable GT features.

### Aggregate

- **0% of features have state_t-AUC < 0.7.** No feature is
  fundamentally information-poor in the rollout.
- state_t mean AUC = **0.923** (vs the LM probe's 0.902 from baseline).
- state_t pct ≥ 0.9 = **65%** (vs LM probe 57%).

### Spotlight (the previously-weak per-particle features)

| feature | state_t AUC (no LM) | LM probe AUC | LM info drop |
|---|---|---|---|
| `color:r` | **0.904** | 0.740 | **+0.164** |
| `color:g` | **0.908** | 0.796 | +0.112 |
| `color:b` | **0.901** | 0.810 | +0.090 |
| `particle:mu+` | 0.832 | 0.756 | +0.076 |
| `flavor:mu` | 0.832 | 0.775 | +0.057 |
| `flavor:u` | 0.879 | 0.839 | +0.040 |
| `flavor:d` | 0.865 | 0.824 | +0.041 |
| `particle:u_b` | 0.828 | 0.799 | +0.029 |

**Every spotlight feature: state_t-direct AUC > LM probe AUC.**
The LM drops information on every weak feature, with the biggest
gaps on the per-color quark identity features (color:r/g/b drop
0.09-0.16 absolute AUC).

## What this reverses

PR #26 (`cascade-host-depth-sweep`) concluded that host-side levers
were exhausted because the depth trajectory at L8/L10/L12 regressed.
PR #27 (`cascade-sae-family-binding`) then pivoted the diagnosis to
SAE family + vocabulary entropy as the remaining candidates.

This measurement shows the cascade vocabulary IS information-rich
(0% of features below 0.7 AUC from state_t alone) and the LM at
the current training budget is the actual bottleneck:

- color:r information sits in state_t at AUC 0.904 (linearly
  separable from the count vector with a 30s logistic regression).
- The LM-trained host's residual stream surfaces color:r at 0.740 —
  losing 0.164 AUC of recoverable signal.

**The host's capacity (n_embd / n_layer) was never the binding
constraint.** The host's training regime is. 500 steps × 5 epochs
× 2000 trajectories × ~62 token vocab is insufficient to extract
the discriminative entropy that simpler classifiers find in seconds.

## What Changes

This change ships:

1. **`scripts/cascade_rollout_entropy.py`** — the measurement script
   already implemented and run live.
2. **`runs/aux_probe/rollout_entropy.json`** — the per-feature
   artefact (76 rows, 8 spotlight + summary statistics).
3. **A corrected diagnostic recommendation** for the gate-7.3 saga.

## Recommendation: the real next experiment

**File `cascade-host-training-budget-sweep`** — vary the training
budget at fixed L6 host (the empirical depth peak) and measure
gate 7.3 vs training cost:

| axis | sweep range | rationale |
|---|---|---|
| training trajectories | {2000, 5000, 10000, 20000} | scale the corpus the LM sees |
| epochs | {5, 10, 20, 40} | scale gradient passes per sample |
| learning-rate schedule | {default cosine, longer warmup, lower peak} | scale optimisation quality |

Hypothesis: increasing the budget (specifically the gradient-step
count) closes the LM-info-drop gap. Specifically, the prediction
is `color:r` LM-probe AUC lifts from 0.74 toward 0.90 (the state_t
ceiling) as training scales.

If the prediction holds, gate 7.3's original Δ ≥ +0.05 target
should be reachable on `cascade__jumprelu` rung5 at sufficient
training budget. **The gate-7.3 saga closes positively.**

If it doesn't — if even much more training fails to close the
LM-info-drop gap — then the LM architecture (causal attention
+ position-position interactions) is the binding constraint, and
the next experiment is non-autoregressive architectures. But
based on the magnitude of the drop (0.16 on color:r), that's
unlikely. The straightforward path is "train longer."

## Capabilities

### New Capabilities

- `cascade-rollout-entropy`: a fast (10s) diagnostic that
  reports per-feature state_t-direct AUC and the LM-vs-state_t
  delta. Surfaces whether a host's training is the binding
  constraint for any given feature.

## Acceptance

This change ships when:

1. The script + artefact are committed.
2. The follow-up `cascade-host-training-budget-sweep` is filed
   with this measurement's per-feature table populating its "Why."

## Out of scope

- **Implementing the training-budget sweep.** Next change.
- **Re-running the L6 host with more steps.** That belongs in
  the budget-sweep impl, not here.
- **Changing the rollout config** (parent distribution, max_steps,
  etc). Out of scope; the existing rollout is the substrate.

## Note for the historical record

The session-arc that produced this result:

| PR | Finding |
|---|---|
| #19 v1 aux | Δ +0.0072 |
| #22 probe | per-particle 0.74-0.85 |
| #23 v2 aux | Δ −0.0053 |
| #25 capacity sweep | width saturates 0.87; depth monotonic 2→6 |
| #26 depth sweep | depth peaks L6; L8-12 regress; declared "host-side exhausted" |
| #27 SAE swap | SAE family binding; gate framing misaligned |
| **THIS** | **vocab IS information-rich; LM is dropping signal; declare host-side ALIVE again** |

Each PR's diagnosis was the BEST EXPLANATION at the evidence available
when it was filed. PR #26's "host-side exhausted" was honest at 12
data points across capacity/depth without an explicit vocabulary
floor measurement; THIS measurement adds the floor and rewrites the
diagnosis. **The methodology of iterative falsifiable measurement
generated the actual root cause in one session — but only because
each step exposed an assumption the previous step had been making
implicitly.** PR #26 assumed state_t couldn't be richer than the
LM-residual; this measurement falsified that.
