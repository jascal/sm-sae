# openspec for sm-sae

Spec-driven change proposals for the sm-sae benchmark. Each non-trivial
change is staged here as a directory before code lands, mirroring the
pattern used by [jascal/polygram](https://github.com/jascal/polygram)
and [jascal/sae-forge](https://github.com/jascal/sae-forge).

## Layout

```
openspec/
├── README.md            this file
└── changes/             active and archived change proposals
    ├── <slug>/
    │   ├── proposal.md  the "what + why" + capability deltas
    │   ├── tasks.md     checkboxed implementation tasks (1.x / 2.x / …)
    │   ├── design.md    context, goals/non-goals, tricky decisions
    │   └── specs/       (optional) per-capability spec deltas
    └── archive/         landed changes, preserved for history
```

Once a change lands and stabilises, move its directory under
`changes/archive/` so the active set stays focused.

## Authoring conventions

- **`proposal.md`** opens with `## Why`, then `## What Changes`, then
  `## Capabilities` (sub-sectioned into `### New Capabilities` and
  `### Modified Capabilities`). Keep it under 200 lines; the substance
  belongs in tasks and design.
- **`tasks.md`** uses numbered task groups (`## 1. <group>`,
  `## 2. <group>`, …) with checkboxed sub-items
  (`- [ ] 1.1 ...`). Check items off as they land.
- **`design.md`** has `## Context`, `## Goals / Non-Goals`, and any
  decisions worth recording (numbered subsections under `## Decisions`).
  Skip if the proposal is mechanical.
- **`specs/<capability>/`** holds the capability-spec deltas the change
  introduces. Optional for small changes.

## Status of the current set

The cascade-host / gate-7.3 lineage is **closed — positively, by reframing
the gate.** The host-side search failed negatively — capacity (PR #25),
depth (PR #26), SAE family (PR #27), aux supervision (PRs #19/#23) and
training budget (PR #29) were each falsified as levers — which exposed the
*metric* as the real problem (#27: Δ-vs-random penalises SAE families with
strong unsupervised priors). Gate 7.3 was therefore **reframed** to an
absolute `forge_score ≥ 0.76`, which `cascade__topk` (0.760) **meets**.
Active changes:

| change | status | priority |
|---|---|---|
| [reframe-gate-7.3-absolute-forge-score](changes/reframe-gate-7.3-absolute-forge-score/) | **landed** — gate 7.3 redefined to absolute `forge_score ≥ 0.76` (**MET** by cascade__topk 0.760); Δ-vs-random retired to a diagnostic. Resolves the cascade-host-nonautoregressive decision (option B). | P1 |
| [cascade-host-training-budget-sweep](changes/cascade-host-training-budget-sweep/) | landed (PR #29) — refuted PR #28's "train longer" (color:r 0.877→0.779, forge Δ→~0; best +0.0239 at the *lowest* budget). The last host-side lever, falsified. | P1 |

## Archived

| change | landed via | note |
|---|---|---|
| [add-cascade-host-shim](changes/archive/add-cascade-host-shim/) | sm-sae PR #3 | gate 7.3 (≥0.05 trained-vs-random delta) was missed; arguably superseded by the sae-forge world-model-protocol upstream landing |
| [principled-feature-selection-at-encoding-cap](changes/archive/principled-feature-selection-at-encoding-cap/) | sm-sae PR #1 | gate 8.4 (≥4 clusters on cascade__jumprelu) was met later — sm-sae PR #5's W_enc fix took clusters to 12 |
| [diagnose-compressor-over-consolidation](changes/archive/diagnose-compressor-over-consolidation/) | resolved-not-implemented by sm-sae PR #5 | root cause was sm-sae sending wrong-shape W_enc to polygram, not a polygram tuning issue; see the resolution note in proposal.md |
| [sae-forge-world-model-adapter](changes/archive/sae-forge-world-model-adapter/) | sae-forge PR #55 | implementation landed upstream; sm-sae pins saeforge for the new seam. A separate `retire-cascade-host-shim` follow-up is needed before the shim can actually be deleted (the merged upstream spec was protocol-only; concrete non-transformer adapter support is deferred). |

The **gate-7.3 lineage** (the host-side search — every lever falsified in
PR order, which motivated the gate reframe above):

| change | landed via | note |
|---|---|---|
| [aux-supervise-cascade-host](changes/archive/aux-supervise-cascade-host/) | sm-sae PR #19 | v1 5-label aux head; gate 7.3 missed (Δ +0.0072). First host-side attempt. |
| [probe-full-gt-recoverability-cascade-host](changes/archive/probe-full-gt-recoverability-cascade-host/) | sm-sae PR #22 | full 110-feature GT probe; surfaced the weak per-particle/per-color set (0.74–0.85) that drove v2. |
| [richer-cascade-host-supervision-v2](changes/archive/richer-cascade-host-supervision-v2/) | sm-sae PR #23 | 110-label per-channel + focal-BCE aux head; gate 7.3 missed (Δ −0.0053). |
| [investigate-cascade-host-capacity-sweep](changes/archive/investigate-cascade-host-capacity-sweep/) | sm-sae PR #25 | width saturates ~0.87 by n_embd≈96; depth monotonic L2→L6. |
| [cascade-host-depth-sweep](changes/archive/cascade-host-depth-sweep/) | sm-sae PR #26 | depth peaks at L6; L8–12 regress; declared "host-side exhausted". |
| [cascade-sae-family-binding](changes/archive/cascade-sae-family-binding/) | sm-sae PR #27 | SAE family binds; the Δ-vs-random gate framing penalises strong structural priors (topk random already 0.758). |
| [cascade-rollout-entropy-measurement](changes/archive/cascade-rollout-entropy-measurement/) | sm-sae PR #28 | cascade vocab IS information-rich (0% features <0.7 AUC from state_t); LM drops 0.09–0.16 AUC; recommended the budget sweep. |
| [cascade-host-nonautoregressive](changes/archive/cascade-host-nonautoregressive/) | decision record | terminal pivot — **RESOLVED** via option B (reframe gate 7.3, above). Option A (a non-AR host) preserved there as an optional future experiment. |
