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

The cascade-host / gate-7.3 lineage is the active research arc. Each
change below landed via a PR; all are stable and **pending archival**
(move to `changes/archive/` together, once the maintainer picks the
gate-7.3 close-out path — see `cascade-host-nonautoregressive`). The
budget sweep exhausted the last host-side lever, so the host-side search
is closed (negatively).

| change | status | priority |
|---|---|---|
| [cascade-host-nonautoregressive](changes/cascade-host-nonautoregressive/) | proposed — terminal pivot + maintainer decision point (build a non-AR host **or** re-frame gate 7.3 to absolute forge_score per PR #27). Not implemented. | P1 |
| [cascade-host-training-budget-sweep](changes/cascade-host-training-budget-sweep/) | **impl landed + run** — refuted PR #28's "train longer": more gradient steps drove color:r 0.877→0.779 and forge Δ to ~0 (best +0.0239 at the *lowest* budget). Closed the host-side search. | P1 |
| [cascade-rollout-entropy-measurement](changes/cascade-rollout-entropy-measurement/) | landed (PR #28) — archive pending | P1 |
| [cascade-sae-family-binding](changes/cascade-sae-family-binding/) | landed (PR #27) — archive pending | P2 |
| [cascade-host-depth-sweep](changes/cascade-host-depth-sweep/) | landed (PR #26) — archive pending | P2 |
| [investigate-cascade-host-capacity-sweep](changes/investigate-cascade-host-capacity-sweep/) | landed (PR #25) — archive pending | P2 |
| [richer-cascade-host-supervision-v2](changes/richer-cascade-host-supervision-v2/) | landed (PR #23) — archive pending | P2 |
| [probe-full-gt-recoverability-cascade-host](changes/probe-full-gt-recoverability-cascade-host/) | landed (PR #22) — archive pending | P2 |
| [aux-supervise-cascade-host](changes/aux-supervise-cascade-host/) | landed (PR #19); gate 7.3 missed at Δ +0.0072 — archive pending | P1 |

## Archived

| change | landed via | note |
|---|---|---|
| [add-cascade-host-shim](changes/archive/add-cascade-host-shim/) | sm-sae PR #3 | gate 7.3 (≥0.05 trained-vs-random delta) was missed; arguably superseded by the sae-forge world-model-protocol upstream landing |
| [principled-feature-selection-at-encoding-cap](changes/archive/principled-feature-selection-at-encoding-cap/) | sm-sae PR #1 | gate 8.4 (≥4 clusters on cascade__jumprelu) was met later — sm-sae PR #5's W_enc fix took clusters to 12 |
| [diagnose-compressor-over-consolidation](changes/archive/diagnose-compressor-over-consolidation/) | resolved-not-implemented by sm-sae PR #5 | root cause was sm-sae sending wrong-shape W_enc to polygram, not a polygram tuning issue; see the resolution note in proposal.md |
| [sae-forge-world-model-adapter](changes/archive/sae-forge-world-model-adapter/) | sae-forge PR #55 | implementation landed upstream; sm-sae pins saeforge for the new seam. A separate `retire-cascade-host-shim` follow-up is needed before the shim can actually be deleted (the merged upstream spec was protocol-only; concrete non-transformer adapter support is deferred). |
