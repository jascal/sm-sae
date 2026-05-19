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

| change | status | priority |
|---|---|---|
| [aux-supervise-cascade-host](changes/aux-supervise-cascade-host/) | proposed (closes [archived add-cascade-host-shim](changes/archive/add-cascade-host-shim/) gate 7.3 if its acceptance gate is met) | P1 |
| [per-encoding-scoreboard-axes-a-b](changes/per-encoding-scoreboard-axes-a-b/) | proposed | P1 |
| [auto-tune-subspace-projector-scale-boost](changes/auto-tune-subspace-projector-scale-boost/) | proposed | P2 |

## Archived

| change | landed via | note |
|---|---|---|
| [add-cascade-host-shim](changes/archive/add-cascade-host-shim/) | sm-sae PR #3 | gate 7.3 (≥0.05 trained-vs-random delta) was missed; arguably superseded by the sae-forge world-model-protocol upstream landing |
| [principled-feature-selection-at-encoding-cap](changes/archive/principled-feature-selection-at-encoding-cap/) | sm-sae PR #1 | gate 8.4 (≥4 clusters on cascade__jumprelu) was met later — sm-sae PR #5's W_enc fix took clusters to 12 |
| [diagnose-compressor-over-consolidation](changes/archive/diagnose-compressor-over-consolidation/) | resolved-not-implemented by sm-sae PR #5 | root cause was sm-sae sending wrong-shape W_enc to polygram, not a polygram tuning issue; see the resolution note in proposal.md |
| [sae-forge-world-model-adapter](changes/archive/sae-forge-world-model-adapter/) | sae-forge PR #55 | implementation landed upstream; sm-sae pins saeforge for the new seam. A separate `retire-cascade-host-shim` follow-up is needed before the shim can actually be deleted (the merged upstream spec was protocol-only; concrete non-transformer adapter support is deferred). |
