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
| [add-cascade-host-shim](changes/add-cascade-host-shim/) | proposed | P0 |
| [sae-forge-world-model-adapter](changes/sae-forge-world-model-adapter/) | proposed (upstream ask of sae-forge; retires the shim) | P1 |
| [principled-feature-selection-at-encoding-cap](changes/principled-feature-selection-at-encoding-cap/) | in progress (gate 8.4 missed; see follow-up) | P1 |
| [diagnose-compressor-over-consolidation](changes/diagnose-compressor-over-consolidation/) | proposed (follow-up to selection) | P1 |
| [per-encoding-scoreboard-axes-a-b](changes/per-encoding-scoreboard-axes-a-b/) | proposed | P1 |
| [auto-tune-subspace-projector-scale-boost](changes/auto-tune-subspace-projector-scale-boost/) | proposed | P2 |
