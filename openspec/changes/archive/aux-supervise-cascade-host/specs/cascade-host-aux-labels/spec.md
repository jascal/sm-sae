# cascade-host-aux-labels Specification

## Purpose

Provides per-state binary labels for the SM cascade that the
cascade-host trainer can use as supervised targets in an auxiliary
loss. The labels are training-side artefacts only — they are not
part of the sm-sae benchmark's ground-truth grading vocabulary and
are independent of what the SAE is scored against.

The capability exists because `add-cascade-host-shim`'s pure
next-state token objective produced trained-vs-random faithfulness
deltas well below the +0.05 gate-7.3 threshold on every cascade cell
(see this change's `proposal.md` for the measured table). Auxiliary
supervision on physically-meaningful per-state facts is the proven
fix from econ-sae's Phase 5.1, transposed here.

## ADDED Requirements

### Requirement: `aux_label_names()` returns the v1 label vocabulary

`smsae.host.aux_labels.aux_label_names() -> list[str]` SHALL return
the v1 label names in a stable, fixed order. The v1 vocabulary is
exactly the following 5 labels, in this order:

1. `total_charge_neutral`
2. `total_baryon_neutral`
3. `originated_from_top`
4. `state_has_higgs`
5. `state_has_top`

The order is load-bearing: it is the column order of the
`(5,)`-shaped vector returned by `compute_aux_labels` and the column
order the trainer feeds to the aux head.

#### Scenario: label order is stable across calls

- **WHEN** `aux_label_names()` is called twice within a single
  process
- **THEN** both calls return the identical list of 5 strings in the
  same order

#### Scenario: label vocabulary is exactly v1

- **WHEN** `aux_label_names()` is called
- **THEN** the result equals
  `["total_charge_neutral", "total_baryon_neutral",
    "originated_from_top", "state_has_higgs", "state_has_top"]`

### Requirement: `compute_aux_labels` returns per-state binary labels

`smsae.host.aux_labels.compute_aux_labels(state: dict[str, int],
initial_parent: str | None) -> np.ndarray` SHALL return a
`(5,)`-shaped `float32` array of 0/1 values matching the v1
vocabulary's column order.

Label semantics:

- `total_charge_neutral`: `1.0` when `sum(charge[p] * state[p] for p
  in state)` equals `0` within numerical tolerance; `0.0` otherwise.
  Charge values are read from `smsae.sm.embeddings.build_sm()`.
- `total_baryon_neutral`: identical contract with baryon number in
  place of charge.
- `originated_from_top`: `1.0` when `initial_parent` is one of
  `{"t_r", "t_g", "t_b"}`; `0.0` otherwise (including when
  `initial_parent is None`).
- `state_has_higgs`: `1.0` when `"H"` appears as a key with
  positive count in `state`; `0.0` otherwise.
- `state_has_top`: `1.0` when any of `t_r`, `t_g`, `t_b` appears as
  a key with positive count in `state`; `0.0` otherwise.

The implementation SHALL cache the `build_sm()` result at module
scope to avoid per-call rebuild cost. It SHALL NOT import torch.

#### Scenario: returned shape and dtype

- **WHEN** `compute_aux_labels(state, initial_parent)` is called for
  any valid state
- **THEN** the result is a `numpy.ndarray` with `shape == (5,)` and
  `dtype == numpy.float32`

#### Scenario: every label is 0.0 or 1.0

- **WHEN** `compute_aux_labels(state, initial_parent)` is called for
  any valid state
- **THEN** every element of the returned array is exactly `0.0` or
  `1.0` (no intermediate values)

#### Scenario: `originated_from_top` requires `initial_parent`

- **GIVEN** a state where the rollout originated from a top quark
  (any of `t_r`, `t_g`, `t_b`) but `initial_parent is None`
- **WHEN** `compute_aux_labels(state, None)` is called
- **THEN** the `originated_from_top` column is `0.0` (the label
  cannot fire without `initial_parent` evidence; it is a lineage
  fact about the rollout, not the current state)

#### Scenario: conservation labels match hand-computed values

- **GIVEN** three fixture cascade rollouts with known charge and
  baryon-number totals
- **WHEN** `compute_aux_labels` is called on each terminal state
- **THEN** `total_charge_neutral` and `total_baryon_neutral` match
  the hand-computed expected values for each fixture

#### Scenario: module does not import torch

- **WHEN** `smsae.host.aux_labels` is imported in a torch-free
  environment
- **THEN** the import succeeds; no `ImportError` for torch is raised

### Requirement: aux labels are training-side only

The aux-labels capability SHALL NOT alter any SAE training feed, any
benchmark grading vocabulary, or any artefact consumed by the
sm-sae scoreboard's ground-truth alignment grid. It is consumed
solely by the cascade-host trainer (`cascade-host-shim` capability).

#### Scenario: no GT vocab change

- **WHEN** the sm-sae benchmark's ground-truth grading vocabulary
  is enumerated
- **THEN** the 5 aux labels are NOT present (they remain training-
  side artefacts; sm-sae's GT vocabulary is unchanged from prior to
  this change)
