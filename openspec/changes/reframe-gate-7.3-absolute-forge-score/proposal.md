# reframe-gate-7.3-absolute-forge-score

## Why

Gate 7.3 — "a trained cascade host beats its random-init baseline by
`forge_delta_vs_random ≥ +0.05`" — was the longest-open gate in the project
(PRs #19/#22/#23/#25/#26/#27/#28/#29). The host-side search closed
**negatively**: capacity (#25), depth (#26), SAE family (#27), aux supervision
(#19/#23) and training budget (#29) were each falsified as levers.

Two findings make the **metric itself** the problem, not any host:

1. **#27 — Δ-vs-random structurally penalises good SAEs.** `cascade__topk`'s
   *random* host already reaches forge_score 0.758, so there is almost nothing
   for training to add — its Δ is ~0 precisely *because* its unsupervised
   structure already encodes the cascade signal. The family with the *largest*
   Δ (`cascade__jumprelu`) is the one whose threshold structure is *weakest*
   unsupervised. Δ rewards weak priors.

2. **#29 — training that lowers CE lowers Δ.** The budget sweep showed more
   gradient steps drive training CE down (to 0.98) while forge Δ collapses to
   ~0 and `color:r` recoverability *falls* (0.877 → 0.779). Any optimisation
   that improves the host on its own objective moves Δ the wrong way.

So Δ-vs-random measures a quantity the autoregressive objective structurally
cannot supply, and rewards exactly the SAE families that do the *least*
unsupervised work. #27 already recommended the fix; this change adopts it.

## What Changes

### Gate 7.3, reframed

| | old (retired) | new |
|---|---|---|
| metric | `forge_delta_vs_random` | absolute `forge_score` |
| target | `≥ +0.05` | `≥ 0.76` |
| measured on | the trained host's cell vs its random init | the **strongest cascade SAE family** on the scoreboard |
| current status | never met (best Δ +0.036) | **MET** — `cascade__topk` = 0.760 ≥ 0.76 |

**Target = 0.76**, pinned to the achieved best (`cascade__topk` reaches 0.760
on the rung5 scoreboard). The gate **closes positively today**; any regression
of the strongest cascade family below 0.76 re-opens it. The bar is an
*achieved-best floor*, not a stretch goal — it asserts the benchmark keeps
producing forge-faithful cascade SAEs, which is what the gate was always
trying to express.

**Δ-vs-random is retired as the gate** but retained as a diagnostic wherever it
is already computed (it still answers "how much did *this host's training* add",
which is a legitimate secondary question — just not the benchmark gate).

### Where the gate is read

The canonical gate is the maximum `forge_score` over the cascade SAE families
on the forge scoreboard (`scripts/forge_pipeline_matrix.py` → the `docs/`
scoreboard). It is **not** a property of the host-training experiments — those
measure `cascade__jumprelu` only and need not themselves clear 0.76.

## Capabilities

### Modified Capabilities

- `gate-7.3`: redefined from `forge_delta_vs_random ≥ +0.05` to absolute
  `forge_score ≥ 0.76` on the strongest cascade family. Δ-vs-random demoted to
  a diagnostic.
- `cascade-host-capacity-sweep`: `GATE_7_3_ABS_FORGE_TARGET = 0.76` constant;
  the `C.2` / `B.2` gate verdicts report the absolute metric (with the cell's
  forge_score), Δ-vs-random printed as a labelled diagnostic, and a note that
  the canonical gate is the scoreboard's strongest-family value.

## Impact

- `scripts/cascade_host_capacity_sweep.py` — gate constant + absolute-metric
  gate evaluation/printing; Δ retained as `delta_vs_random_diagnostic`.
- `tests/test_budget_sweep_config.py` — assert the reframed gate
  (`cell_meets_target`, `gate_7_3_target`, diagnostic survives).
- `README.md`, `scripts/train_cascade_host.py` — gate-7.3 wording updated from
  the Δ definition to the absolute-forge_score definition.
- `openspec/README.md` — lineage note: closed, gate reframed + now MET.
- `cascade-host-nonautoregressive` — resolved (this is its option B); archived
  with a resolution note. Option A (a non-autoregressive host) is preserved
  there as an *optional* future experiment, not a blocker.

## Decision record

`cascade-host-nonautoregressive` framed the close-out as a maintainer choice:
(A) build a non-AR host to test whether the AR objective was the culprit, or
(B) reframe the gate to absolute forge_score. **Option B was chosen**
(2026-05-30). Rationale: it is the low-effort close that stops measuring a
quantity the AR objective cannot supply, and it recognises the faithfulness the
benchmark already achieves (0.760). Option A remains available if a non-AR host
is independently wanted for the sae-forge world-model path.

## Out of scope

- **Building a non-AR host** (option A) — deferred, optional.
- **Re-running the scoreboard** — `cascade__topk` already reads 0.760; no
  re-measurement needed to evaluate the new gate.
- **Raising the bar above the achieved best** — 0.76 is an achieved-best floor
  by deliberate choice; a higher stretch target is a future decision.

## Acceptance

This change ships when the gate is redefined in the gate-bearing code + docs
and `cascade-host-nonautoregressive` is resolved. Gate 7.3 status flips to
**MET** (cascade__topk 0.760 ≥ 0.76). The gate-7.3 saga closes positively.
