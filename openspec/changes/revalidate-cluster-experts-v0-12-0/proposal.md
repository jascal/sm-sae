# revalidate-cluster-experts-v0-12-0

## Why

sm-sae PR #16 (`f6f0c4e`) ran the first GT-labeled answer-key validation of
polygram's `cluster_experts` expert-routing — does decoder-cosine clustering
recover the known SM feature structure? — on polygram **0.10.0**. It passed
(see Result below). But two gaps make that finding fragile:

1. **Version drift.** The repo now pins polygram **v0.12.0** (PR #21), two minor
   versions past the 0.10.0 the validation ran on (v0.11 added expert-routing
   diagnostics; `cluster_experts` could have changed). The validation was never
   re-confirmed on the pinned version, and the local venv still had 0.10.0.
2. **No durable record.** The validation result lives only in the PR #16 commit
   message and a gitignored `runs/cluster_experts/<cell>/results.json`. There is
   no committed summary artifact and no openspec record — so the benchmark's
   answer-key result for cluster_experts is not inspectable from the repo.

This change re-runs the validation on the pinned v0.12.0 and captures it
durably.

## What Changes

### Re-validation on polygram v0.12.0 (done; no API fix needed)

Installed v0.12.0 and re-ran `scripts/cluster_experts_demo.py` on the two cells
PR #16 validated. `cluster_experts`'s signature and the `ExpertDictionary`
(`.experts` / `.n_experts` / `.n_features`) interface are unchanged, so the demo
runs end-to-end with no code change. The validation **reproduces** (see Result).

### Durable capture

- Add a consolidated, compact `runs/cluster_experts/summary.json` (per-cell
  headline + an overall pass verdict + the polygram version), written by the
  demo, and **force-add** it (matching the capacity/depth/budget-sweep
  convention — `runs/` is gitignored, only the compact summary is tracked).
- Update the demo's module docstring (currently says "polygram 0.10.0") to
  record the v0.12.0 re-validation.
- Note the result in `README.md` (Phase C / relationship-to-polygram) and keep
  this openspec change as the durable record.

### One v0.12.0 observation (not a fix)

At `coherence_threshold=0.5` on cascade__jumprelu, v0.12.0 emits a
`UserWarning: Degenerate cosine partition` (all features go singleton; max
off-diagonal cosine 0.44, recommended fallback 0.352). Informative, not a
failure — the meaningful clustering is at threshold 0.3. Recorded, not silenced.

## Result (live, polygram v0.12.0, 2026-05-30)

| cell | thr | experts | multi-member | multi-meta ≥0.80 | ≥0.70 | GT cov ≥.95 | µAUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| cascade__jumprelu (rung5) | 0.3 | 65 | 25 | 2 | 10 | 20 | 0.900 |
| embedded__topk (rung5) | 0.5 | 37 | 12 | **12** | 12 | 24 | 0.970 |

Matches PR #16 (0.10.0): cascade strongest cluster `kind:antilepton` @ 0.884;
embedded **12/12** multi-member clusters recover a META label at ≥0.80
(`is_scalar`/`flavor:H` — Higgs substructure from decoder cosine alone).
**Verdict: cluster_experts' answer-key validation holds on the pinned v0.12.0.**

## Capabilities

### Modified Capabilities

- `cluster-experts-gt-validation`: re-confirmed on polygram v0.12.0; gains a
  committed `runs/cluster_experts/summary.json` + openspec record. No change to
  the scoring logic.

## Impact

- `scripts/cluster_experts_demo.py` — write a consolidated `summary.json`
  (per-cell headline + verdict + polygram version); update the docstring.
- `runs/cluster_experts/summary.json` — force-added (compact).
- `README.md` — record the v0.12.0 validation result.
- `tests/` — a unit test for the summary builder (pure dict→dict; no training).

## Out of scope

- **Re-running the full forge/scoreboard** on v0.12.0 — separate concern.
- **The degenerate-partition fallback** — upstream polygram tuning, not sm-sae.
- **New SAE cells** beyond the two PR #16 validated — the two are the substrate.

## Acceptance

Ships when the demo runs clean on v0.12.0 for both cells, the compact
`summary.json` is committed, the docstring + README record the v0.12.0 result,
and the suite is green.
