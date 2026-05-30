# design — probe-full-gt-recoverability-cascade-host

## Context

`aux-supervise-cascade-host`'s v1 measurement (this session, 2026-05-20) confirmed two things and falsified one:

**Confirmed:**
1. The pooled aux head works mechanically: aux loss dropped 0.77 → 0.006 in 5 epochs at `n_embd=61`; saved model contains the aux_head weights; trainer/forge_pipeline integration is clean.
2. The 5-label aux vocabulary's 3 measurable labels are already at AUC ≥ 0.97 in the LM-only host's pooled hidden state. The proposal's Risk #1 ("a clever cascade-trained LM might learn to compute them implicitly") was the load-bearing risk.

**Falsified:**
The implicit hypothesis behind `aux-supervise-cascade-host` was: *"the host's LM objective doesn't reward encoding conservation / lineage / existence facts, so SAE alignment misses them."* The probe falsifies this: the host **does** encode them (LM gradients pressure the host to predict next-state tokens, which requires representing exactly these facts).

This change funds the next-most-likely diagnosis: somewhere between the host's residual stream and the SAE's GroundTruthTarget score, the cascade-structure signal is getting dropped. Three candidate layers:

1. **SubspaceProjector** — projects the host's hidden state into the SAE's feature basis. If the projection's scale_boost is mis-calibrated (the `'auto'` mode picks `1.0000` for this configuration), the post-projection signal may bear no relationship to the residual signal.
2. **The SAE itself** — `cascade__jumprelu` is a JumpReLU sparse autoencoder. Its threshold-based activation may discard discriminative information that the projection preserves.
3. **The post-compression polygram dictionary** — the Compressor merged 128 SAE features down to 12 clusters. If those 12 clusters span the wrong subspace, the GroundTruthTarget can't recover GT features it should be able to.

The probe doesn't *fix* any of these — it identifies which one to file the next change against.

## Goals / Non-Goals

**Goals:**
- Measure GT-feature recoverability from the LM-only host's pooled residual stream (full 110-feature GT vocabulary, not just the 5 aux labels).
- Measure GT-feature recoverability from the **post-projection** residual stream (after `SubspaceProjector` has encoded into the feature basis).
- Surface the bucket A / B / C interpretation as a single one-line summary.
- File the matching follow-up openspec change with the per-feature AUC table.

**Non-Goals:**
- Retrain anything. The probe consumes existing artefacts.
- Modify `SubspaceProjector` or the SAE pipeline.
- Expand the GT vocabulary. (That's bucket C's follow-up.)
- A version of the probe that compares against `cascade__l1` or `cascade__topk` SAEs — `cascade__jumprelu` is the gate-7.3 cell.

## Decisions

### Decision 1 — Probe at two layer depths, not just one

The single-layer probe (residual only) tells us *whether* the signal is in the host. The two-layer probe (residual + post-projection) tells us *which layer drops it*. Cost is roughly 2× a single-layer probe (~10 minutes total on Intel macOS); the diagnostic value is binary — without the second measurement, we'd guess at the bottleneck.

### Decision 2 — Use the existing `aux-supervise-cascade-host`'s tooling

`scripts/probe_host_aux_recoverability.py` already implements the per-label LogisticRegression probe; the new script extends it to a wider label vocab and adds the post-projection measurement. No new dependencies, no new infrastructure.

### Decision 3 — Bucket A/B/C thresholds are heuristic, not load-bearing

The 0.9 / 0.7 / 80% / 50% thresholds in the bucket definitions are starting points. The per-feature AUC table is the load-bearing artefact; the bucket label is just a routing hint. If the data lands ambiguously, file all three follow-ups and let evidence steer the priority.

### Decision 4 — Don't change `aux-supervise-cascade-host` v1

PR #19 ships sound machinery. The pooled aux head, the trainer integration, the scoreboard rendering, the existing probe script — all useful for future v2 label vocabularies. We don't want to revert any of that.

What we *do* want to capture: in the v1 archive, a note that the 5-label vocabulary is empirically inadequate (3 labels at ceiling, 2 degenerate) and the diagnosis pivoted downstream. The archive entry should link to this change's probe results.

### Decision 5 — Per-particle labels (bucket C's follow-up) are bigger than the v1 vocabulary

If the probe shows bucket C, the v2 aux vocabulary should target **per-particle** features (e.g. `particle:t_b`, `flavor:u`, `color:g`, `is_antiparticle`), not aggregates (`state_has_top`). That's ~110 labels (full GT vocab). The aux head's output dim goes from 5 to ~110. The dual-head + focal-loss recipe from econ-sae Phase 6.2 becomes mechanically necessary at that scale (per-channel head for high-cardinality labels; focal loss for the long tail of rare particles). That work would be substantial; this change does not commit to it, only files the proposal under bucket C.

## Risks / Trade-offs

- **Post-projection probe needs a working `SubspaceProjector` calibration.** The probe loads the cascade host and runs it through `from_host` to apply the projection. If the projection fails (rare, but `scale_boost='auto'` calibration has edge cases), the probe falls back to residual-only and notes the failure in its output.

- **The 110-feature GT vocab includes some structurally-trivial labels.** `is_charged`, `is_colored`, `is_fermion`, etc. are derivable from a handful of others. Probe AUCs for these will be near 1.0 even with no useful host representation. Mitigation: the bucket interpretation gates on a *fraction* of features (≥ 80% or ≥ 50%), not on individual feature AUCs. Trivial labels saturate at 1.0 in both columns and don't move the bucket.

- **n=5000 trajectory budget yields ~10k samples.** Sufficient for LogisticRegression on 61-dim or smaller features. If a future probe needs more (e.g. higher-dimensional projected features), bump to n=20000 (still < 5 min).

- **The interpretation buckets are heuristic.** The probe's value is the per-feature AUC table; the bucket label routes to the right follow-up but doesn't replace human triage of the numbers.

## Migration

None. The probe is additive; existing scripts continue to work unchanged.
