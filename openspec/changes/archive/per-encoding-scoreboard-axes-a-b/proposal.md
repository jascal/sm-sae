# per-encoding-scoreboard-axes-a-b

## Why

The scoreboard's Axis A (SAE reconstruction quality) and Axis B
(ground-truth feature alignment) currently report **pre-polygram**
numbers — they measure the SAE alone, before any compression has
happened. Axis C (polygram cancellation) is per-encoding, but A and B
are not.

This is a real gap on the benchmark's claim of being a comprehensive
compression-extraction scoreboard. A reader looking at the report can
see:

- *"this Rung3 config passes 4/4 cancellations"* (Axis C)
- *"the SAE this came from explained 96% of variance"* (Axis A)
- *"the SAE recovered 37% of GT features at AUC ≥ 0.95"* (Axis B)

…but they cannot see *"after compressing with Rung3, the kept feature
set explains 81% of variance and recovers 22% of GT features"*. That
is the actually-actionable cost-of-compression number, and it's
encoding-dependent because different encodings keep different feature
subsets.

User raised this directly in chat: *"Why are axes A and B not broken
out by polygram encoding?"*. The honest answer is "pre-polygram A and B
are encoding-independent; post-polygram A and B should be per-encoding
and we don't currently measure them". This change closes that gap.

## What Changes

- **`forge_pipeline.py` gains stage 6.5**: after Compressor produces
  the compressed safetensors, re-score the kept-feature subset on the
  feed.
  - **Post-A (reconstruction)**: build a reduced decoder
    `W_dec_kept = W_dec[:, kept_ids]`; encode the feed via the SAE,
    zero out the non-kept feature columns of the latent code, decode
    with the kept-only decoder. Variance explained:
    `1 - var(X - X_hat_kept) / var(X)`.
  - **Post-B (GT alignment)**: AUC of each kept feature's encoded
    activations against each GT label; report `coverage_0.95`,
    `coverage_0.90`, `mean_best_auc` matching the existing Axis-B
    summary shape.
- **`forge_results.json` gains a `post_compress_score` block** mirroring
  the existing `baseline_score` block, populated from stage 6.5.
- **Scoreboard "Forge pipeline runs" table** gains four new columns:
  `post-A`, `post-A delta`, `post-B cov≥0.95`, `post-B delta`. Delta
  columns are colour-coded (green ↑, red ↓) so the cost-of-compression
  is visible at a glance.
- **A new sweep entry point** `scripts/forge_pipeline_matrix.py` (or a
  `--matrix` flag on the existing script) that loops over
  (feed, variant, encoding) combinations and writes one
  `forge_results.json` per cell. Lets the scoreboard fill out the full
  matrix without per-cell manual invocation.

## Capabilities

### New Capabilities

- `post-compression-scoring`: re-evaluation of the SAE's kept-feature
  subset against the feed (Axis A) and the GT labels (Axis B) after
  polygram has decided which features to drop. Distinct from
  `baseline-score` (pre-polygram) and from `forge-score`
  (post-projection-into-feature-basis via sae-forge).

### Modified Capabilities

- `forge-pipeline`: gains a new stage 6.5 between Compressor and forge,
  emitting `post_compress_score` in the result payload.
- `scoreboard-forge-pipeline-runs`: gains four new columns; the prose
  is updated to explain the per-encoding-A/B framing the user asked
  about.
- `scoreboard-axes`: the "Three axes of measurement" overview table
  gains a note that Axis A and Axis B have both pre- and post-polygram
  variants; the Forge pipeline runs table is where the post values
  live.
