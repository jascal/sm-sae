# design — per-encoding-scoreboard-axes-a-b

## Context

The benchmark is structured around three axes:

- **A** — reconstruction quality (does the SAE reproduce its inputs?)
- **B** — ground-truth feature alignment (do the SAE's features map onto
  known physical labels?)
- **C** — compression-extraction quality (does polygram / sae-forge
  preserve the structure once compressed?)

Axes A and B are properties of the **SAE alone**. They don't depend on
polygram's encoding choice; you can compute them from a raw
`runs/<feed>__<variant>.pt` checkpoint with no knowledge of polygram.
That's why they live in the top-level "Axis A" and "Axis B" tables on
the scoreboard, not the "Forge pipeline runs" table.

Axis C is per-encoding — different polygram encodings produce different
cancellation results. We surface this in the polygram encoding sweep
table.

But once polygram compresses the SAE dictionary (keeps N of M features
based on the Compressor's merge strategy), the **kept feature subset**
has its own A and B. These are encoding-dependent because different
encodings cap the feature count differently (MPSRung1 → 8, Rung3 → 16,
etc.) and the Compressor's merge decisions depend on the encoding's
gram matrix.

User raised the gap in chat:

> "Why are axes A and B not broken out by polygram encoding?"

This change adds the post-compression A and B as new columns on the
"Forge pipeline runs" table — distinct from the pre-polygram A and B
that already exist.

## Goals

- **Make the cost of compression visible per encoding.** A reader
  should be able to look at the scoreboard and answer: *"on
  `cascade__jumprelu`, did Rung3 cost us more variance-explained than
  Rung5 did?"*
- **Reuse existing scoring infrastructure** (`auc_matrix` for B,
  trivial decoder reapplication for A). No new dependencies.
- **Make the matrix easy to fill out.** A sweep entry point that
  produces one cell per (run_id, encoding) combination is the natural
  way to populate the scoreboard.

## Non-Goals

- **Re-training the SAE under different polygram targets.** That's
  what sae-forge's basis-loop (compress ↔ regrow) is designed to do.
  We're not duplicating that here; we're just measuring the
  single-shot cost of compression at fixed-SAE.
- **A new "Axis A' / Axis B'" promotion.** Post-compression A and B
  stay in the Forge pipeline runs table. Promoting them to top-level
  axes would clutter the overview; cells in a sub-table is the right
  level.
- **Comparing to other compression tools** (e.g. PCA, k-means on the
  decoder). Same fixture, same scoring, would make the comparison
  trivial — but the scoping pressure here is sm-sae as a polygram
  benchmark, not a generic compression benchmark.

## Decisions

### 1. Where post-A and post-B live

Inside `forge_results.json` next to `baseline_score` and
`forge_score`. The JSON shape becomes:

```json
{
  "baseline_score":      {var_explained, coverage_0.95, mean_best_auc, ...},
  "post_compress_score": {var_explained, coverage_0.95, mean_best_auc, n_kept, ...},
  "forge_score":         float | null,
  ...
}
```

This keeps the three measurement stages clearly distinguished:
**baseline** = SAE alone, **post_compress** = SAE with polygram's kept
features, **forge** = the sae-forge native model.

### 2. How post-A is computed

The SAE has an encoder `W_enc` and decoder `W_dec`. Polygram
compression drops features by zeroing rows of `W_dec` and removing the
corresponding rows of `W_enc`. To re-score reconstruction with the
kept subset:

1. Encode the feed normally: `z = sae.encode(x)`.
2. Mask out non-kept columns of `z`: `z_masked[:, ~kept_mask] = 0`.
3. Decode: `x_hat = sae.decode(z_masked)`.
4. Compute `var_explained = 1 - var(x - x_hat) / var(x)`.

Step 2 reflects what polygram's Compressor actually does (it zeros
non-representative features); the post-A score is what reconstruction
quality drops to under that intervention.

### 3. How post-B is computed

Same as Axis B today, restricted to the kept feature columns of `z`:

1. `z = sae.encode(x)`, take only `z[:, kept_ids]`.
2. Compute `auc_matrix(z[:, kept_ids], Y)`.
3. Per-GT-feature take `max` across kept SAE features.
4. Report coverage at thresholds + mean best AUC.

Equivalent to running the existing Axis-B scorer with a feature-subset
filter applied. No new scoring math.

### 4. Delta colour-coding

The scoreboard table shows both post-A (absolute) and `Δ A` (delta vs
baseline). Delta cells colour:

- `pass` (green): `Δ ≥ -0.02` (essentially no degradation; might be
  marginally better due to noise filtering).
- `partial` (amber): `-0.10 ≤ Δ < -0.02`.
- `fail` (red): `Δ < -0.10`.

These thresholds are starting points; reasonable to tune once we see
distributions of real deltas across runs.

### 5. Sweep matrix scope

Initial sweep: `{embedded__topk, cascade__jumprelu} × {mps_rung1,
rung3, rung4_amp_budget, rung5_amp_budget}` = 8 cells. Cheap to run
(each cell takes < 1 minute given the existing pipeline). Skips cells
where the SAE checkpoint doesn't exist.

The full per-axis matrix (3 feeds × 3 SAE variants × 4 encodings = 36
cells) is technically possible but produces a wide table that's hard
to read. Defer the full matrix until the 8-cell preview surfaces
something interesting.

## Risks

- **Post-A might be uninteresting if Compressor produces 0 clusters**.
  Currently `embedded__topk` / Rung3 → 3 clusters / 3 kept / 3 zeroed.
  That's a tiny intervention; post-A will be nearly equal to baseline.
  Mitigation: the
  `principled-feature-selection-at-encoding-cap` change should land
  first and produce richer Compressor outputs, after which post-A
  becomes informative.
- **Wall-clock for the matrix mode** scales with the number of cells.
  8 cells at < 1 min each is fine. If the matrix grows, parallelize
  cells via `multiprocessing` or just accept the time cost.
- **Delta thresholds are heuristic**: ±0.02 / ±0.10 are guesses.
  Acceptable for v1; revisit once we have distributions.
