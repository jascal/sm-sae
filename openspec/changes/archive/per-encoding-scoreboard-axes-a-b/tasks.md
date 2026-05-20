# tasks — per-encoding-scoreboard-axes-a-b

## 1. Post-A scoring (reconstruction)

- [x] 1.1 Add `score_post_compression_reconstruction(sae, feed,
      kept_ids: list[int]) -> dict`. Returns
      `{"var_explained": float, "recon_loss_mse": float,
        "n_kept": int, "n_total": int}`.
- [x] 1.2 Implementation:
      (a) compute `z = sae(feed.X).z`;
      (b) zero out columns of `z` not in `kept_ids`;
      (c) decode via `x_hat = sae.decode(z_masked)`;
      (d) return variance-explained and MSE.
- [x] 1.3 Unit test on a fixture SAE: with `kept_ids = all feature
      indices`, post-A should equal the SAE's baseline reconstruction
      quality.
- [x] 1.4 Unit test: with `kept_ids = []`, post-A `var_explained`
      should be ≤ 0 (decoder reduces to bias term).

## 2. Post-B scoring (GT alignment)

- [x] 2.1 Add `score_post_compression_gt(sae, feed, kept_ids:
      list[int]) -> dict`. Returns
      `{"coverage_0.95": float, "coverage_0.90": float,
        "mean_best_auc": float, "n_kept": int, "n_gt_features": int}`.
- [x] 2.2 Implementation: reuse `auc_matrix(z[:, kept_ids], Y)` from
      `visualize.py`; per-GT-feature take `max` over kept SAE features;
      report coverage thresholds and mean-best matching the existing
      Axis-B summary format.
- [x] 2.3 Unit test: post-B with all features should equal the SAE's
      baseline Axis-B numbers.

## 3. Wiring stage 6.5 into forge_pipeline

- [x] 3.1 After Compressor runs in stage 6, extract the kept feature
      IDs from the compressed safetensors (use
      `FeatureBasis.from_polygram_checkpoint`'s `kept_ids` attribute).
- [x] 3.2 Call both post-A and post-B scorers; combine into a single
      `post_compress_score` dict.
- [x] 3.3 Log a one-line summary in the console output:
      `[6.5] post-compression: var_explained=0.81 (Δ -0.15)
            cov≥0.95=22.3% (Δ -14.9%)  mean_best_auc=0.72 (Δ -0.05)`.
- [x] 3.4 Add `post_compress_score` to the `forge_results.json` payload
      next to `baseline_score` and `forge_score`.

## 4. Scoreboard rendering

- [x] 4.1 Update `_format_forge_pipeline_results` to add columns:
      `post-A`, `Δ A`, `post-B cov`, `Δ B`. Render deltas with a
      colour class (`pass` = improvement, `fail` = regression,
      `partial` = within ±0.02).
- [x] 4.2 If `post_compress_score` is missing (older runs), show "—"
      in the new columns rather than failing.
- [x] 4.3 Update the surrounding prose: explicitly answer the user's
      question — *"why are A and B not per-encoding?"* — with a
      paragraph distinguishing pre- and post-polygram A/B and pointing
      the reader at the new columns.

## 5. Sweep entry point

- [x] 5.1 Add `scripts/forge_pipeline.py --matrix` mode (or
      `scripts/forge_pipeline_matrix.py`, whichever keeps the CLI
      cleaner). Loops over the cross-product
      `{embedded__topk, cascade__jumprelu} × {mps_rung1, rung3, rung4,
      rung5}` (8 cells) and writes one result per cell under
      `runs/sae_forge/<run_id>__<encoding>/forge_results.json`.
- [x] 5.2 Skip cells where the corresponding SAE checkpoint doesn't
      exist (with a clear log line).
- [x] 5.3 Resumable: skip cells whose `forge_results.json` already
      exists, unless `--force` is passed.
- [x] 5.4 Print a final summary table showing
      `(run_id × encoding → post-A, post-B, forge_score)`.

## 6. Update scoreboard overview

- [x] 6.1 The "Three axes of measurement" intro table near the top of
      the benchmark section gains a footnote: *"A and B have both
      pre-polygram (encoding-independent) and post-polygram
      (per-encoding) variants. Pre-polygram values are reported in the
      tables under Axes A and B; post-polygram values are in the Forge
      pipeline runs table below."*
- [x] 6.2 Verify the recommended-defaults table still cites the right
      evidence after the new columns land.

## 7. Verify

- [x] 7.1 Run the matrix sweep; confirm 8 cells populated.
- [x] 7.2 Visually inspect: do any encoding choices show wildly
      different post-A or post-B? (Expected: yes — Rung3 keeps 16
      features, Rung5(n_amp_qubits=2) keeps 32, so the columns should
      differ.)
- [x] 7.3 If a clear winner-per-axis emerges, update the
      recommended-defaults table to cite the cross-encoding post-A and
      post-B evidence.

## 8. Documentation

- [x] 8.1 Update the scoreboard section's aside to flag this is the
      first time post-polygram A and B are surfaced; cite the
      provenance.
- [x] 8.2 Archive this change directory once landed.

## 9. Acceptance gate

- [x] 9.1 `forge_results.json` includes `post_compress_score` with
      `var_explained`, `coverage_0.95`, `coverage_0.90`,
      `mean_best_auc`, `n_kept`.
- [x] 9.2 Scoreboard "Forge pipeline runs" table has the four new
      columns and a per-cell delta colour.
- [x] 9.3 At least 4 (run_id × encoding) cells measured and rendered.
- [x] 9.4 The post-A / post-B columns produce non-uniform values
      across encodings (i.e. the new axis actually carries
      information).
