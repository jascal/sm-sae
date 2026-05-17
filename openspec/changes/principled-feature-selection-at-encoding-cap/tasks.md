# tasks — principled-feature-selection-at-encoding-cap

## 1. Selector abstraction

- [ ] 1.1 In `scripts/forge_pipeline.py`, add a `SELECTORS` registry:
      `dict[str, Callable[[sae, feed, records], list[int]]]`.
- [ ] 1.2 Each selector returns the full ordered ID list (best first);
      the caller applies `[:cap]` after.
- [ ] 1.3 Add a tiny `_resolve_selector(arg) -> Callable` helper that
      accepts either a registry key or a user-supplied callable.

## 2. Built-in selectors

- [ ] 2.1 `select_by_head(sae, feed, records)` — returns
      `sorted(records.keys())`. Equivalent to the current
      `feat_ids[:cap]` behaviour; lets users opt back into it.
- [ ] 2.2 `select_by_firing_rate(sae, feed, records)` — compute
      `z = sae(feed.X).z`; firing-rate per feature is
      `(z > 1e-9).float().mean(dim=0)`; return indices sorted by
      firing rate descending. Tie-break on feature_id ascending for
      determinism.
- [ ] 2.3 `select_by_gt_alignment(sae, feed, records)` — build the GT
      matrix `Y` via `build_gt_matrix(feed)`; run `auc_matrix(z, Y)`;
      per feature take `max` across labels; sort descending. Same
      tie-break.
- [ ] 2.4 Unit tests for each selector against a fixture SAE: assert
      the returned ID list is a permutation of `records.keys()`, that
      the ordering matches the selection criterion, and that
      determinism holds across two runs.

## 3. build_dictionary plumbing

- [ ] 3.1 `build_dictionary(records, encoding_name, selector="firing_rate",
      sae=None, feed=None)` — add new args. `sae` / `feed` required
      when the selector needs them (i.e. anything except `head`).
- [ ] 3.2 Resolve the selector via `_resolve_selector(selector)`; call
      it; apply `[:cap]`; thread the resulting IDs into
      `from_sae_lens(...)`.
- [ ] 3.3 Return the kept-ID list alongside the dictionary and
      selection metadata (`{"method": "firing_rate", "n_candidates":
      128, "n_kept": 16, "kept_ids": [...]}`) so `main()` can record
      it.

## 4. CLI

- [ ] 4.1 Add `--select-by {firing_rate, gt_alignment, head}` to the
      `argparse` in `main()`. Default `firing_rate`.
- [ ] 4.2 Pass the chosen selector name through to `build_dictionary`.
- [ ] 4.3 Print the selection summary in the stage-4 console line:
      `[4] wrap as polygram.Dictionary (encoding=rung3, select-by=firing_rate)`.

## 5. Result payload

- [ ] 5.1 In `forge_results.json` the `dictionary` block becomes:
      ```json
      "dictionary": {
        "name": "...",
        "n_features": 16,
        "encoding_max": 16,
        "selection": {
          "method": "firing_rate",
          "n_candidates": 128,
          "n_kept": 16,
          "kept_ids": [12, 41, 88, ...]
        }
      }
      ```
- [ ] 5.2 Update `_format_forge_pipeline_results` in `visualize.py` to
      show the selector method either as a new "selection" column or as
      a `(firing_rate)` suffix on the encoding column. Pick whichever
      keeps the table narrower.

## 6. Verify on the bad case

- [ ] 6.1 Re-run `scripts/forge_pipeline.py cascade__jumprelu` with
      `--select-by firing_rate` and confirm: (a) Compressor produces
      more than 1 cluster (target ≥ 4 given 21 confirmed pairs in
      stage 5), (b) post-compression forge_score is non-trivially
      different from the `head`-selector run.
- [ ] 6.2 Run again with `--select-by gt_alignment`; record both rows
      on the scoreboard for visual comparison.

## 7. Documentation

- [ ] 7.1 Update `scripts/forge_pipeline.py` module docstring to
      document the selector options.
- [ ] 7.2 Add a one-paragraph aside to the visualization scoreboard
      section explaining why selection matters and how to read the
      "selection" column.
- [ ] 7.3 Move this change to `openspec/changes/archive/` once landed.

## 8. Acceptance gate

- [ ] 8.1 `--select-by firing_rate` runs end-to-end on both
      `embedded__topk` and `cascade__jumprelu`.
- [ ] 8.2 `forge_results.json` includes the new `selection` block.
- [ ] 8.3 The scoreboard shows the selector method per row.
- [ ] 8.4 The `cascade__jumprelu` Compressor produces ≥ 4 clusters with
      `firing_rate` selection (vs the current 1).
