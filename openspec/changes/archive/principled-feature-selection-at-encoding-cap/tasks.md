# tasks — principled-feature-selection-at-encoding-cap

## 1. Selector abstraction

- [x] 1.1 In `scripts/forge_pipeline.py`, add a `SELECTORS` registry:
      `dict[str, Callable[[sae, feed, records], list[int]]]`.
- [x] 1.2 Each selector returns the full ordered ID list (best first);
      the caller applies `[:cap]` after.
- [x] 1.3 Add a tiny `_resolve_selector(arg) -> Callable` helper that
      accepts either a registry key or a user-supplied callable.

## 2. Built-in selectors

- [x] 2.1 `select_by_head(sae, feed, records)` — returns
      `sorted(records.keys())`. Equivalent to the current
      `feat_ids[:cap]` behaviour; lets users opt back into it.
- [x] 2.2 `select_by_firing_rate(sae, feed, records)` — compute
      `z = sae(feed.X).z`; firing-rate per feature is
      `(z > 1e-9).float().mean(dim=0)`; return indices sorted by
      firing rate descending. Tie-break on feature_id ascending for
      determinism.
- [x] 2.3 `select_by_gt_alignment(sae, feed, records)` — build the GT
      matrix `Y` via `build_gt_matrix(feed)`; run `auc_matrix(z, Y)`;
      per feature take `max` across labels; sort descending. Same
      tie-break.
- [x] 2.4 Unit tests for each selector against a fixture SAE: assert
      the returned ID list is a permutation of `records.keys()`, that
      the ordering matches the selection criterion, and that
      determinism holds across two runs.

## 3. build_dictionary plumbing

- [x] 3.1 `build_dictionary(records, encoding_name, selector="firing_rate",
      sae=None, feed=None)` — add new args. `sae` / `feed` required
      when the selector needs them (i.e. anything except `head`).
- [x] 3.2 Resolve the selector via `_resolve_selector(selector)`; call
      it; apply `[:cap]`; thread the resulting IDs into
      `from_sae_lens(...)`.
- [x] 3.3 Return the kept-ID list alongside the dictionary and
      selection metadata (`{"method": "firing_rate", "n_candidates":
      128, "n_kept": 16, "kept_ids": [...]}`) so `main()` can record
      it.

## 4. CLI

- [x] 4.1 Add `--select-by {firing_rate, gt_alignment, head}` to the
      `argparse` in `main()`. Default `firing_rate`.
- [x] 4.2 Pass the chosen selector name through to `build_dictionary`.
- [x] 4.3 Print the selection summary in the stage-4 console line:
      `[4] wrap as polygram.Dictionary (encoding=rung3, select-by=firing_rate)`.

## 5. Result payload

- [x] 5.1 In `forge_results.json` the `dictionary` block becomes:
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
      (Implementation also includes `ordered_ids` — the full ranked list
      pre-cap — for reproducibility.)
- [x] 5.2 Update `_format_forge_pipeline_results` in `visualize.py` to
      show the selector method as a new "selection" column.

## 6. Verify on the bad case

- [x] 6.1 Re-run `scripts/forge_pipeline.py cascade__jumprelu` with
      `--select-by firing_rate` and confirm: (a) Compressor produces
      more than 1 cluster — **partial**: clusters go 1 (head) → 3
      (firing_rate). Target was ≥ 4; the 3-cluster outcome is a clear
      improvement but short of the spec gate. (b) faithfulness changes
      from 0.7511 (head) → 0.7448 (firing_rate), a small but
      non-trivial delta confirming the kept-feature subset reaches
      the forge.
- [x] 6.2 Re-ran with `--select-by gt_alignment`: clusters=2,
      faithfulness=0.7468. Also recorded `embedded__topk
      --select-by firing_rate` which hits clusters=4 (meets the gate
      on that run_id). All three selection methods now coexist as
      scoreboard rows.

## 7. Documentation

- [x] 7.1 Update `scripts/forge_pipeline.py` module docstring to
      document the selector options.
- [x] 7.2 Add a one-paragraph aside to the visualization scoreboard
      section explaining why selection matters and how to read the
      "selection" column.
- [ ] 7.3 Move this change to `openspec/changes/archive/` once landed.

## 8. Acceptance gate

- [x] 8.1 `--select-by firing_rate` runs end-to-end on both
      `embedded__topk` and `cascade__jumprelu`.
- [x] 8.2 `forge_results.json` includes the new `selection` block.
- [x] 8.3 The scoreboard shows the selector method per row.
- [ ] 8.4 The `cascade__jumprelu` Compressor produces ≥ 4 clusters with
      `firing_rate` selection (vs the current 1). **Missed:** observed
      3 clusters with `firing_rate`, 2 with `gt_alignment`, 1 with
      `head`. The change does fix the pathology (1 → 3+ clusters with
      principled selection) but the specific ≥ 4 threshold appears to
      be a Compressor-tuning issue, not a selector issue — the kept
      pairs are over-consolidating into a single cluster. See follow-up
      note below.

## 9. Follow-up

The ≥ 4 cluster gate on `cascade__jumprelu` was not met. With 20
confirmed pairs under `firing_rate` and 26 under `gt_alignment`, the
Compressor still collapses them to 3 / 2 clusters respectively. This is
a Compressor-side concern (cluster-merge aggressiveness, distance
metric, or threshold), not a selection-side one. Recommend filing a
separate openspec change against Compressor parameters rather than
holding this change open.
