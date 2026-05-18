# principled-feature-selection-at-encoding-cap

## Why

`scripts/forge_pipeline.py:build_dictionary` currently caps the SAE's
feature list at the polygram encoding's `max_features` by taking the
first N in feature-id order (`feat_ids = feat_ids[:cap]`). Feature IDs
in our SAE checkpoints are essentially arbitrary — they reflect the
order the SAE training loop happened to allocate features, not anything
about feature utility. For `cascade__jumprelu` (128 SAE features →
Rung3's 16-feature cap), this discards 112 features chosen at random
and feeds an arbitrary subset to the Compressor.

Concrete consequence: on `cascade__jumprelu`, stage 5 finds **21
confirmed pairs** in the synthesized ValidationReport, but stage 6's
Compressor only produces **1 cluster** — because the arbitrary slice
through the 128 features happens not to contain many of the pairs the
ValidationReport identified. We're hiding most of the SAE's structure
from the downstream pipeline by accident of indexing.

This is a P1 cleanup: cheap to fix, materially affects what the
Compressor and forge stages can actually see and merge.

## What Changes

- **New CLI flag** `--select-by {firing_rate, gt_alignment, head}` on
  `scripts/forge_pipeline.py`. Default: `firing_rate`.
- **`build_dictionary` accepts a feature-selection callable** that maps
  `(sae, feed, records) -> list[int]` returning the IDs to keep, in
  preferred order. Cap is applied after selection, not before.
- **Three built-in selectors**:
  - `firing_rate`: mean `(z > 1e-9)` over the feed; descending sort.
    Captures features that actually fire on real inputs.
  - `gt_alignment`: max-AUC across GT labels for each feature
    (reuses `auc_matrix` from `visualize.py`); descending sort.
    Captures features most useful for the benchmark.
  - `head`: the current behaviour — first N by feature_id. Retained
    for reproducibility of pre-change runs.
- **Selection method recorded in `forge_results.json`** under
  `selection.method` and `selection.kept_feature_ids` so the scoreboard
  can annotate and so re-runs are deterministic.
- **Scoreboard table gains a "selection" column** (or annotates the
  encoding column) so the reader can see at a glance whether a run
  used `firing_rate` vs `gt_alignment` vs `head`.

## Capabilities

### New Capabilities

- `feature-selection-at-encoding-cap`: a pluggable strategy that picks
  the top-`cap` SAE features for the polygram Dictionary. Three
  built-in strategies; the selection is recorded in run artifacts for
  reproducibility.

### Modified Capabilities

- `forge-pipeline`: `build_dictionary` signature gains a
  `selector: Callable | str` parameter (string maps to a built-in;
  callable for user code). Default selector becomes `firing_rate`.
- `scoreboard-forge-pipeline-runs`: the per-row encoding annotation
  now records which selector was used.
