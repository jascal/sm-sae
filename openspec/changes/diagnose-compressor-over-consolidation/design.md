# design — diagnose-compressor-over-consolidation

## Context

The polygram Compressor takes a SAE checkpoint + ValidationReport and
produces a compressed safetensors file, with a report including
`n_clusters`, `n_features_kept`, `n_features_zeroed`. On our fixture it
behaves like this:

```
cascade__jumprelu (16 features, 20 confirmed pairs, firing_rate select)
  → clusters=3  kept=3  zeroed=10
embedded__topk (16 features,  7 confirmed pairs, firing_rate select)
  → clusters=4  kept=4  zeroed=7
```

Naively, 20 confirmed pairs spread over 16 features should yield ~8
disjoint pair-clusters, not 3. The Compressor is collapsing
transitively — `merge` strategy with `freq_weighted` mode appears to
pull near-overlapping features into a single representative even when
the chain is weak.

`CompressionConfig` exposes the relevant knobs (introspected via
`inspect.signature`):

```
CompressionConfig(
    strategy: str = "merge",
    rep_selection: str = "scale_aware",
    merge_mode: str = "freq_weighted",
    confirmer: str | None = None,
    target_n_features_kept: int | None = None,
    score_field: str = "polygram_overlap",
)
```

Today's `compress()` ignores all of these except `strategy`.

## Goals

- Establish whether the over-consolidation is a Compressor-config issue
  (Hypothesis A) or a side-effect of our `kl_ablate_*=0.0` shortcut in
  the synthesized ValidationReport (Hypothesis B).
- Make the Compressor config a first-class CLI knob so future tuning
  is one command away.
- Record the resolved config in `forge_results.json` so the scoreboard
  rows are interpretable.

## Non-Goals

- Re-implementing polygram's clustering. This change is diagnostic +
  configurability; if the conclusion is "default polygram clustering
  is wrong for small bases," that's a polygram-side issue and gets
  filed upstream.
- Adding host-ablation signal to the ValidationReport. That's a
  separate piece of work (lives near [[add-cascade-host-shim]]).
- Touching selection or scoring; both are settled by their own
  openspec changes.

## Decisions

### 1. Diagnostic before fix

Order of work is hypothesis-test → understand → wire. Picking a config
without first knowing which knob matters risks codifying a fragile
recommendation. The compressor_sweep.py script is the
hypothesis-testing tool.

### 2. CLI surface is JSON, not per-field flags

`--compressor-config '{"rep_selection": "by_overlap"}'` rather than
`--rep-selection by_overlap --merge-mode argmax --…`. Reasons: the
field set may grow upstream; we don't want sm-sae's CLI to track
polygram's API one field at a time; users running sweeps are
already comfortable with JSON.

### 3. Sweep cells are hand-picked, not cross-product

Cross-product of (4 rep_selection × 3 merge_mode × 2 score_field × 2
confirmer) = 48 cells, several of which are mutually exclusive or
upstream-deprecated. A hand-picked list of ≤ 12 known-meaningful
triples keeps the sweep tractable on CPU and the summary table
readable.

## Open questions

- Is the `0.0` `kl_log_ratio_abs` in our synthesized vreport
  filtered-out (treated as "no signal") or treated-as-evidence
  (treated as "score=0 → all-pairs-tied → merge-everything"?
  Hypothesis B hinges on this.
- Does polygram's Compressor expose any cluster-size cap (e.g.
  `max_cluster_size` or a transitive-closure stopping criterion)? Not
  in the public `CompressionConfig`, but worth checking the source.
