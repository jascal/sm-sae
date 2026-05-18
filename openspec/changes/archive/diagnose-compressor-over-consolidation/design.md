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
`inspect.signature` against the installed polygram; upstream source at
[github.com/jascal/polygram](https://github.com/jascal/polygram)):

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

## Expected vs observed cluster counts

What we see today vs what a non-degenerate Compressor should produce on
this fixture, given the ValidationReport's `confirmed` pair count and
the encoding cap:

| run_id | encoding cap | confirmed pairs | observed clusters | naive expected* |
|---|---|---|---|---|
| `cascade__jumprelu` (head)         | 16 | 21 | 1 | 8–10 |
| `cascade__jumprelu` (firing_rate)  | 16 | 20 | 3 | 8–10 |
| `cascade__jumprelu` (gt_alignment) | 16 | 26 | 2 | 6–8  |
| `embedded__topk` (firing_rate)     | 16 |  7 | 4 | 9–11 |

*Naive expected = `cap - min_disjoint_pair_cover`, i.e. the cluster
count you'd get if each confirmed pair merged exactly one feature into
its partner and the rest were singletons. Real Compressor output should
be ≤ naive but should track it; today it sits at 25–50% of the naive
floor.

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

The pinned 12-cell grid (`scripts/compressor_sweep.py` walks this in
order):

| # | strategy | rep_selection | merge_mode      | score_field      | confirmer | rationale |
|---|---|---|---|---|---|---|
| 1  | merge | scale_aware | freq_weighted   | polygram_overlap | none | current default — baseline |
| 2  | merge | scale_aware | argmax          | polygram_overlap | none | tests whether `freq_weighted` is the over-merger |
| 3  | merge | scale_aware | freq_weighted   | jaccard          | none | tests whether `polygram_overlap` is overcrowding the score |
| 4  | merge | by_overlap  | freq_weighted   | polygram_overlap | none | swaps representative-picker only |
| 5  | merge | by_overlap  | argmax          | polygram_overlap | none | both swaps; isolates additivity |
| 6  | merge | first       | argmax          | polygram_overlap | none | minimal-machinery sanity row |
| 7  | merge | scale_aware | freq_weighted   | polygram_overlap | strict | gate-pass-aware confirmer (Hypothesis B counter-test) |
| 8  | merge | scale_aware | argmax          | jaccard          | strict | combined relaxation |
| 9  | prune | scale_aware | —               | polygram_overlap | none | non-merge strategy — does it consolidate less? |
| 10 | prune | by_overlap  | —               | polygram_overlap | none | prune variant |
| 11 | merge | scale_aware | freq_weighted   | polygram_overlap | none | **+ vreport patch**: `kl_log_ratio_abs ~ N(0.1, 0.05)` |
| 12 | merge | scale_aware | freq_weighted   | polygram_overlap | none | **+ vreport patch**: drop `kl_*` fields entirely if API allows |

Cells 11 and 12 are how Hypothesis B is tested: same default config as
cell 1, only the ValidationReport changes. If cells 11/12 jump from 3
to ≥ 6 clusters, B is the dominant cause. If cells 2–10 jump and cells
11/12 do not, A is.

### 4. "Prove unreachable" needs explicit criteria

Acceptance gate 7.2 (in `tasks.md`) lets us close this change by
*either* finding a config that hits ≥ 4 clusters on
`cascade__jumprelu` *or* proving ≥ 4 is unreachable on this fixture.
The second branch is only valid if:

1. The full 12-cell grid above has been run end-to-end (no cell
   skipped for unrelated errors).
2. The polygram source for the merging step has been read (we want a
   confident statement, not "we tried some things and none worked").
3. The argument for why ≥ 4 is unreachable is written into
   `design.md`'s Investigation section, citing the specific
   transitive-closure / threshold behaviour observed.
4. PR #1's gate 8.4 is then explicitly revised in its `tasks.md` with
   a cross-link back to this section, so the gate revision has a paper
   trail rather than disappearing.

## Open questions

- Is the `0.0` `kl_log_ratio_abs` in our synthesized vreport
  filtered-out (treated as "no signal") or treated-as-evidence
  (treated as "score=0 → all-pairs-tied → merge-everything"?
  Hypothesis B hinges on this.
- Does polygram's Compressor expose any cluster-size cap (e.g.
  `max_cluster_size` or a transitive-closure stopping criterion)? Not
  in the public `CompressionConfig`, but worth checking the source.
