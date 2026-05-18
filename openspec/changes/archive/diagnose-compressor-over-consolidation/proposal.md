# diagnose-compressor-over-consolidation

> **Resolution (2026-05-18, archived):** Closed-as-not-the-bug-we-thought
> by [sm-sae PR #5](https://github.com/jascal/sm-sae/pull/5). The
> over-consolidation was not a polygram Compressor tuning issue
> (Hypothesis A) nor a degenerate-vreport issue from
> `kl_ablate_*=0.0` (Hypothesis B). Root cause: sm-sae's
> `convert_to_safetensors` was writing W_enc with shape
> `(n_features, input_dim)` while polygram documents the canonical
> layout as `(input_dim, n_features)` (see
> `polygram/behavioural/validator.py:311`). With the axes swapped,
> polygram's `apply_merge` zeroed the wrong tensor slice, which
> silently worked while `n_features <= input_dim` and merged
> over-aggressively (or raised `IndexError`) once the SAE was
> overcomplete. PR #5 fixed the W_enc transpose on write and the
> cluster count on `cascade__jumprelu` jumped from 1 to 12 with no
> Compressor-side changes. The sweep machinery this proposal
> scoped was therefore never built; if a future regression looks
> like over-consolidation, re-open this with the actual diagnosis.

## Why

After [[principled-feature-selection-at-encoding-cap]] landed, the
Compressor still collapses the kept feature set far more aggressively
than the ValidationReport pair count would suggest:

| run_id | selector | confirmed pairs | clusters | kept |
|---|---|---|---|---|
| cascade__jumprelu | head        | 21 |  1 |  1 |
| cascade__jumprelu | firing_rate | 20 |  3 |  3 |
| cascade__jumprelu | gt_alignment | 26 |  2 |  2 |
| embedded__topk    | firing_rate |  7 |  4 |  4 |

With 20–26 confirmed pairs over 16 candidate features, the Compressor is
producing 1–3 clusters — effectively collapsing the whole basis to a
handful of representatives. The principled-selection change's
acceptance gate ("≥ 4 clusters on cascade__jumprelu with firing_rate")
was missed because of this, even though selection itself is now working
as intended.

Two possible root causes, both worth ruling in or out:

1. **Compressor params are wrong for our small-basis regime.** Our
   `compress()` call uses defaults (`strategy="merge"`,
   `rep_selection="scale_aware"`, `merge_mode="freq_weighted"`,
   `score_field="polygram_overlap"`). Polygram was tuned for
   GPT-2-scale SAEs; with 16 features, default merge thresholds may
   transitively pull most features into one big cluster (`a~b` and
   `b~c` ⇒ all of `{a,b,c}` collapse, even if `a` and `c` are weak).

2. **Our synthesized `ValidationReport` is mis-shaped.** In
   `synthesize_validation_report` we set `kl_ablate_*` and
   `kl_log_ratio_abs` to `0.0` (no host-ablation signal). If a
   Compressor strategy weights those fields, "everything ties at 0" is
   indistinguishable from "everything is the same feature."

Until we know which of these is dominant, the scoreboard's Axis-C
forge_score numbers reflect "the Compressor merged 13/16 features
into one survivor" rather than any real claim about cascade structure.
That makes Axis-C harder to interpret than it should be.

P1 — the wiring is end-to-end and the warning is honest, but the
forge-score column is currently noise-bounded by the cluster collapse.

## What Changes

- **`scripts/forge_pipeline.py:compress` accepts a `CompressionConfig`
  override.** Today it only forwards `strategy`. The signature becomes
  `compress(vreport, sae_path, out_path, *, config=None,
  strategy="merge")` and a `--compressor-config <json>` CLI flag lets
  users pass a one-off override (`{"rep_selection":"by_overlap",
  "merge_mode":"argmax"}`, etc.).
- **New script `scripts/compressor_sweep.py`** that, for a single
  (run_id, selector) pair, runs `compress()` across a small grid of
  `(rep_selection, merge_mode, score_field)` and reports
  `(n_clusters, n_kept, faithfulness)` per cell. Grid is hand-picked
  to ≤ 12 cells so the sweep finishes in a few minutes on CPU.
- **Diagnostic block in `forge_results.json`**: `compress` gains a
  `config` field recording the resolved `CompressionConfig` so two runs
  with different defaults are distinguishable in the scoreboard.
- **Investigation note** in `design.md` documenting (a) whether the
  `kl_ablate_*=0.0` shortcut in our synthesized ValidationReport is
  the cause and (b) which polygram parameter combination, if any,
  recovers a cluster count proportional to confirmed-pair count.
- **Scoreboard caveat**: until this investigation lands, add a one-line
  caveat under the Forge pipeline runs table naming the
  `cascade__jumprelu` fixture specifically (the case where the
  collapse is most visible — 20+ confirmed pairs → 1–3 clusters
  regardless of selector) and flagging that the cluster collapse is a
  known Compressor-tuning limitation, not a property of the
  kept-feature
  subset.

## Capabilities

### Modified Capabilities

- `forge-pipeline`: `compress()` becomes config-overridable; the CLI
  surfaces the override; the result payload records the resolved
  config.
- `scoreboard-forge-pipeline-runs`: gains a caveat paragraph; the
  `compress` cell may grow a `(config-fingerprint)` annotation when
  non-default config is used.

### New Capabilities

- `compressor-sweep`: a stand-alone diagnostic that varies polygram
  Compressor knobs against a fixed (SAE, selector, ValidationReport)
  and reports which knobs move the cluster count.
