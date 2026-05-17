# design — principled-feature-selection-at-encoding-cap

## Context

Polygram encodings have hard caps on feature count: `MPSRung1` at 8,
`Rung3` at 16, `Rung4` and `Rung5(n_amp_qubits=2)` at 32. Our SAEs
ship with 32 / 64 / 128 features depending on the feed. So we always
have to subset.

The current subset (`feat_ids[:cap]`) is *unintended*: it's the
shortest expression of "we have to pick some, pick whatever". For
`cascade__jumprelu` the consequence is dramatic — the Compressor
finds 21 pairs to merge in stage 5, then in stage 6 finds only 1
cluster because most of those pairs involved features whose IDs were
above the 16-cap.

A principled selector closes this gap. We don't need a clever learned
selector; we just need to stop using indexing-order as the criterion.

## Goals

- **Reproducibility**: every run records its kept feature IDs so the
  Dictionary it built is rebuildable.
- **Apples-to-apples comparison** between selectors: same SAE, same
  encoding, varying only the selection method should show up cleanly
  in the scoreboard.
- **No new dependencies**: the work uses `auc_matrix` we already have.

## Non-Goals

- **Learned feature selection**: a small classifier trained to pick the
  best 16 features for downstream tasks is interesting research, but
  it's not the right intervention before we've measured what the
  heuristic selectors do.
- **Hierarchical / clustering-based selection**: e.g. pick a
  representative from each cluster of similar features. Deferred — the
  Compressor already does cluster-merging downstream; doing it twice
  is duplicate work.
- **Per-encoding-cap autotuning**: e.g. dynamically choosing `MPSRung1`
  vs `Rung3` based on how many features pass a firing-rate threshold.
  Separate concern; this change is just about the selection step.

## Decisions

### 1. Three built-in selectors only

`head` (current behaviour, kept for reproducibility), `firing_rate`
(most-active features), `gt_alignment` (most-benchmark-useful features).
Three is enough to compare and small enough to maintain. Future
selectors can land as separate proposals once there's evidence we need
them.

### 2. Default to `firing_rate`, not `gt_alignment`

`gt_alignment` is tempting as the default because it directly optimises
for what the benchmark measures. Rejected because: (a) it overfits the
selector to the eval metric, which makes the scoreboard's Axis-B
numbers less informative (we'd be partially measuring "did we pick
features that already align well", not "did the SAE find well-aligned
features"); (b) `firing_rate` is encoding-agnostic and would
generalize to real LLMs where GT labels don't exist; (c) firing-rate is
the standard heuristic in the SAE-interpretability literature.

### 3. Why record `kept_ids` in the result payload

A future reader (or a re-run six months from now) needs to be able to
reconstruct exactly which features the Dictionary saw. Selectors are
deterministic given their tie-break rule, but recording the kept IDs
explicitly removes any room for ambiguity if the selector
implementation changes.

### 4. Ordering and tie-breaking

Selectors return a full-ordered list, not just the top-N. This keeps
the abstraction reusable (callers can apply their own `[:cap]`) and
makes determinism explicit. Tie-break: feature ID ascending — picks
the lower-ID feature when the criterion is equal.

## Risks

- **GT-alignment selector is benchmark-specific**: only meaningful on
  fixtures with known labels. The selector raises a clear error if
  called with a feed that doesn't have a GT matrix, rather than
  silently returning garbage.
- **Firing rate can be dominated by stable particles**: in cascade
  feeds, photons / electrons / neutrinos fire on essentially every
  sample. A pure firing-rate ranking might pick "fires often" over
  "fires informatively". Mitigation: log a per-selector summary in the
  result payload so we can spot this happening. If it becomes a
  problem, the obvious next selector is `entropy` (features whose
  firing pattern carries the most information about samples).
- **`gt_alignment` ranking depends on the AUC metric**: a feature can
  fire on the right things but with the wrong magnitude pattern and
  still score well under AUC. Acceptable for v1; flag in the proposal
  as a known limitation.
