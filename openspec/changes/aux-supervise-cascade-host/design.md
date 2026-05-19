# design — aux-supervise-cascade-host

## Context

The cascade-host shim (`smsae.host.tiny_gpt2` + `scripts/train_cascade_host.py`)
trains a tiny GPT-2 on cascade-transition next-state prediction. It's
the host that sae-forge's `ForgePipeline.run_synthetic` projects into
the polygram feature basis to produce the forged residual stream the
Axis-C `GroundTruthTarget` scores against the sm-sae GT label matrix.

Today's loss:

```
L = cross_entropy(logits[:, :-1, :], next_state_tokens[:, 1:])
```

Per-token, per-position. Standard LM training. The model is
incentivised to be a good cascade-transition language model and
nothing else.

The benchmark's question of interest is:
*does the forged residual stream encode features that align with
sm-sae's physical GT labels?*

The host's current loss never sees those labels (nor anything
correlated with them in particular). Whether they land in the
residual stream is a side-effect of the LM loss, not a contracted
property of it. The
[refreshed PR-#10 sae-forge section](../../../docs/index.html)
spells out the diagnosis and points at this change as the response.

## Goals

- **Move gate 7.3** (≥ 0.05 trained-vs-random faithfulness delta on
  at least one of `cascade__jumprelu` / `embedded__jumprelu`) from
  missed to met.
- **Preserve byte-identical behaviour** when aux supervision is off.
  No surprises for callers who don't opt in.
- **Stay minimal**. The 5-label v1 vocabulary is chosen to be
  small, easy to compute, and structurally diverse (one conservation
  pair, two lineage facts, two existence facts). Don't ship more
  until v1 has been measured.

## Non-Goals

- **A full conservation-supervision suite.** sm-sae's seven
  conservation algebra coordinates could each become an aux head.
  Until we know one such head moves the needle, scoping seven is
  speculative.
- **A separate "lineage feed".** The aux labels here are
  training-side only; they don't change what gets scored.
- **Architectural changes to the host.** No new attention block, no
  GRU, no MoE. The Q to answer is "does adding a single supervised
  signal change the outcome?", not "does a richer architecture?".
- **Closing the loop on per-channel supervision in this change.**
  Phase 5.2/6.2-style follow-ups get their own changes if needed.

## Decisions

### 1. Pool the residual stream before the aux head

Mean-pool over the sequence axis: `pooled = h.mean(dim=1)` →
`(B, n_embd)`. Then `aux_logits = Linear(pooled)` →
`(B, n_aux_labels)`. BCE against per-state binary labels.

Rationale:
- Matches econ-sae's Phase 5.1 recipe, which already lifted regime
  mAUC 0.60 → 0.97 in that setting. Per-channel and dual-head
  variants delivered marginal gains on top.
- Avoids the "which channel encodes which label" allocation question
  that Phase 5.2 had to solve.
- One Linear layer; negligible parameter cost (~`n_embd * n_aux`).

### 2. v1 label set: 5 binary labels

The five labels in the proposal cover three structural classes that
the SM cascade exposes:

| label | class | computed from | typical prevalence |
|---|---|---|---|
| `total_charge_neutral` | conservation | sum of charge over state | high (≥ 90%) — the simulator preserves charge by construction; the label is a sanity-check axis |
| `total_baryon_neutral` | conservation | sum of baryon number over state | high — same reason |
| `originated_from_top` | lineage | rollout-side metadata (parent particle) | low (~20%, since other heavy starts dilute it) |
| `state_has_higgs` | existence | `"H" in state` | very low — most cascades don't terminate with H surviving |
| `state_has_top` | existence | any of `t_{r,g,b}` in state | low, decreasing across the rollout |

This mixes high- and low-prevalence labels deliberately. If any one
of them shows the expected lift, we know aux supervision is doing
work; if specifically the rare-class labels (`originated_from_top`,
`state_has_higgs`) stall, that's the same class-imbalance pattern
econ-sae's Phase 6.1 hit and the fix (focal loss / pos_weight) is
already documented upstream.

The two high-prevalence conservation labels are intentional:
- They're cheap **sanity gradients**. Even when the next-state
  objective already pressures the model to represent token-level
  conservation implicitly, an explicit per-state target gives a
  small, stable signal that the pooled hidden state is correctly
  summarising the multiset.
- They're the **falsification axis** for the recoverability probe
  (Task 9.1). If the un-aux-trained host already recovers them at
  AUC > 0.95, then aux supervision on those specific labels is
  cosmetic — useful as a control, not as a lever.

Expected unsupervised recoverability is a prediction this change
will *measure*, not assume. The probe runs before training; if all
five labels are already recoverable at AUC > 0.9 from a non-aux
host, we expect a flat gate-7.3 outcome and the diagnosis pivots
elsewhere.

### 3. `λ = 1.0` default, override via `--aux-lambda`

Same scaling as econ-sae Phase 5.1, which empirically worked. Both
losses live on the same logit scale (cross-entropy on `vocab=62`
tokens, BCE on 5 binary labels) — order-of-magnitude similar. A
sweep over `λ ∈ {0.1, 1.0, 10.0}` is a small follow-up if v1 lands
near-miss.

Per-loss components are logged separately every 100 steps (see
[`tasks.md`](tasks.md) 4.3(e)) and the final BCE value is recorded
in `config.json` as `aux_loss_final`, so a near-miss can be
diagnosed as "balance off" vs "labels not learnable" without a
re-run.

### 4. Aux supervision is opt-in; default behaviour unchanged

`--aux-supervision` defaults to `off`. Existing
`scripts/forge_pipeline.py` runs that pick up the pre-aux host model
get the same behaviour they had in PR #10. New runs that opt into
aux supervision write to the same `runs/cascade_host/<n_embd>/`
directory, **overwriting** any prior host there — there's no
versioned subdirectory. Rationale: hosts are cheap to regenerate
(~15s); a "which host did this forge run use?" audit trail lives
in `forge_results.json.forge.host.aux_supervision`, not in the
filesystem.

### 5. Per-channel + dual-head live in follow-ups, not this change

If v1 pooled supervision clears gate 7.3, follow-ups are noise. If
v1 misses but moves the needle non-trivially (Δ ≥ 0.03 on
cascade__jumprelu vs the +0.0082 we have today), file
`per-channel-cascade-host-supervision` and try the Phase 5.2 recipe.
If v1 misses AND moves nothing, the diagnosis pivots: the issue
isn't the host's training objective at all, but some other layer
(probably the projection bottleneck reading more aggressively than
expected once aux info is present — TBD).

## Risks

- **Aux labels are computable from state.** A clever cascade-trained
  LM might learn to compute them implicitly from the token stream
  even without supervision. In that case the aux loss would only
  *strengthen* what's already there, not add anything new. Worth a
  diagnostic: post-v1, run the un-aux-supervised host through the
  same aux-label-prediction probe (linear classifier from pooled
  hidden state to each label, frozen host) and see how recoverable
  the labels are unsupervised. If most labels are already at AUC > 0.9
  without supervision, v1 won't move the gate.
- **Loss balance.** If `BCE(aux)` dominates `CE(token)` (or vice
  versa), the host degrades on its primary objective. `--aux-lambda`
  is the knob; monitor per-loss curves and report final values in
  config.json.
- **Pooling discards token-level structure.** A per-state label
  that's actually a per-position property (e.g. "this *token* is a
  top quark") becomes a "*some* token is a top quark" label after
  mean-pooling. For v1 this is fine — all 5 labels are genuinely
  per-state — but future labels with per-position semantics would
  need the per-channel or token-level variant.

## Migration

- v1 lands as additive opt-in. No migration required for existing
  runs.
- `runs/cascade_host/<n_embd>/config.json` schema gains four fields
  (`aux_supervision`, `aux_labels`, `aux_lambda`, `aux_loss_final`).
  Hosts trained before this change have those fields absent;
  `_build_synthetic_host` falls back to `aux_supervision="off"`
  when missing, preserving the legacy reading.
- The scoreboard's `host` column renders the legacy 🎓 marker
  unchanged when `aux_supervision=="off"`; gains a 🎓+aux variant
  when on.
