# add-cascade-host-shim

## Why

`scripts/forge_pipeline.py:_build_synthetic_host` returns a
**random-init** tiny GPT-2 for sae-forge's `ForgePipeline.run_synthetic`.
The forged residual stream therefore carries no real signal from the SM
cascade dynamics that the SAE was trained on, and the
`GroundTruthAlignment` faithfulness numbers landing today
(`embedded__topk` → 0.892, `cascade__jumprelu` → 0.751) reflect *"did a
random projection happen to produce features whose max-over-features AUC
against the GT labels is high"*, not *"did the forge preserve cascade
structure"*. The benchmark's entire purpose is the second framing.

This is a P0 blocker for any scientifically meaningful claim about
forge-stage faithfulness on the sm-sae fixture. Every other axis of the
scoreboard (A, B, polygram-C) is comparable across runs today; Axis-C
forge-score is not.

## What Changes

- **New script `scripts/train_cascade_host.py`** that trains a tiny
  causal transformer to predict next-step cascade state from current
  state. Reuses `smsae.sm.cascade.cascade()` for trajectory generation.
  Outputs a host checkpoint that satisfies the shape `forge_pipeline.py`
  expects.
- **Standardised on-disk format**: `runs/cascade_host/<n_embd>/host.safetensors`
  + `runs/cascade_host/<n_embd>/config.json`. Keyed by `n_embd` so
  hosts built for different SAE `input_dim` values coexist.
- **`forge_pipeline.py:_build_synthetic_host` updated** to look for a
  trained host at the canonical path first, fall back to random init
  with a `UserWarning` when not found.
- **New `forge_results.json` field**: `host` becomes a structured dict
  with `kind: "trained" | "random_init"`, `n_embd`, `n_layer`,
  `n_head`, and (if trained) `train_loss_final` + `n_train_trajectories`.
- **Scoreboard "Forge pipeline runs" row gains a "host" column** that
  surfaces the kind + an icon (🎓 trained / 🎲 random) so a reader can
  immediately tell which rows are scientifically meaningful and which
  are wiring-only.
- **README update** documenting the train-host step as a required
  prerequisite for meaningful forge-score interpretation.

## Capabilities

### New Capabilities

- `cascade-host-shim`: a tiny causal transformer trained on SM cascade
  transitions, used as the in-memory host for
  `saeforge.ForgePipeline.run_synthetic`. Distinct from the substrate
  (the SM bundle itself, which is not a transformer) and from the SAEs
  (which consume cascade *outputs*, not transitions).

### Modified Capabilities

- `forge-pipeline`: `_build_synthetic_host` becomes load-then-fallback
  instead of always-random. The pipeline's behaviour stays
  byte-identical when no trained host exists (clear `UserWarning`
  surfaced).
- `scoreboard-forge-pipeline-runs`: gains a `host` column; downstream
  prose updates to flag trained-vs-random as a first-class
  interpretation knob.
