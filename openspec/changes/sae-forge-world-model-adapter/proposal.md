# sae-forge-world-model-adapter

> **Scope note**: this change is an *upstream* ask of
> [jascal/sae-forge](https://github.com/jascal/sae-forge) with sm-sae as
> the driving consumer. The spec lives here because sm-sae is the
> downstream that motivates it and will retire local scaffolding once
> it lands. Implementation lands in sae-forge; sm-sae files a follow-up
> after that to remove `smsae.host` and `scripts/train_cascade_host.py`.

## Why

`ForgePipeline.run_synthetic` projects a host transformer's weight
matrices (Q/K/V/O attention, FFN up/down, embeddings) into a polygram
feature basis via `SubspaceProjector` + a family adapter
(`GPT2Adapter`, `LlamaAdapter`, …). This pipeline is implicitly
transformer-only.

For sm-sae, the substrate is the SM cascade — a stochastic decay
simulator with no weight matrices. The current workaround
([[add-cascade-host-shim]]) trains a tiny GPT-2 to predict next-state
cascade transitions and uses *that transformer's weights* as the host.
This makes the wiring work but adds three structural problems:

1. **Information loss at projection.** The trained host has ~100k
   params encoding cascade dynamics. The projector compresses those
   into a small feature basis (often 3–8 features post-Compressor),
   discarding most of what training bought you. The
   [[add-cascade-host-shim]] gate 7.3 (≥ 0.05 trained-vs-random
   faithfulness delta) is missed primarily for this reason.
2. **Translation overhead.** Cascade states are multisets of
   particles; cascade transitions are stochastic rewriting rules. The
   shim serializes these into token sequences for a transformer LM
   objective, which is a lossy re-encoding. The "trained host" learns
   a transformer-shaped approximation of cascade dynamics rather than
   evaluating cascade dynamics directly.
3. **Maintenance cost.** sm-sae carries `smsae.host`,
   `scripts/train_cascade_host.py`, and the load-then-fallback machinery
   in `_build_synthetic_host` purely to satisfy sae-forge's
   transformer-shape assumption. None of this would exist if sae-forge
   accepted non-transformer substrates.

The fix is to teach sae-forge about non-transformer substrates via a
`WorldModelAdapter` protocol. sm-sae then registers a
`CascadeWorldModel` that wraps `smsae.sm.cascade.cascade()` directly,
and the tiny-GPT-2 shim becomes deletable.

P1. Not urgent (the shim works for wiring), but the longer it sits the
more the Axis-C faithfulness numbers misrepresent what they're
actually measuring.

## What Changes

### Upstream (sae-forge)

- **New `saeforge.WorldModelAdapter` protocol** with these required methods:
  - `extract_features(input_ids: Tensor) -> Tensor` returning
    `(batch, seq, n_features)`. Replaces the family-adapter's
    `model.transformer(input_ids)` hidden-state extraction.
  - `n_features: int` property reporting the substrate's native
    feature dimension.
  - `project_into(basis: FeatureBasis) -> WorldModelAdapter` returning
    a new adapter whose `extract_features` output is shaped to the
    feature basis. Implementations can no-op this (substrate already
    matches), build a transformer-shaped synthetic via the existing
    `SubspaceProjector`, or do something substrate-specific.
- **`ForgePipeline.run_synthetic` accepts `world_model:
  WorldModelAdapter`** as an alternative to the current `host_model`
  argument. When both are passed, `world_model` wins. When only
  `host_model` is passed, behaviour is byte-identical to today
  (`host_model` is wrapped in a `TransformerHostAdapter` internally).
- **`SubspaceProjector` becomes optional** in the WorldModel path.
  `WorldModelAdapter.project_into` returns whatever shape the
  substrate prefers; the pipeline no longer assumes a Q/K/V/O
  projection is meaningful for every substrate.
- **Result schema gain**: `ForgeResult.host` becomes a discriminated
  union — `{family: "transformer", ...}` or `{family: "world_model",
  adapter_id: str, ...}`. Distinguishes in result artifacts whether a
  forge run used a real substrate adapter or the transformer-shaped
  fallback.
- **Backward compatibility**: the existing `GPT2Adapter` / `LlamaAdapter`
  / etc. become concrete `WorldModelAdapter` implementations under the
  hood. `TransformerHostAdapter` wraps any `transformers`-style model
  for the legacy entry point. No existing sae-forge consumer breaks.

### Downstream (sm-sae)

Filed as a separate follow-up `retire-cascade-host-shim` openspec
change once the upstream lands. Sketch:
- New `smsae.world_model.CascadeWorldModel` implementing
  `WorldModelAdapter`. `extract_features(input_ids)` runs
  `cascade()` from each particle-multiset input and returns the
  intermediate-state count vectors over the 61-particle vocab.
- `scripts/forge_pipeline.py:forge` calls
  `ForgePipeline.run_synthetic(world_model=CascadeWorldModel(...))`
  instead of building a synthetic GPT-2 host.
- Delete `smsae.host/`, `scripts/train_cascade_host.py`,
  `_build_synthetic_host`, the `runs/cascade_host/` artifacts, and the
  ten host-shim tests. Roughly -700 LOC.
- Scoreboard `host` column adds a third class: 🌐 `world_model`.
  🎓 trained and 🎲 random remain for back-compat with archived runs
  but become "legacy" in the prose.

## Capabilities

### New Capabilities (sae-forge-side)

- `world-model-adapter`: a substrate-agnostic protocol for what
  `ForgePipeline.run_synthetic` consumes. Transformers are one
  implementation; the SM cascade is another; future substrates
  (graph world models, agent dynamics, …) need only implement the
  protocol to drop in.

### Modified Capabilities (sae-forge-side)

- `forge-pipeline`: `run_synthetic` signature gains
  `world_model: WorldModelAdapter | None`; existing transformer path
  is preserved via a default `TransformerHostAdapter` wrapper.
- `subspace-projector`: still the default for transformer hosts; no
  longer mandatory in the pipeline. WorldModelAdapter implementations
  decide whether projection is meaningful for their substrate.
- `forge-result-schema`: `host` becomes a discriminated union.

### Modified Capabilities (sm-sae-side, filed in retire-cascade-host-shim)

- `forge-pipeline` (sm-sae): switches to the world-model entry point;
  retires `_build_synthetic_host`, `smsae.host`,
  `scripts/train_cascade_host.py`.
- `scoreboard-forge-pipeline-runs`: `host` column gains a 🌐
  `world_model` class; prose explains the retirement of the shim.

## Open questions

These belong in `design.md` once we have more information, but
flagging here so reviewers see them:

- **GroundTruthAlignment compatibility**: the current sm-sae
  `GroundTruthAlignment.score` reads `forged.torch_module.transformer(
  input_ids)` to get hidden states. With a WorldModel host, that
  attribute doesn't exist. Either (a) the FaithfulnessTarget receives
  the WorldModelAdapter and calls `extract_features` directly, or
  (b) the WorldModelAdapter exposes a `torch_module`-compatible
  shim. Decision deferred until the sae-forge API shape is fixed.
- **Polygram coupling**: polygram-compressed safetensors carry
  feature-basis metadata that today drives `SubspaceProjector`. If a
  WorldModel substrate-matches the basis natively (or if projection is
  skipped), does polygram's compressed-checkpoint format change? Best
  guess: no — the WorldModelAdapter still consumes a `FeatureBasis`
  via `project_into`, and the basis carries the same metadata.
- **Cascade rollout determinism**: `CascadeWorldModel.extract_features`
  needs to be deterministic given an input or the faithfulness
  numbers will be noisy across re-evals. Either fix the cascade RNG
  per call, or take the expected-value over many rollouts. To be
  decided in the downstream change.
