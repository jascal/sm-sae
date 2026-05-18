# design — sae-forge-world-model-adapter

## Context

sae-forge today (v0.4) is built around a hard assumption that the host
is a transformer:

```python
# pseudo-code of the current control flow
host = GPT2LMHeadModel(cfg)                            # transformer
basis = FeatureBasis.from_polygram_checkpoint(path)
projector = SubspaceProjector(basis)                   # Q/K/V/O reshape
pipeline = ForgePipeline(basis, projector, ...)
result = pipeline.run_synthetic(host, ...)
```

Every layer assumes the host has transformer weight tensors:

- `SubspaceProjector` reshapes attention and FFN matrices into the
  feature basis.
- The per-family adapters (`GPT2Adapter`, `LlamaAdapter`, …) tell
  `SubspaceProjector` which weight names to find.
- `ForgePipeline.run_synthetic` returns a `ForgedGPT2`-shaped object
  whose `.transformer(input_ids)` hands back hidden states; downstream
  faithfulness targets call into that.

For sm-sae the substrate is the SM cascade. The current shim
([[add-cascade-host-shim]]) trains a tiny GPT-2 to *imitate* cascade
dynamics, then hands the imitator to sae-forge. The shim is honest
scaffolding (the design.md for that change marks it as such) but it
introduces structural information loss that the gate-7.3 miss made
concrete.

The premise of this change: sae-forge should accept a
`WorldModelAdapter` — a protocol that produces feature-shaped
activations for an input, without committing to any particular weight
shape. Transformers become *one implementation*; the SM cascade
becomes *another*; future substrates plug in without forking sae-forge.

## Goals

- **Substrate-agnostic forge pipeline.** `ForgePipeline.run_synthetic`
  consumes anything implementing `WorldModelAdapter`, not just
  HuggingFace transformer models.
- **Zero breaking changes upstream.** Existing sae-forge consumers
  (sm-sae included, today) keep working byte-identically when they
  pass a transformer host.
- **Clean retirement path for sm-sae's shim.** Once this lands,
  `smsae.host` and `scripts/train_cascade_host.py` are deletable; the
  scoreboard's 🎓/🎲 split collapses to a single 🌐 row class for
  WorldModel-backed forges.

## Non-Goals

- **Defining what every WorldModel substrate looks like.** The protocol
  is minimal; implementations decide how to extract features from
  their dynamics. This change ships the *shape of the door*, not a
  catalog of substrates.
- **A new faithfulness target.** sm-sae's `GroundTruthAlignment` keeps
  doing what it does. The only adaptation is how it reads activations
  from the host — captured in the open-questions list, not solved
  here.
- **Polygram-side changes.** `FeatureBasis` and
  `from_polygram_checkpoint` stay as-is; WorldModelAdapter consumers
  decide how (or whether) to project into the basis.

## Decisions

### 1. Protocol surface, not a base class

A Python `Protocol` (`typing.Protocol`) rather than an ABC. Reasons:

- No inheritance requirement — existing classes (including HF
  transformer models wrapped in a `TransformerHostAdapter`) qualify
  structurally.
- Aligns with how sae-forge already type-hints `FaithfulnessTarget`
  (also a Protocol per the v0.4 pluggable-faithfulness work).
- Lets sm-sae's `CascadeWorldModel` implement the protocol without
  importing from sae-forge at all (avoiding a circular-dep when
  sm-sae tests stub it).

The protocol:

```python
class WorldModelAdapter(Protocol):
    n_features: int

    def extract_features(self, input_ids: Tensor) -> Tensor:
        """Return (batch, seq, n_features)."""
        ...

    def project_into(self, basis: FeatureBasis) -> "WorldModelAdapter":
        """Return a new adapter whose output is shaped to `basis`.
        May be a no-op for substrates that already match."""
        ...
```

### 2. `host_model` stays; `world_model` is additive

`ForgePipeline.run_synthetic(host_model=..., world_model=...)` accepts
either. When both are passed, `world_model` wins and `host_model` is
ignored (with a `UserWarning` so the inconsistency is loud, not silent).
When neither is passed, behaviour matches today (error or
documented-default depending on sae-forge's current handling).

This keeps every existing call site working. The migration is opt-in:
consumers add a `world_model=` kwarg when they want substrate-native
behaviour; everyone else keeps their current call.

### 3. Transformers become a WorldModelAdapter, not a sibling

`TransformerHostAdapter(host_model)` implements
`WorldModelAdapter`. `host_model=` passed to `run_synthetic` is wrapped
internally:

```python
def run_synthetic(self, host_model=None, world_model=None, ...):
    if world_model is None:
        world_model = TransformerHostAdapter(host_model)
    ...
```

This means there's exactly one code path downstream of the entrypoint:
the WorldModel path. The `host_model=` arg becomes thin syntactic
sugar over the wrapper. Reasons:

- Single code path is easier to test and to keep consistent.
- The pluggable-faithfulness work already established the pattern of
  "rewrap the legacy API into the new protocol."
- Future contributors don't have to think about whether their change
  affects "the host_model path" or "the world_model path" — there's
  only one.

### 4. `project_into` returns an adapter, not a tensor

The obvious shape would be `project_into(basis) -> Tensor`. Returning
an *adapter* instead lets the projection be lazy or stateful:

- A transformer adapter wraps a `SubspaceProjector` and applies the
  projection at `extract_features` call time.
- A native-substrate adapter (e.g. `CascadeWorldModel`) can return
  `self` if its output already matches the basis dimension.
- A "synthesize a transformer first" adapter (the cascade-host shim
  pattern, if we want to keep it as a fallback) can return a new
  adapter that wraps the synthesized transformer.

This generalises cleanly. A pipeline implementation looks like:

```python
projected = world_model.project_into(basis)
for batch in eval_batches:
    feats = projected.extract_features(batch)
    target.score(features=feats, ...)
```

No special-casing per substrate.

### 5. Discriminated union in the result schema

`ForgeResult.host` becomes:

```python
{"family": "transformer", "model_id": "gpt2", "n_params": 124_000_000, ...}
# or
{"family": "world_model", "adapter_id": "smsae.world_model.CascadeWorldModel",
 "n_features": 61, ...}
```

The legacy stringly-typed `host` field is removed in the same release.
This is the only breaking change in the upstream API surface; the
migration is mechanical (parse the dict, switch on `family`).

## Risks

- **GroundTruthAlignment compatibility** is the largest unknown.
  sm-sae's faithfulness target reads `forged.torch_module.transformer(
  input_ids)` to get hidden states. With a WorldModel host, that path
  doesn't exist. Two viable resolutions:
  - **(a)** sae-forge passes `extract_features` results directly to
    `FaithfulnessTarget.score(features=...)`, decoupling the target
    from the host shape. Cleaner; requires changes to the
    `FaithfulnessTarget` protocol.
  - **(b)** `WorldModelAdapter` exposes a `torch_module` property that
    returns an `nn.Module` whose `forward` is `extract_features`.
    Preserves the target signature; slightly more invasive on the
    adapter side.
  Recommendation: (a). Defer the decision until upstream review.

- **WorldModel ⇄ FeatureBasis dimensional mismatch.** Today a
  transformer host's `n_embd` is forced to equal the SAE's `input_dim`.
  A WorldModel might have a *fixed* feature dimension that doesn't
  match the SAE basis. `project_into` is the contract for handling
  this, but if the substrate dimension is < basis dimension the
  adapter has to up-project or pad, both of which lose information
  asymmetrically vs the current down-projection case. Worth a
  validation step + a clear error message.

- **Coordination with sae-forge maintainer.** This is a non-trivial
  upstream change in someone else's repo (also yours, in this case —
  `jascal/sae-forge`). Recommended sequencing: file this proposal,
  link it in a sae-forge issue, agree on the protocol shape, then
  implement upstream. sm-sae's `retire-cascade-host-shim` follow-up
  blocks on the sae-forge release that ships this.

## Migration plan

1. sae-forge ships `WorldModelAdapter` + `TransformerHostAdapter` +
   updated `run_synthetic` signature in a minor release (v0.5).
2. sm-sae's `retire-cascade-host-shim` follow-up lands, pinning
   `saeforge>=0.5`, adding `CascadeWorldModel`, and deleting
   `smsae.host` + `scripts/train_cascade_host.py`.
3. Scoreboard rebuild: rows that ran under the trained-shim era
   (`host.kind == "trained"`) keep their data and the 🎓 marker but
   are flagged "legacy"; new rows render 🌐 `world_model`.
4. `add-cascade-host-shim` gets archived; its gate 7.3 is closed
   "obsoleted by world-model adapter" with a cross-link here.
