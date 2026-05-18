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

    def extract_features(self, input_ids: Tensor,
                         seed: int | None = None) -> Tensor:
        """Return (batch, seq, n_features).

        `seed` is honoured by stochastic substrates (e.g. cascade
        rollouts). Deterministic substrates ignore it. When None,
        adapters with internal RNG default to their constructor seed
        — never to wall-clock entropy — so a re-eval at the same
        input is reproducible.
        """
        ...

    def project_into(self, basis: FeatureBasis) -> "WorldModelAdapter":
        """Return a new adapter whose output is shaped to `basis`.

        - If `self.n_features == basis.dim`, may return `self` (no-op).
          This is the fast path for substrates that already match.
        - If `self.n_features > basis.dim`, project down (e.g. via
          `SubspaceProjector` for transformer adapters).
        - If `self.n_features < basis.dim`, the adapter MUST raise
          rather than silently up-pad. Up-projection introduces
          information asymmetric to down-projection and almost
          always indicates the SAE basis is wider than the
          substrate can faithfully represent.
        """
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

### 5. Stochastic substrates carry their own RNG

`WorldModelAdapter.extract_features` accepts an optional `seed:
int | None = None`. Deterministic substrates ignore it; stochastic
ones (e.g. `CascadeWorldModel`, which calls `cascade()` under the
hood) honour it. When `seed=None`, adapters with internal RNG
default to their *constructor seed*, never to wall-clock entropy.

Rationale:

- A re-eval of the same `(adapter, input_ids)` pair must produce the
  same features, otherwise sweep results aren't reproducible and the
  scoreboard rows can't be re-derived from artifacts.
- Per-call `seed=` lets the caller take multiple rollouts when
  expected-value evaluation matters, without forcing every adapter to
  ship its own averaging machinery.
- Constructor-seed default keeps the common case (deterministic-feeling
  call) trivially reproducible.

Implementation sketch for `CascadeWorldModel`:

```python
class CascadeWorldModel:
    def __init__(self, ..., seed: int = 0):
        self._default_seed = seed

    def extract_features(self, input_ids, seed=None):
        rng = random.Random(seed if seed is not None else self._default_seed)
        # ... run cascade rollouts using `rng` ...
```

### 6. FaithfulnessTarget signature transition: dual-signature pattern

Resolves design.md Risks open-question (a), per reviewer feedback.

`FaithfulnessTarget.score` gains a `features=` keyword. Existing
targets that read `forged.torch_module.transformer(input_ids)` keep
working unchanged; new targets can opt into the cleaner signature.

```python
class FaithfulnessTarget(Protocol):
    name: str
    better_when: Literal["higher", "lower"]

    def score(self, *,
              forged=None,            # legacy: ForgedModel with .torch_module
              host=None,              # legacy: host wrapper
              ctx=None,               # legacy: pipeline context
              features=None,          # new: (batch, seq, n_features) Tensor
              ) -> tuple[float, float]:
        """Return (score, perplexity_analog).

        Implementations may consume `features=` (new path) or the
        `forged`/`host`/`ctx` triple (legacy path). sae-forge always
        passes both during v0.5–v0.6 so existing targets keep
        working; the legacy triple is removed in v0.7."""
        ...
```

Migration path for sm-sae's `GroundTruthAlignment`:

```python
def score(self, *, forged=None, features=None, ctx=None, **_):
    if features is not None:
        feats = features.mean(dim=1).cpu().numpy()   # new path
    else:
        # legacy path — preserved for v0.5–v0.6 transition window
        input_ids = ctx["_eval_input_ids"]
        hidden = forged.torch_module.transformer(input_ids)
        feats = hidden.mean(dim=1).cpu().numpy()
    ...
```

A two-release deprecation window (v0.5 ships both; v0.6 warns on
legacy; v0.7 removes) keeps the transition non-disruptive.

### 7. Discriminated union in the result schema

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

- **GroundTruthAlignment compatibility** — resolved in Decision 6 via
  the dual-signature `score(features=..., forged=..., ctx=...)`
  pattern with a v0.5→v0.7 deprecation window. Existing targets keep
  working; new targets opt into the cleaner `features=` path. Remaining
  unknown: whether the dual-signature window length is right for
  external sae-forge consumers we don't know about; revisit during
  upstream review.

- **WorldModel ⇄ FeatureBasis dimensional mismatch** — addressed in
  the protocol docstring (Decision 1): `project_into` MUST raise
  rather than silently up-pad when `n_features < basis.dim`. Down-
  projection (`>`) and no-op (`==`) cases stay implementation-defined.
  A `validate_basis_compatibility(adapter, basis)` helper in
  `saeforge.adapters` would catch this at pipeline-construction time
  instead of at the first eval batch; suggested but not required.

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
