# cascade-host-shim Specification (delta)

## MODIFIED Requirements

### Requirement: `tiny_gpt2` accepts an optional pooled aux head

`smsae.host.tiny_gpt2.tiny_gpt2(...)` SHALL accept a new keyword-only
argument `aux_heads: int = 0`. When `aux_heads == 0` (the default),
the constructed model is **byte-identical** to the pre-change
artefact: same submodules, same parameter count, same forward-pass
output for any input. When `aux_heads > 0`, an `nn.Linear(n_embd,
aux_heads)` SHALL be attached to the model as `model.aux_head`. The
aux head is invoked separately by the trainer; the model's standard
forward pass is unchanged regardless of `aux_heads`.

The aux head's parameter count is exactly `n_embd * aux_heads +
aux_heads` (weight + bias).

#### Scenario: `aux_heads=0` preserves byte-identity

- **WHEN** `tiny_gpt2(n_embd=61)` is called (omitting `aux_heads`)
- **THEN** the returned model has no `aux_head` attribute; total
  parameter count matches the pre-change `tiny_gpt2(n_embd=61)`
  exactly; the forward pass produces logits with the same shape and
  values it did pre-change

#### Scenario: `aux_heads=5` adds one Linear

- **WHEN** `tiny_gpt2(n_embd=61, aux_heads=5)` is called
- **THEN** `model.aux_head` exists and is an `nn.Linear` with
  `in_features == 61` and `out_features == 5`; the model's total
  parameter count exceeds the `aux_heads=0` call by exactly
  `61 * 5 + 5 == 310` parameters

#### Scenario: forward-pass shape unchanged

- **WHEN** the forward pass is invoked on a `tiny_gpt2(n_embd=61,
  aux_heads=5)` model
- **THEN** the returned `logits` tensor has the same shape as the
  `aux_heads=0` case (the aux head is consumed separately, not in
  the standard forward path)

### Requirement: trainer accepts `--aux-supervision` and `--aux-lambda`

`scripts/train_cascade_host.py` SHALL accept two new CLI flags:

- `--aux-supervision {off, pooled, per_channel, dual}` — default
  `off`. Only `pooled` is implemented in v1; `per_channel` and
  `dual` SHALL raise `NotImplementedError` at startup with an error
  message naming the follow-up changes
  (`per-channel-cascade-host-supervision`,
  `dual-head-cascade-host-supervision`).
- `--aux-lambda FLOAT` — default `1.0`. Ignored when
  `--aux-supervision off`.

When `--aux-supervision off`, the training loop, the data feed, and
the saved-model artefact SHALL be byte-identical to the pre-change
shim. When `--aux-supervision pooled`, the trainer:

1. Builds the model with `aux_heads = len(aux_label_names())` (5).
2. Consumes 3-tuple `(input_ids, target_ids, aux_labels)` batches
   from `cascade_transitions(..., with_aux=True)`.
3. Computes the pooled hidden state by averaging the final layer's
   hidden states across the sequence axis.
4. Computes loss `L = CE(token_logits, target_ids) + λ *
   BCE_with_logits(aux_logits, aux_labels)`.
5. Logs both loss components separately every 100 steps.
6. Saves the model via `save_pretrained`; the `aux_head` submodule
   ships with the rest of the state dict.

#### Scenario: `--aux-supervision off` preserves prior behaviour

- **WHEN** `scripts/train_cascade_host.py --n-embd 16
  --n-trajectories 100 --epochs 1` is run with `--aux-supervision`
  omitted (or explicitly `off`)
- **THEN** the resulting `runs/cascade_host/16/` directory contains
  the same model artefact (parameter shapes, param values modulo
  RNG-dependent training) the pre-change trainer would have written

#### Scenario: `--aux-supervision pooled` writes aux-aware artefact

- **WHEN** `scripts/train_cascade_host.py --n-embd 16
  --n-trajectories 100 --epochs 1 --aux-supervision pooled` is run
- **THEN** the run completes in < 60 seconds; the saved model has
  an `aux_head` submodule with `out_features == 5`; the run's
  `config.json` has `aux_supervision == "pooled"`

#### Scenario: unimplemented modes raise at startup

- **WHEN** `scripts/train_cascade_host.py --aux-supervision
  per_channel` (or `--aux-supervision dual`) is run
- **THEN** the process exits with `NotImplementedError` before
  consuming any training data; the error message names the relevant
  follow-up openspec change

### Requirement: `config.json` records the aux-supervision regime

`runs/cascade_host/<n_embd>/config.json` SHALL include the following
fields when written by this change's trainer:

- `aux_supervision: str` — one of `"off"`, `"pooled"`,
  `"per_channel"`, `"dual"`. Mirrors the `--aux-supervision` flag.
- `aux_labels: list[str]` — the output of `aux_label_names()` when
  `aux_supervision != "off"`; an empty list otherwise.
- `aux_lambda: float` — the `--aux-lambda` value used; `0.0` when
  `aux_supervision == "off"`.
- `aux_loss_final: float | null` — the last-recorded BCE-on-aux
  value at end of training; `null` when `aux_supervision == "off"`.

Downstream readers (notably `_build_synthetic_host`) SHALL tolerate
missing-key configs (pre-change hosts): missing fields default to
`aux_supervision = "off"`, `aux_labels = []`, `aux_lambda = 0.0`,
`aux_loss_final = None`.

#### Scenario: `off` mode writes the default sentinel fields

- **GIVEN** a trainer run with `--aux-supervision off`
- **WHEN** `config.json` is loaded
- **THEN** `aux_supervision == "off"`, `aux_labels == []`,
  `aux_lambda == 0.0`, `aux_loss_final is None`

#### Scenario: `pooled` mode records aux loss

- **GIVEN** a trainer run with `--aux-supervision pooled
  --aux-lambda 1.0`
- **WHEN** `config.json` is loaded
- **THEN** `aux_supervision == "pooled"`, `aux_labels == [the v1
  list]`, `aux_lambda == 1.0`, `aux_loss_final` is a non-negative
  float

#### Scenario: pre-change config is back-compatible

- **GIVEN** a `config.json` from a host trained before this change
  (no `aux_*` fields)
- **WHEN** `_build_synthetic_host` reads it
- **THEN** the returned `host_info` dict has `aux_supervision ==
  "off"`, `aux_loss_final is None`; no exception is raised

### Requirement: forge-pipeline surfaces aux-supervision in results

`scripts/forge_pipeline.py`'s `forge()` SHALL thread the new
`aux_supervision` and `aux_loss_final` fields from the loaded host's
`config.json` into `forge_results.json` under
`forge.host.aux_supervision` and `forge.host.aux_loss_final`.
Existing fields under `forge.host` are unchanged. Hosts without aux
training surface `aux_supervision == "off"` and `aux_loss_final ==
null`.

The aux head's weights are NOT used by sae-forge; only the
token-side weights are projected into the polygram feature basis.
The aux head is a training-side artefact and does not affect the
forged residual stream.

#### Scenario: aux-trained host's result records the regime

- **GIVEN** a host at `runs/cascade_host/61/` trained with
  `--aux-supervision pooled`
- **WHEN** `python scripts/forge_pipeline.py cascade__jumprelu` is
  run
- **THEN** `forge_results.json.forge.host.aux_supervision ==
  "pooled"` AND `forge_results.json.forge.host.aux_loss_final` is a
  positive float

#### Scenario: pre-change host's result preserves the legacy reading

- **GIVEN** a host from before this change (no `aux_*` fields in
  its `config.json`)
- **WHEN** `forge_pipeline.py` is run against it
- **THEN** `forge_results.json.forge.host.aux_supervision == "off"`,
  `aux_loss_final is null`; no exception is raised
