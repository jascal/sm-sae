# design — add-cascade-host-shim

## Context

sae-forge's `ForgePipeline.run_synthetic` projects a host transformer's
weight matrices into a polygram-derived feature basis, producing a
"native" small transformer whose residual stream *is* the SAE feature
space. The faithfulness target then scores how well that native model
behaves like the host on a held-out distribution.

For real LLMs, the host is something like GPT-2 — a transformer
pre-trained on a real corpus, whose weights encode genuine linguistic
structure. The native model inherits a meaningful residual stream.

For sm-sae, the substrate is the SM cascade — not a transformer. Today
we work around this by passing a random-init tiny GPT-2 as the
"synthetic host". The native model inherits noise. `GroundTruthAlignment`
scores against SM physical labels (`origin:t_b`, `color:r`, etc.) and
returns numbers in the [0.7, 0.9] range that come from the max-over-61-
features pooling stumbling onto coincidental correlations, not from any
real structure preservation. The wiring is correct; the numbers are not
interpretable.

To produce meaningful per-row Axis-C numbers on the scoreboard, we need
a host that actually encodes cascade dynamics. The simplest such host:
a tiny transformer trained on next-state prediction over cascade
trajectories.

## Goals

- **Meaningful forge faithfulness numbers** on the sm-sae fixture: a
  forged model derived from a cascade-aware host should score
  differently (and ideally *better*) than one derived from random init.
- **No changes to sae-forge** required. This is purely sm-sae-side
  integration work.
- **Reproducible**: training runs deterministically from a seed,
  artifacts written to a canonical path.
- **Cheap**: full train + forge pipeline runnable in under 10 minutes
  end-to-end on a developer laptop for `n_embd=16`.

## Non-Goals

- **Multi-step roll-out evaluation** of the trained host. Single-step
  next-state prediction is the minimum viable training target. Roll-out
  faithfulness against trajectories is a separate follow-up if there's
  appetite.
- **A real cascade WorldModel** at the substrate level. The trained
  transformer is a *shim*: it gives sae-forge something to project into
  the feature basis. It is not a replacement for the cascade simulator.
- **Generalizing across `n_embd` values automatically**. Hosts trained
  at `n_embd=16` are only usable by SAEs with `input_dim=16`. One host
  per SAE-input-dim; trained on demand.

## Decisions

### 1. Why a transformer and not e.g. an MLP or RNN

sae-forge's `SubspaceProjector` and per-family adapters
(`GPT2Adapter`, `LlamaAdapter`, etc.) expect transformer-shaped weight
matrices. Using a non-transformer host would require either writing a
new adapter (a sae-forge-side change tracked separately, see the
`WorldModel` openspec prompt in sm-sae's chat history) or skipping the
projection entirely. The minimum viable shim is therefore the transformer
shape sae-forge already supports.

### 2. Vocabulary

61 SM particles + 1 PAD token = 62. The `PAD` token is special-cased
in the loss (`ignore_index=-100`); the model never has to predict it.
Alternative considered: vocab = 61, no PAD, training only on full-length
sequences. Rejected because cascade trajectories have wildly variable
length and packing them into fixed sequences is awkward.

### 3. Input encoding: count vector → sequence

Each cascade state is a multiset over particles. Encoding choices
considered:

- **count vector as a single embedding** (input_ids = [particle_idx],
  weighted by count): loses count structure, doesn't fit the transformer
  shape.
- **sequence of particles by count, canonical order** (current choice):
  preserves multiset structure, fits transformer naturally. Canonical
  (alphabetical) order so the model doesn't have to learn permutation
  invariance — that's a separable problem.
- **sequence with shuffling per step**: forces permutation invariance
  during training. Rejected for simplicity; canonical-order is a
  reasonable starting point.

### 4. Output encoding: per-position next-state token

Each position predicts what particle is in that slot of the next-state
multiset. Alternative: a single bag-of-particles output head per
sequence (one prediction per trajectory step). Rejected because it
doesn't match the autoregressive transformer's natural output shape;
position-wise prediction lets the standard GPT-2 lm_head do the work.

### 5. Why no roll-out evaluation in v1

Single-step training + faithfulness scoring is sufficient to get the
forge pipeline producing meaningful numbers. Roll-out evaluation (does
the trained host autoregressively reproduce full trajectories) is
genuinely interesting but adds compute and complexity; defer until v1
demonstrates non-trivial faithfulness improvement.

### 6. Per-`n_embd` host vs single shared host

A host trained at `n_embd=16` cannot be used by an SAE with
`input_dim=61` because the SubspaceProjector requires
`host.hidden_size == sae.input_dim`. Decision: keep one host file per
`n_embd`, indexed by directory name. Disk cost is negligible (~100k
params per host).

## Risks

- **Trained host might not improve faithfulness** measurably. If the
  transformer is too small or the next-state task is too easy /
  too hard, the projected weights might not differ meaningfully from
  random. Mitigation: training-loss curve is logged; if final loss is
  near the random-init loss, that's a clear signal to revisit
  architecture.
- **CPU training time**: with 2k–10k trajectories and a 2-layer
  transformer, training should fit in single-digit minutes. If it
  exceeds 10 minutes consistently, drop to 1 layer or shorten sequences.
- **HuggingFace `GPT2LMHeadModel` is heavy machinery** for what's
  effectively a tiny FFN+attention. Considered rolling our own minimal
  transformer (~50 lines of PyTorch); rejected because the HF model is
  exactly what sae-forge's `GPT2Adapter` expects, and the import is
  already required by `forge_pipeline.py`.
