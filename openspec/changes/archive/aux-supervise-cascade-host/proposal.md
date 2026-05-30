# aux-supervise-cascade-host

## Why

[`add-cascade-host-shim`](../add-cascade-host-shim/) shipped a
cascade-trained tiny GPT-2 that
[`forge_pipeline.py:_build_synthetic_host`](../../../../scripts/forge_pipeline.py)
loads as the sae-forge host. The shim's training objective is
**pure next-state token prediction** (cross-entropy on per-position
particle logits). With the W_enc shape fixed (PR #5) and all 9 forge
cells re-run with trained hosts (PR #10), the measured
trained-vs-random faithfulness deltas are:

| cell | random | trained | Δ |
|---|---|---|---|
| `embedded__topk` | 0.8740 | 0.8892 | +0.0152 |
| `embedded__l1` | 0.8745 | 0.8762 | +0.0017 |
| `embedded__jumprelu` | 0.8494 | 0.8825 | **+0.0331** |
| `cascade__topk` | 0.7485 | 0.7461 | −0.0024 |
| `cascade__l1` | 0.6967 | 0.7327 | **+0.0360** |
| `cascade__jumprelu` | 0.7344 | 0.7262 | −0.0082 |

All directionally positive on average, but **none cross the
≥ 0.05 gate-7.3 threshold** that
[`add-cascade-host-shim`](../add-cascade-host-shim/tasks.md)
originally proposed. The largest delta is +0.036 on `cascade__l1`.

The diagnosis pointed at by both the
[refreshed sae-forge scoreboard section](../../../../docs/index.html)
and by [`econ-sae`'s Phase 5+ findings](https://github.com/jascal/econ-sae)
(transposed to this domain): an SAE substrate only encodes what its
training objective rewards encoding. **Next-state token prediction
doesn't demand that conservation totals or lineage facts land in the
host's residual stream.** They're computable from the input, but the
gradient pressure to allocate hidden dims to them is zero — once the
model has correctly predicted the next-state tokens, there's no
incentive to also represent "this state conserves charge" or "this
cascade originated from a top quark."

econ-sae's analogous experiment pushed regime-tier mAUC from 0.60 (no
aux) to 0.99 (aux-supervised, dual-head + focal loss recipe, Phase 6.2).
The first-pass single-pooled-head version (Phase 5.1) already hit
0.97, so the lift is mostly from *having* aux supervision, not from
the sophisticated recipe.

This change adds the minimum-viable supervised aux head to the cascade
host's training loop and re-measures gate 7.3. If the MVP doesn't
clear it, follow-ups (per-channel supervision, dual-head, focal loss
for class imbalance) get filed against this seam.

P1 — directly addresses the longest-open gate in the project.

## What Changes

- **New `scripts/train_cascade_host.py` flag** `--aux-supervision
  {off, pooled, per_channel, dual}`. Defaults to `off` so existing
  behaviour is preserved byte-identically. v1 implements `pooled`;
  the other two are scoped here as follow-ups.
- **`smsae.host.aux_labels`** (new module): computes per-state binary
  aux labels from a cascade state. v1 set:
  - `total_charge_neutral` — `sum(state)` over the charge axis equals
    0 (modulo numerical tolerance).
  - `total_baryon_neutral` — analogous for baryon number.
  - `originated_from_top` — true if any of the 3 colour-tagged top
    quarks (`t_r`, `t_g`, `t_b`) was the initial particle of the
    rollout.
  - `state_has_higgs` — true iff `H` appears in `state`.
  - `state_has_top` — true iff any of the 3 top quarks appears in
    `state`.
  These five labels cover one conservation pair, two lineage facts,
  and two existence facts. Per-state binary; same row count as the
  existing token-CE batch.
- **`smsae.host.tiny_gpt2.tiny_gpt2(aux_heads: int = 0)`**: adds an
  optional pooled-supervision head (mean-pool the residual stream
  across the sequence axis → Linear → `aux_heads` logits). When
  `aux_heads=0` (default) no head is constructed; the model is
  byte-identical to today.
- **`scripts/train_cascade_host.py` training loop**: when
  `--aux-supervision pooled`, the loss becomes `L = CE(next_state) +
  λ * BCE(pooled_aux_logits, aux_labels)` with `λ=1.0` default
  (`--aux-lambda FLOAT` override). When `off`, loss is unchanged.
- **`runs/cascade_host/<n_embd>/config.json` gains** `aux_supervision`,
  `aux_labels`, `aux_lambda`, `aux_loss_final` fields so the forge
  pipeline can surface "this host was aux-trained" in the result
  payload and the scoreboard.
- **`forge_results.json.forge.host` gains** `aux_supervision` and
  `aux_loss_final` mirroring the new host config; the scoreboard's
  `host` column annotates trained hosts as 🎓+aux when applicable.

## Capabilities

### Modified Capabilities

- `cascade-host-shim`: training loop accepts an opt-in aux-supervision
  head with a fixed 5-label vocabulary. Model architecture +
  inference path unchanged for `aux_supervision=off`; one new pooled
  head (one Linear + sigmoid) when on. Result artefacts grow two
  fields recording what aux-supervision was active.
- `scoreboard-forge-pipeline-runs`: the 🎓 marker becomes 🎓+aux when
  `host.aux_supervision != "off"`; the host-column aside gets an
  extra sentence on what aux-supervision means.

### New Capabilities

- `cascade-host-aux-labels`: a small module mapping cascade states to
  per-state binary labels for use as supervised targets in the host
  training loop. Five v1 labels covering conservation, lineage, and
  particle-presence. Pure-Python, no torch import.

## Out of scope (explicit non-goals)

- **Per-channel supervision** (econ-sae Phase 5.2 style, where last K
  hidden dims are reserved as direct label channels). The MVP pooled
  variant is sufficient to test whether aux supervision moves gate
  7.3; per-channel + dual-head + focal loss follow if needed.
- **Continuous-valued aux labels** (e.g. cascade-step count, fraction
  of generation-3 particles). v1 is binary-only to keep the loss
  uniform.
- **Token-level aux supervision** (predict aux labels per-position
  instead of pooled). Pooled is simpler and matches econ-sae Phase
  5.1's approach; per-token is a future follow-up if cleaner per-
  feature recovery turns out to need per-position structure. Going
  per-token would mean (a) reshaping the aux head from one Linear
  on pooled `(B, n_embd)` to a per-position Linear on
  `(B, T, n_embd)`, (b) deriving per-position labels (most v1
  labels are per-state, not per-token, and so don't even make
  sense per-position), and (c) re-balancing λ to account for the
  T-fold increase in per-batch aux loss magnitude.
- **Class-imbalance handling** (focal loss, pos_weight). Defer until
  the v1 measurement shows whether class imbalance is actually
  hurting any of the 5 labels.
- **A new openspec change for sm-sae's GT vocab**. The aux labels
  here are training-side only — they don't change what the SAE is
  scored against on the benchmark.

## Acceptance gate

The headline question this change exists to answer: **does pooled aux
supervision move the trained-vs-random faithfulness delta on
`cascade__jumprelu` to ≥ 0.05?**

If yes — gate 7.3 from
[`add-cascade-host-shim`](../add-cascade-host-shim/tasks.md)
is closed retroactively; the original change is fully delivered.

If no — file follow-ups (`per-channel-cascade-host-supervision`,
`dual-head-cascade-host-supervision`) against this seam following the
econ-sae Phase 5.2/6.2 recipe.

Either outcome is informative.
