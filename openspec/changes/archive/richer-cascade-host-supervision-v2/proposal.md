# richer-cascade-host-supervision-v2

## Why

This is the bucket-C follow-up to `probe-full-gt-recoverability-cascade-host`
(merged this session as PR #20). The live probe ran on the existing
`runs/cascade_host/61/` baseline against the full 110-feature sm-sae
GT vocabulary at two layer depths and surfaced this pattern (2026-05-20,
n=5000 trajectories, 7932 probe samples):

**Strong (LM-only baseline AUC ≥ 0.95):** coarse-structural features
that the LM objective already encodes implicitly.

| feature | residual AUC |
|---|---|
| `kind:gauge` | 1.000 |
| `is_boson` | 1.000 |
| `particle:W+` | 1.000 |
| `charge_sign:+` | 0.997 |
| `kind:antineutrino` | 0.991 |
| `flavor:b` | 0.988 |
| `kind:lepton` | 0.985 |
| `is_charged` | 0.968 |
| `generation:3` | 0.966 |

**Weak (LM-only baseline AUC 0.74-0.85):** per-particle / per-color /
per-flavor identity features.

| feature | residual AUC | projected AUC | Δ |
|---|---|---|---|
| `color:r` | 0.740 | 0.721 | −0.019 |
| `flavor:tau` | 0.843 | — | — |
| `flavor:u` | 0.839 | 0.864 | +0.025 |
| `flavor:d` | 0.824 | 0.852 | +0.029 |
| `color:b` | 0.810 | 0.787 | −0.024 |
| `particle:s_b` | 0.800 | 0.795 | −0.005 |
| `particle:u_b` | 0.799 | — | — |
| `color:g` | 0.796 | 0.784 | −0.013 |
| `flavor:mu` | 0.775 | 0.803 | +0.028 |
| `particle:mu+` | 0.756 | 0.770 | +0.014 |

**Bucket assignment**: probe says "ambiguous" — neither strict A
(80% residual ≥ 0.9 AND 80% projected ≥ 0.9) nor strict C
(50% residual < 0.7) thresholds hit. **But the data tells a clear
story the heuristic misses**: 57% of measured features are at AUC ≥ 0.9
in BOTH residual and projected (matching closely, Δ within ±0.05), and
the remaining 43% are PER-PARTICLE / PER-COLOR / PER-FLAVOR identity
features sitting in the 0.74-0.85 range. The bucket assignment loses
this signal because it's a thresholded fraction; the per-feature
table is the load-bearing artefact and points squarely at v2.

**Key consequence**: the SAE encode preserves residual structure
almost exactly (typical Δ between residual and projected AUC is
±0.02). So the bottleneck is NOT the SAE encode (which is what
bucket B would imply); it's that the host's training objective leaves
fine-grained identity bits at MEDIUM strength rather than at ceiling.
Per-particle supervision — explicitly rewarding the host for
correctly representing each of the 17 ungauged-color quarks, 3
charged leptons, 3 neutrino flavors, and 8+ gluon color states —
is the falsifiable lever.

This is exactly the recipe econ-sae Phase 6.2 used at the
110-label scale (their regime-tier mAUC lifted from 0.595 to 0.991
via the dual-head + focal-BCE combination). sae-forge's
`add-concept-anchored-finetune` (now shipped on `main`) provides the
forge-side machinery; sm-sae's `aux-supervise-cascade-host` provides
the cascade-host machinery. What this change adds is the **right
label vocabulary at the right scale**.

## What Changes

### v2 aux vocabulary: 110 per-particle GT features

Replace the 5-label v1 aux vocabulary
(`total_charge_neutral`, `total_baryon_neutral`,
`originated_from_top`, `state_has_higgs`, `state_has_top`) with the
full sm-sae GT vocabulary surfaced by
`smsae.sae.data.all_ground_truth_features()`:

- 61 `particle:*` labels (one per SM particle)
- ~10 `kind:*` labels (quark/antiquark/lepton/.../gauge/higgs/gluon)
- ~12 `flavor:*` labels (u/d/s/c/b/t/e/mu/tau/...)
- 3 `color:*` labels (r/g/b)
- `chirality:*`, `generation:*`, `charge_sign:*`, `is_*` family

Per-state label derivation reuses `smsae.sae.data.particle_features()`
(already used by the probe). For each state, the 110-label vector is
the OR of per-particle feature sets across particles in the state.

### Aux head: per-channel + focal-BCE (econ-sae Phase 6.2 recipe)

The pooled head from v1 won't scale to 110 labels — pooled output of
shape `(n_embd, 110)` would have 6710 params for n_embd=61, and a
single linear head would struggle to disentangle the per-particle
distinctions the probe shows are linearly separable but
sub-discriminative.

Switch to:

1. **Per-channel head**: Each of the 110 labels gets its own scalar
   readout from a learned residual-stream slice (matches Phase 6.2's
   per-channel head pattern). Output shape `(B, T, 110)`.
2. **Focal BCE loss**: `focal_bce_loss(logits, labels, gamma=2.0)`
   from `saeforge.training.heads` (re-exported from
   `add-concept-anchored-finetune`). Handles the long tail of rare
   particles without `pos_weight` hacks.
3. **Optional dual head** (pooled + per-channel) per the full Phase
   6.2 recipe. The proposal scopes this as an opt-in `--dual-head`
   flag; v1 ships per-channel only.

### `train_cascade_host.py --aux-supervision per_channel`

Adds a new value to the existing `--aux-supervision` enum (currently
`{off, pooled}`; v2 raises `NotImplementedError` for
`per_channel` / `dual`). Implementing `per_channel` is the natural
next value.

### Backward compatibility

The v1 `pooled` mode stays available — useful for the 5-label
baseline comparison + the existing probe protocol. New default
remains `off`; users opt in via `--aux-supervision per_channel`.

## Acceptance gates

**Gate v2.1 (mechanical)**: aux head + focal-BCE loss trains to a
finite final loss in `≤ 60s` at n_embd=61 / 5 epochs / 2000
trajectories on Intel CPU. Output `runs/cascade_host/61_aux_v2/` has
the standard host weights + `aux_head` weights.

**Gate v2.2 (probe-lifting)**: re-run
`scripts/probe_full_gt_recoverability.py` against the v2 aux host.
The proposal SHALL ship when ALL THREE of the following hold:

1. **The 14 currently-weak per-particle features** (table above)
   each lift to AUC ≥ 0.92 in the v2-trained host's pooled residual.
   This is the falsifiable claim: the v2 aux head can lift per-particle
   features that v1 LM-only didn't.
2. **The 9 currently-strong features** (table above) stay at AUC ≥ 0.97
   (i.e., the v2 head doesn't degrade what was already at ceiling).
3. **`probe_full_gt_recoverability`'s bucket** becomes `A`
   (≥ 80% residual ≥ 0.9 AND ≥ 80% projected ≥ 0.9). The
   measurement is reported in the matching archive entry.

**Gate v2.3 (forge-faithfulness — the original gate 7.3)**: re-run
`scripts/forge_pipeline.py cascade__jumprelu --encoding rung5` against
the v2 aux host. Acceptance: trained-vs-random Δ ≥ 0.05 on
`forge_score`.

This is the ORIGINAL gate 7.3 target. v1 missed it at +0.0072; if v2
hits it, the entire `aux-supervise-cascade-host` lineage closes
positively. If v2 misses it BUT v2.1+v2.2 pass, the diagnosis pivots
again — the host now has the signal but it's getting dropped between
projection and the SAE faithfulness measurement. File a separate
`forge-projection-faithfulness-deep-dive` change.

## Capabilities

### Modified Capabilities

- `cascade-host-supervision`: aux vocabulary extends from 5 labels
  (v1) to ~110 labels (v2). Aux head switches from pooled to
  per-channel. Loss becomes focal BCE (γ=2.0). New
  `--aux-supervision per_channel` value on
  `scripts/train_cascade_host.py`.

## Impact

- `smsae/host/aux_labels.py` — extend `aux_label_names()` /
  `compute_aux_labels()` to produce the 110-feature vocabulary using
  `particle_features()`. The v1 5-label paths remain available behind
  a feature-flag for legacy comparisons.
- `smsae/host/tiny_gpt2.py` — generalise `aux_heads` parameter from
  scalar count to support per-channel readout. The current
  `nn.Linear(n_embd, aux_heads)` pattern works but we'll need to
  reshape labels accordingly.
- `scripts/train_cascade_host.py` — implement the `per_channel`
  branch of `--aux-supervision`; wire focal-BCE loss (re-import from
  `saeforge.training.heads`); update logging.
- `scripts/probe_full_gt_recoverability.py` — already supports the
  full vocab; no change needed for the verification probe.
- `smsae/polygram_bridge.py` — no change.
- Tests: `tests/test_aux_labels.py` extended for 110 labels;
  `tests/test_train_cascade_host_v2.py` (new) covers the per-channel
  head + focal-BCE loss path on a tiny model.

No breaking changes; v1 paths stay available.

## Risks / Trade-offs

- **Compute cost**: 110-head training is ~20× the v1 head size.
  Estimated training time at n_embd=61 / 2000 trajectories / 5 epochs:
  60-120 seconds on Intel CPU. Still cheap.
- **Aux head bandwidth**: 110 binary labels supervised by a single
  pooled hidden state may saturate. The per-channel head pattern
  splits the n_embd dimensions across labels (à la econ-sae's
  Phase 6.2) to give each label its own bandwidth. If saturation
  still occurs, the dual-head extension (separate per-channel + pooled
  heads) is the documented follow-up.
- **Label imbalance**: many of the 110 labels are rare (e.g.,
  `particle:t_b` appears in < 5% of states). Focal BCE γ=2.0
  handles this by down-weighting confident predictions; this is the
  exact Phase 6.2 pattern that worked at 0.991 mAUC scale.
- **Probe's gluon labels are degenerate**: 36/110 labels were
  unmeasurable in the probe (gluon color states, H, W-, Z appear
  too rarely or never in state_{t+1}). v2 SHALL skip these during
  training — supervising a label that's always 0 contributes no
  gradient. The 74 measurable labels are the realistic target.

## Out of scope

- **The dual-head extension** (per-channel + pooled simultaneously).
  v1 of this change ships per-channel only. Dual-head is the natural
  follow-up if per-channel saturates on a specific label cluster.
- **Refreshing labels mid-training**. Phase 6.2 used frozen labels;
  we follow.
- **The gluon-color-degeneracy fix** (re-engineering the cascade so
  state_{t+1} samples color octets more uniformly). Out of scope
  here; the 74-measurable subset is sufficient.

## Acceptance summary

This change ships when:

1. `--aux-supervision per_channel` trains to completion in ≤ 60s on
   Intel CPU (gate v2.1).
2. Probe v2 against the trained v2 host shows ALL 14 currently-weak
   per-particle features at AUC ≥ 0.92 AND no currently-strong
   feature regresses below 0.97 (gate v2.2).
3. `forge_pipeline.py cascade__jumprelu --encoding rung5` against the
   v2 aux host shows trained-vs-random Δ ≥ 0.05 (gate v2.3 — original
   gate 7.3).

Either outcome of v2.3 is informative — see the "If v2 misses but
v2.1+v2.2 pass" branch above for the v3 pivot.
