# tasks — richer-cascade-host-supervision-v2

## 1. v2 aux vocabulary

- [x] 1.1 Add `gt_aux_label_names()` to `smsae/host/aux_labels.py` —
      returns the full 110-feature GT vocabulary from
      `smsae.sae.data.all_ground_truth_features()`.
- [x] 1.2 Add `compute_gt_aux_labels(state)` — produces a binary
      label vector of length `len(gt_aux_label_names())`; each bit is
      1 iff any particle in the state carries the corresponding GT
      feature (OR over per-particle feature sets via
      `particle_features()`).
- [x] 1.3 Cache the `particle_name -> feature-set` map (lru-cached
      module-scope helper) so per-state computation is O(particles)
      not O(particles × vocab).

## 2. `cascade_transitions` extension

- [x] 2.1 Add `with_gt_aux: bool = False` flag to `cascade_transitions`,
      orthogonal to the existing `with_aux`. Mutually exclusive (raise
      `ValueError` if both True).
- [x] 2.2 When `with_gt_aux=True`, yield 3-tuples
      `(input_ids, target_ids, gt_aux_labels)` with the 110-binary
      vector computed by `compute_gt_aux_labels(state_{t+1})`.

## 3. `train_cascade_host.py per_channel` mode

- [x] 3.1 Remove `NotImplementedError` for `--aux-supervision per_channel`.
- [x] 3.2 When `per_channel`: build dataset with `with_gt_aux=True`;
      construct the model with `aux_heads=n_gt_features` (110); use
      focal-BCE loss instead of plain BCE.
- [x] 3.3 Add `--focal-gamma` CLI arg (default 2.0; matches Phase 6.2).
- [x] 3.4 Local `_focal_bce_loss` helper (10-line implementation;
      `gamma=0` reduces exactly to plain BCE).
- [x] 3.5 Write `focal_gamma` into the saved `config.json` metadata
      so downstream consumers can detect v2 vs v1.

## 4. Tests

- [x] 4.1 `test_gt_aux_label_names_is_full_gt_vocab` — vocabulary
      matches `all_ground_truth_features()` length + order.
- [x] 4.2 `test_compute_gt_aux_labels_shape_and_dtype` —
      `(110,)` float32, values in {0.0, 1.0}.
- [x] 4.3 `test_compute_gt_aux_labels_top_quark_state_hits_expected_features`
      — particle:t_r → flavor:t, color:r, is_colored, is_charged all 1.
- [x] 4.4 `test_compute_gt_aux_labels_empty_state_is_all_zero`.
- [x] 4.5 `test_compute_gt_aux_labels_zero_counts_ignored`.
- [x] 4.6 `test_compute_gt_aux_labels_multi_particle_or` — labels are
      the OR over constituent particles.

  All 6 new tests added on top of the existing 23-test aux suite.
  Total sm-sae test count: 67 → 74. All pass.

## 5. Gate measurements (live on this machine, 2026-05-20)

### v2.1 (mechanical): trains in ≤ 60s on Intel CPU

**RESULT: PASS** — trained in **17.5s** at n_embd=61 / 2000 trajectories /
5 epochs. Aux focal-BCE loss dropped 0.167 → 0.036 over the run.
Output: `runs/cascade_host/61_aux_v2/` with `host/` weights + `config.json`
recording `aux_supervision="per_channel"`, `aux_labels=` 110-element list,
`focal_gamma=2.0`.

### v2.2 (probe-lifting): re-run probe against v2-aux host

**RESULT: PARTIAL** — directional but doesn't hit strict 14/14 target.

Weak-feature lifts (from spec table; baseline → v2-aux):

| feature | baseline | v2-aux | Δ | ≥ 0.92? |
|---|---|---|---|---|
| `color:r` | 0.740 | **0.882** | +0.141 | no |
| `color:b` | 0.808 | **0.897** | +0.089 | no |
| `color:g` | 0.796 | **0.893** | +0.097 | no |
| `particle:s_b` | 0.800 | **0.927** | +0.127 | **yes** |
| `particle:mu+` | 0.756 | **0.855** | +0.099 | no |
| `flavor:u` | 0.839 | 0.875 | +0.036 | no |
| `flavor:d` | 0.823 | 0.864 | +0.041 | no |
| `flavor:mu` | 0.775 | 0.833 | +0.058 | no |
| `flavor:tau` | 0.843 | 0.851 | +0.008 | no |
| `particle:u_b` | 0.799 | 0.823 | +0.024 | no |

**3/14 weak features** hit ≥ 0.92 (target was all 14). **8/9 strong**
stay ≥ 0.97 (target was all 9; `is_charged` slipped from 0.968 to 0.968
— didn't move but didn't drop). Overall median residual AUC lifted
0.930 → 0.939; mean 0.902 → 0.924; pct ≥ 0.92 lifted 51% → 59%.

The color features especially responded well to supervision (+0.09 to
+0.14). But the per-flavor and per-particle features stayed in the
0.83-0.89 range despite explicit supervision.

### v2.3 (forge-faithfulness — the original gate 7.3): trained-vs-random Δ ≥ 0.05

**RESULT: FAIL** — v2 is actually **WORSE than random** by 0.0053.

| host | `cascade__jumprelu` rung5 forge_score | Δ vs random |
|---|---|---|
| **v2 aux** (per_channel, 110 labels) | **0.7257** | **−0.0053** |
| v1 aux (pooled, 5 labels) | 0.7382 | +0.0072 |
| LM-only baseline | 0.7262 | −0.0048 |
| random init | 0.7310 | — |

v2 regressed by 0.0125 vs v1. The 110-label aux head clearly steals
gradient bandwidth from the LM CE — final ce_loss went 1.05 (v1) →
1.10 (v2) — and that LM-side degradation propagates to forge
faithfulness.

## 6. Diagnosis (load-bearing finding)

Combining all evidence accumulated across the
`aux-supervise-cascade-host` → `probe-full-gt-recoverability` →
`richer-cascade-host-supervision-v2` arc:

- The LM-only cascade host (61-dim, 2-layer) already encodes coarse
  features at ceiling AUC (kind, charge_sign, gauge particles).
- The host encodes per-particle / per-color / per-flavor identity at
  MEDIUM strength (AUC 0.74-0.85) — adequate for some forge use, not
  ceiling.
- **Adding aux supervision (v1 or v2) does NOT lift forge
  faithfulness.** Both supervised runs miss gate 7.3 — v1 marginally
  ahead of random (+0.0072), v2 marginally behind (−0.0053).
- The aux head sharpens *labelled* features (color AUCs +0.09 to +0.14)
  but degrades LM accuracy at this scale (n_embd=61, 2 layers, 500
  training steps).

**The binding constraint is host capacity, not supervision objective.**
Per-particle / per-color identity needs more residual-stream
bandwidth than 61 dimensions can carry alongside the LM
next-token-prediction objective. Aux supervision shuffles which
features get sharpened but can't add net signal because there's
nowhere to put it.

## 7. Recommendation for follow-up

This change closes the gate-7.3 lineage with a SET OF NEGATIVE RESULTS
that pinpoint the actual bottleneck:

- **DO NOT** file v3 with even richer labels — both v1 and v2 prove
  more labels ≠ better forge faithfulness at this host capacity.
- **DO** file `investigate-cascade-host-capacity-sweep` as the next
  experiment: sweep n_embd ∈ {61, 96, 128, 192} and n_layer ∈ {2, 4}
  WITHOUT aux supervision; measure both gate 7.3 (forge score) and
  the probe (per-feature AUC). The hypothesis to test: forge score
  scales with host capacity (n_embd × n_layer), and the per-particle
  AUC gap closes by capacity alone.
- If the capacity sweep shows the binding constraint IS host size,
  the supervised-aux machinery (this change, v1, the probe) stays as
  diagnostic tooling but isn't the lever for closing the gate.

## 8. Archive trigger

This change SHALL be archived (without merging spec deltas — there
are none; this is a negative-result change) once the
`investigate-cascade-host-capacity-sweep` proposal is filed.
