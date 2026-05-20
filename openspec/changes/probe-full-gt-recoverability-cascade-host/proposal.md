# probe-full-gt-recoverability-cascade-host

## Why

`aux-supervise-cascade-host` (PR #19) shipped the pooled-aux MVP and measured gate 7.3 against the live cascade-trained host. Both the headline metric AND the diagnostic probe ran against the existing measurement infrastructure on this machine.

**Measured Δ on `cascade__jumprelu` (Rung5):**

| host | forge_score | Δ vs random |
|---|---|---|
| **aux-trained (pooled, v1)** | **0.7382** | **+0.0072** |
| LM-only trained baseline | 0.7262 | −0.0048 |
| random init (no host) | 0.7310 | — |

**Gate 7.3 missed.** Target Δ ≥ 0.05; observed +0.0072.

**Probe `scripts/probe_host_aux_recoverability.py` (5000 trajectories, frozen-host logistic-regression on pooled hidden state):**

| label | baseline (LM-only) AUC | aux-trained AUC | ΔAUC |
|---|---|---|---|
| `total_charge_neutral` | **0.9802** | 0.9975 | +0.0173 |
| `total_baryon_neutral` | **0.9948** | 1.0000 | +0.0052 |
| `originated_from_top` | **0.9967** | 1.0000 | +0.0033 |
| `state_has_higgs` | NaN (no positive samples even at n=5000) | NaN | — |
| `state_has_top` | NaN (no positive samples even at n=5000) | NaN | — |

Two findings:

1. **The three measured aux labels are already at AUC ≥ 0.98 in the LM-only host's pooled hidden state.** Next-state token prediction already encodes them implicitly — there's no headroom for supervised pressure to add anything load-bearing. Risk #1 from `aux-supervise-cascade-host/proposal.md` fired exactly as anticipated.

2. **`state_has_higgs` and `state_has_top` are structurally unmeasurable at this rollout scale** — both particles decay within 1 cascade step, so `state_{t+1}` after their parent's decay almost never contains them. The v1 vocab effectively had 3 usable labels (all at ceiling) and 2 degenerate ones.

The conclusion sm-sae's diagnostic protocol prescribes for this outcome (`aux-supervise-cascade-host/design.md` §5):

> If v1 misses AND moves nothing, the diagnosis pivots: the issue isn't the host's training objective at all, but some other layer (probably the projection bottleneck reading more aggressively than expected once aux info is present — TBD).

This change funds the "TBD" — a focused diagnostic that locates *which* layer of the forge pipeline is the gate-7.3 bottleneck before we file any per-channel / dual-head / focal-loss follow-ups (those wouldn't help — the host already has the signal the v1 vocab was supervising).

P1 — gate 7.3 remains the longest-open gate in the project, and the v1 aux experiment proved the host's training objective is **not** what's blocking it.

## What Changes

Adds **one new diagnostic script** and **two follow-up openspec proposals** (which this change does not implement — it just files them with the empirical evidence captured here).

### `scripts/probe_full_gt_recoverability.py` (new)

Extends `probe_host_aux_recoverability.py`'s pattern to the **full 110-feature sm-sae GT vocabulary** (not just the 5 aux-supervision labels):

- Reads the full GT feature vocabulary via `smsae.sae.data.all_ground_truth_features()`.
- For each GT feature, builds per-state binary labels from the cascade trajectory (sample-level: "does any particle in the state carry this feature?").
- Trains a frozen-host logistic-regression probe from the LM-only host's **pooled** hidden state to each GT feature; reports per-feature baseline AUC.
- Optional `--aux-host` flag computes ΔAUC, paralleling the existing probe.
- Optional `--from-projected` flag: probe from the **post-projection** hidden state (i.e., after `SubspaceProjector`) instead of the raw residual. Comparing the two answers the load-bearing question: *which layer is dropping the signal?*

Cheap (~5 min CPU). Output `runs/aux_probe/full_gt_recoverability.json` with per-feature AUCs.

### Acceptance criteria

The script outputs a 3-column table per GT feature:

| feature | baseline AUC (residual) | baseline AUC (post-projection) |
|---|---|---|

Three interpretation buckets and their implied follow-ups:

- **A. Residual ≥ 0.9, post-projection ≥ 0.9** for ≥ 80% of GT features → host *and* projection both carry the signal; **the bottleneck is the SAE itself** (JumpReLU + the polygram compression). File `investigate-cascade-jumprelu-sparsity-loss` to vary sparsity / encoding rung.
- **B. Residual ≥ 0.9, post-projection < 0.7** for ≥ 50% of GT features → host carries it but the projection drops it; **bottleneck is `SubspaceProjector` calibration** (scale_boost or basis rank). File `investigate-projection-bottleneck-cascade`. The `scale_boost='auto'` calibration may be selecting too aggressive a value for the cascade SAE's geometry.
- **C. Residual < 0.7 for ≥ 50% of GT features** → host doesn't actually carry the signal we thought; the LM objective is leaving out per-particle / per-flavor info the GT vocabulary needs. File `richer-cascade-host-supervision-v2` to redesign the aux vocabulary (per-particle labels, not coarse aggregates).

### Follow-up changes (file but don't implement)

Based on which bucket the probe lands in, one of the three follow-up changes above gets filed with the probe's per-feature AUC table as its empirical "Why."

### What this change explicitly does NOT do

- **No new aux-supervision recipe.** Per-channel and dual-head variants are pointless until we know whether the host has the signal at all (probe buckets B and C).
- **No model retraining.** The probe consumes the existing `runs/cascade_host/61/` baseline host plus the existing `runs/cascade_host/61_aux/` aux-trained host.
- **No GT vocabulary expansion.** The probe scores against the current 110-feature vocab from `all_ground_truth_features()`. v2 vocabularies are scoped under bucket C's follow-up.
- **No change to `aux-supervise-cascade-host` shipped behaviour.** That change's machinery (aux head, `--aux-supervision` flag, config.json schema, scoreboard rendering) stays as-is. The probe just adds a triage step.

## Capabilities

### New Capabilities

- `gt-recoverability-probe`: a diagnostic script that scores how linearly recoverable each sm-sae GT feature is from the cascade-trained host's pooled hidden state, at two layer depths (raw residual + post-projection). Reports per-feature AUC, optional ΔAUC against an aux-trained host, and a per-bucket interpretation summary that names the implicated layer.

## Acceptance

This change ships when:

1. `scripts/probe_full_gt_recoverability.py` runs against `runs/cascade_host/61/host` and writes `runs/aux_probe/full_gt_recoverability.json` with per-feature AUCs for all 110 GT features.
2. The output identifies which interpretation bucket (A/B/C) the host falls into.
3. The matching follow-up openspec change (`investigate-cascade-jumprelu-sparsity-loss`, `investigate-projection-bottleneck-cascade`, or `richer-cascade-host-supervision-v2`) is filed with the probe's per-feature table as its empirical "Why."

Either outcome of the probe is informative — there is no fail mode for this change.
