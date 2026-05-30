# tasks — probe-full-gt-recoverability-cascade-host

## 1. `scripts/probe_full_gt_recoverability.py`

- [ ] 1.1 Create the script. Structurally mirror `scripts/probe_host_aux_recoverability.py`: argparse CLI, builds a probe dataset from `cascade_transitions`, loads the host via `transformers.GPT2LMHeadModel.from_pretrained`, trains a frozen-host LogisticRegression probe per label, reports AUC.
- [ ] 1.2 Replace the 5-label aux vocabulary with the full sm-sae GT vocabulary. Read from `smsae.sae.data.all_ground_truth_features()` (returns a sorted list of ~110 strings).
- [ ] 1.3 Per-state label derivation: for each cascade state `state_{t+1}`, build a binary label per GT feature by checking whether **any** particle in the state carries that feature. Reuse `smsae.sae.data.particle_features(name)` for the per-particle → feature-set map.
- [ ] 1.4 Pooled-hidden-state extraction: identical to the existing probe (`out.hidden_states[-1].mean(dim=1)` under `torch.no_grad`).
- [ ] 1.5 **NEW**: optional `--from-projected` flag. When set, the probe additionally extracts the post-projection hidden state and produces a second AUC column.
  - Construct a `SubspaceProjector` from the cascade SAE checkpoint (`runs/cascade__jumprelu.pt`).
  - Apply `projector.encode(...)` to each pooled hidden state row.
  - Train a parallel LogisticRegression probe from the projected features to each GT label.
- [ ] 1.6 Output: `runs/aux_probe/full_gt_recoverability.json` containing:
  - `baseline_host`, `aux_host` (string paths), `n_trajectories`, `n_samples`, `seed`.
  - `gt_features: list[str]` (the 110 feature names in stable order).
  - `baseline_aucs_residual: dict[str, float | None]`.
  - `baseline_aucs_projected: dict[str, float | None] | None` (when `--from-projected` is set; null otherwise).
  - `aux_aucs_residual: dict[str, float | None] | None` (when `--aux-host` is supplied).
  - `aux_aucs_projected: dict[str, float | None] | None`.
  - `bucket_summary: dict` with the heuristic bucket label and per-bucket fractions.

## 2. Bucket interpretation logic

- [ ] 2.1 In the script, after collecting AUCs, compute:
  - `pct_residual_ge_0.9 = count(auc ≥ 0.9 in baseline_aucs_residual) / n_features`.
  - `pct_projected_ge_0.9 = same for projected` (when available).
  - `pct_residual_lt_0.7 = count(auc < 0.7 in baseline_aucs_residual) / n_features`.
- [ ] 2.2 Label the bucket via the proposal's rules:
  - Bucket A: `pct_residual_ge_0.9 ≥ 0.8 AND pct_projected_ge_0.9 ≥ 0.8`.
  - Bucket B: `pct_residual_ge_0.9 ≥ 0.8 AND pct_projected_ge_0.9 < 0.5`.
  - Bucket C: `pct_residual_lt_0.7 ≥ 0.5`.
  - Otherwise: "ambiguous" — surface all three fractions in the summary and recommend manual triage.
- [ ] 2.3 Print a top-line single-line summary at the end of the script:
  `"[probe] bucket={A|B|C|ambiguous}; residual ≥ 0.9: 87/110; projected ≥ 0.9: 12/110"`.

## 3. Run + analyse

- [ ] 3.1 Run `python scripts/probe_full_gt_recoverability.py --baseline-host runs/cascade_host/61/host --aux-host runs/cascade_host/61_aux/host --from-projected --n-trajectories 5000` against the existing artefacts on this machine.
- [ ] 3.2 Commit the output JSON to `runs/aux_probe/full_gt_recoverability.json` for traceability (gitignored by default; add a `runs/aux_probe/.gitkeep` and force-add this specific file).

## 4. Follow-up changes (file but don't implement)

Author the placeholder proposals corresponding to each bucket; populate only the one matching the probe's verdict. The others stay as drafts.

- [ ] 4.1 `openspec/changes/investigate-cascade-jumprelu-sparsity-loss/proposal.md` — bucket A. Investigate whether JumpReLU's sparsity threshold drops discriminative information; compare against `cascade__topk` and `cascade__l1` heads-up.
- [ ] 4.2 `openspec/changes/investigate-projection-bottleneck-cascade/proposal.md` — bucket B. Sweep `scale_boost` over {0.5, 1.0, 1.5, 2.0, auto}; measure pre/post-projection rank ratio; check whether the polygram basis is under-rank for the residual stream's effective dim.
- [ ] 4.3 `openspec/changes/richer-cascade-host-supervision-v2/proposal.md` — bucket C. Redesign the aux vocabulary to per-particle / per-flavor labels; commit to the dual-head + focal-loss recipe from econ-sae Phase 6.2 at full 110-label scale.

## 5. Documentation

- [ ] 5.1 Add a paragraph to `aux-supervise-cascade-host/proposal.md` (in-archive note) capturing the v1 outcome: gate 7.3 missed at Δ +0.0072; probe shows 3/5 labels already at AUC ≥ 0.98 in the LM-only host; 2/5 labels degenerate; diagnosis pivots to this change.
- [ ] 5.2 Update the README Phase C section (or the appropriate landing surface) to point at `runs/aux_probe/` for the diagnostic artefacts.

## 6. Validation

- [ ] 6.1 The script runs to completion in ≤ 10 minutes on Intel macOS.
- [ ] 6.2 Output JSON has all four AUC dicts (baseline_residual, baseline_projected, aux_residual, aux_projected) when invoked with all flags.
- [ ] 6.3 Bucket summary correctly identifies A / B / C / ambiguous against synthetic test inputs (unit test the bucketing logic in isolation; the live probe's bucket is the empirical finding).
- [ ] 6.4 At least one follow-up openspec proposal under §4 is filed with the live probe's per-feature table populating its "Why."
