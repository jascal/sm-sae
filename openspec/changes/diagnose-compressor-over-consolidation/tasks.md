# tasks — diagnose-compressor-over-consolidation

## 1. Reproduce the collapse

- [ ] 1.1 Capture the exact `compress()` inputs (the saved
      `validation_report.json` + `sae.safetensors`) for
      `cascade__jumprelu --select-by firing_rate` so the diagnostic
      runs are reproducible without re-invoking stages 1–5.
- [ ] 1.2 Confirm the baseline behaviour: same inputs, default config →
      3 clusters. Pin this as the reference for sweep cells.

## 2. Test the two hypotheses

- [ ] 2.1 **Hypothesis A — Compressor params.** Run the default
      Compressor with each of the documented
      `(rep_selection, merge_mode, score_field)` triples and a couple
      of `confirmer` settings. Record `n_clusters` per cell.
- [ ] 2.2 **Hypothesis B — synthesized vreport is degenerate.**
      Re-run with hand-perturbed `kl_ablate_*` values (e.g. sample from
      `N(0.1, 0.05)` instead of `0.0`) and see whether the cluster
      count moves. If yes → fix the vreport synthesizer; if no →
      hypothesis A wins.
- [ ] 2.3 Write up the evidence in `design.md` "Investigation" section
      with a small markdown table per hypothesis.

## 3. forge_pipeline.compress() overridable

- [ ] 3.1 Change `compress()` signature to accept a `config:
      CompressionConfig | None = None`. When `None`, build a default
      `CompressionConfig(strategy=strategy)` to preserve current
      behaviour byte-for-byte.
- [ ] 3.2 Add `--compressor-config <json>` to the CLI. Accepts a JSON
      string mapping any subset of `CompressionConfig` fields to
      values; validates against the dataclass.
- [ ] 3.3 Record the resolved config (as a dict) under
      `compress.config` in `forge_results.json`.

## 4. compressor_sweep.py

- [ ] 4.1 New script `scripts/compressor_sweep.py` that takes a
      `--run-id` and a fixed selector, runs stages 1–5 once, then
      loops over a small grid of Compressor configs. Writes one
      `forge_results.json` per cell under
      `runs/sae_forge/<run_id>__sweep/<config_fp>/`.
- [ ] 4.2 Print a final summary table sorted by `n_clusters` desc
      then `faithfulness` desc, so the visually-best cells are at the
      top.
- [ ] 4.3 Resumable: skip cells whose `forge_results.json` already
      exists.

## 5. Scoreboard caveat

- [ ] 5.1 In `_format_forge_pipeline_results`, add a one-paragraph
      caveat under the table flagging that 1–3 clusters on
      `cascade__jumprelu` reflects a known Compressor-tuning issue,
      with a link to this change's design.md.
- [ ] 5.2 Once a winning config is found, recommend it in
      `_format_recommended_defaults` (Compressor row).

## 6. Documentation

- [ ] 6.1 Update `scripts/forge_pipeline.py` docstring to mention
      `--compressor-config`.
- [ ] 6.2 Document the diagnostic workflow (run sweep → pick config
      → re-run pipeline) in `docs/forge_pipeline.md` if/when that
      file lands from [[add-cascade-host-shim]] (task 6.2 there).

## 7. Acceptance gate

- [ ] 7.1 `design.md` "Investigation" section names the dominant root
      cause with evidence.
- [ ] 7.2 At least one non-default Compressor config is documented
      that produces ≥ 4 clusters on `cascade__jumprelu` with the
      `firing_rate` selector, OR the investigation conclusively shows
      that ≥ 4 is unreachable on this fixture and the
      principled-feature-selection acceptance gate (8.4) is revised
      accordingly.
- [ ] 7.3 `forge_results.json` records the resolved Compressor config
      so different runs are distinguishable.
- [ ] 7.4 Scoreboard caveat is visible until a recommended config
      lands.
