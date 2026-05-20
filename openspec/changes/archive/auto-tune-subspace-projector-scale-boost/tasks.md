# tasks — auto-tune-subspace-projector-scale-boost

## 1. Confirm the sae-forge API

- [x] 1.1 Verify `SubspaceProjector(basis, scale_boost="auto")` is
      accepted by the installed sae-forge version (v0.4.0+).
- [x] 1.2 Find the attribute that exposes the resolved float value
      post-construction (likely `projector.scale_boost` after auto
      resolution, or `projector.scale_boost_resolved`). Record exact
      attribute name in `design.md`.
- [x] 1.3 If the auto behaviour differs across the sae-forge versions
      we pin (`[forge]` vs `[forge-intel]`), document the divergence
      in design.md and pick a version constraint that gives consistent
      behaviour.

## 2. forge_pipeline change

- [x] 2.1 In `scripts/forge_pipeline.py:forge`, replace
      `SubspaceProjector(basis, scale_boost=1.0)` with
      `SubspaceProjector(basis, scale_boost=scale_boost)` where
      `scale_boost` comes from the `forge()` signature, default
      `"auto"`.
- [x] 2.2 After construction, read the resolved float value and store
      it for the result payload.
- [x] 2.3 In `forge_results.json`, add a `projector` block:
      `{"scale_boost_arg": "auto", "scale_boost_resolved": 0.31, ...}`.

## 3. CLI

- [x] 3.1 Add `--scale-boost` arg (accepts `"auto"` or any float;
      validate as `auto | positive_float`; default `auto`).
- [x] 3.2 Pass it through `main() → forge(..., scale_boost=...)`.
- [x] 3.3 Print resolved value in the stage-7 console output:
      `[7] sae-forge ForgePipeline.run_synthetic (scale_boost=0.31)`.

## 4. Verify

- [x] 4.1 Run `python scripts/forge_pipeline.py embedded__topk` and
      confirm no `SubspaceProjector` warning is emitted to stderr.
- [x] 4.2 Confirm the resolved scale_boost is recorded in the result
      payload.
- [x] 4.3 Compare forge faithfulness number with `--scale-boost auto`
      vs the previous `1.0` baseline. Document the delta in the
      design.md ("results" section).

## 5. Scoreboard update

- [x] 5.1 In `scripts/visualize.py:_format_recommended_defaults`, flip
      the `SubspaceProjector(scale_boost=...)` row's evidence tag from
      `provisional` to `measured` and update the prose to cite the
      sm-sae sweep + no-warning evidence.
- [x] 5.2 If a single `auto` value tends to win across runs, note it
      in the "sm-sae recommendation" column; otherwise note `"auto"`
      itself as the recommendation.

## 6. Archive

- [x] 6.1 Move this change directory to `openspec/changes/archive/`
      once landed.

## 7. Acceptance gate

- [x] 7.1 Forge pipeline output is free of the over-complete-basis
      warning.
- [x] 7.2 `forge_results.json` records the resolved scale_boost.
- [x] 7.3 Forge faithfulness with `auto` is no worse than the `1.0`
      baseline on either of `embedded__topk` / `cascade__jumprelu`
      (absolute AUC; equality counts as no-worse).
- [x] 7.4 The recommended-defaults table cites measured evidence.
