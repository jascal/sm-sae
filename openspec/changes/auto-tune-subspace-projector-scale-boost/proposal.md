# auto-tune-subspace-projector-scale-boost

## Why

Every `scripts/forge_pipeline.py` run prints a `UserWarning` from
sae-forge's `SubspaceProjector`:

```
UserWarning: SubspaceProjector: over-complete basis detected
(n_features=61 > d_model=16) with scale_boost=1.0. The default is often
too large in this regime — empirically GPT-2 (d_model=768) with 1024
features needed scale_boost≈0.25 to train stably. Consider
scale_boost='auto' or a hand-picked value < 1.0; tune from there if
needed.
```

This is sae-forge correctly diagnosing that our `SubspaceProjector(basis,
scale_boost=1.0)` is wrong for over-complete bases — which sm-sae
**always** has, because polygram caps feature counts low (8 / 16 / 32)
and our SAEs are sized 32 / 64 / 128.

The fix is a single keyword change to pass `scale_boost="auto"` instead
of `1.0`, plus recording the chosen value in the result payload for
transparency. The warning goes away, the projection is more numerically
sensible, and forge faithfulness numbers should change in a
characterisable direction.

P2 because the wiring is already running end-to-end and the warning
is annoying-not-broken. Worth fixing soon because the longer it sits,
the longer the published faithfulness numbers reflect a known-suboptimal
default.

## What Changes

- **`scripts/forge_pipeline.py:forge` passes** `scale_boost="auto"` to
  `SubspaceProjector` instead of the literal `1.0`.
- **Record the actual scale_boost** value chosen by `auto` in
  `forge_results.json` under a new `projector.scale_boost` field
  (sae-forge's `SubspaceProjector` exposes the resolved value as
  `projector.scale_boost_resolved` or equivalent — confirm in the API
  and pull it).
- **Add an opt-out CLI flag** `--scale-boost` (default `auto`, accepts
  any float) so a user can hand-tune for experimentation without
  editing code.
- **Update the recommended-defaults table**: the
  `SubspaceProjector(scale_boost=...)` row gets a `measured` evidence
  badge, citing the no-warning + faithfulness-not-worse criteria.

## Capabilities

### Modified Capabilities

- `forge-pipeline`: `scale_boost` default goes `1.0 → "auto"`. CLI
  exposes the override. Result payload records the resolved value.
- `scoreboard-recommended-defaults`: the `SubspaceProjector.scale_boost`
  row (currently {provisional}) gets re-tagged with measured evidence.
