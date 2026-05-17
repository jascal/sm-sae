# design — auto-tune-subspace-projector-scale-boost

## Context

`SubspaceProjector` scales the projected weight matrices by a constant
factor `scale_boost` when projecting host weights into the SAE feature
basis. For under-complete bases (more host dimensions than features),
the default `scale_boost=1.0` preserves weight magnitudes faithfully.
For over-complete bases (more features than host dimensions, which is
sm-sae's regime), `1.0` amplifies the projected weights well past what
the host produced, producing residual activations that are numerically
larger than expected.

sae-forge ships `scale_boost="auto"` exactly for this case — it computes
a sensible factor from the basis geometry. The auto value typically
lands in `[0.1, 0.5]` for over-complete bases. The published warning is
sae-forge correctly diagnosing that we're using the wrong default.

This change is mechanical: pass `"auto"` instead of `1.0`. The
substantive work is documenting the change, capturing the resolved
value, and confirming the faithfulness numbers don't regress.

## Goals

- **No more spurious warning** at the top of every forge pipeline run.
- **Faithfulness numbers remain at-or-better-than baseline** — auto is
  designed for this regime, so we should not see a regression.
- **Transparency**: the actual scale_boost value used is recorded with
  every run so a reader can reproduce or audit.

## Non-Goals

- **Re-implementing scale_boost.** sae-forge's auto mode is the
  upstream's recommendation; we just consume it.
- **Per-encoding scale_boost tuning.** If different encodings benefit
  from different boost factors, that's a sae-forge concern, not an
  sm-sae one.
- **Comparing auto-resolved values across runs to pick a "best"
  constant**. Defeats the point of auto.

## Decisions

### 1. Default = `"auto"`, not a constant

`auto` is sae-forge's recommended default for over-complete bases.
Hardcoding e.g. `0.25` based on the warning's GPT-2-1024-features
example would over-fit our defaults to one specific shape. `auto`
handles whatever shape the user's basis happens to have.

### 2. CLI override exists but isn't documented prominently

`--scale-boost` accepts any float for users who want to experiment.
Default stays `auto`. Not promoted in the README — the auto path is
intended to Just Work, and exposing the knob too prominently invites
unproductive tuning.

### 3. Resolved value recorded under `projector.scale_boost_resolved`

Naming follows what sae-forge surfaces (confirm exact attribute in
task 1.2). The result payload becomes:

```json
"projector": {
  "scale_boost_arg":      "auto",
  "scale_boost_resolved": 0.31
}
```

If the user passed a constant, `scale_boost_arg` is the constant and
`scale_boost_resolved` equals it.

### 4. Why this is its own change and not bundled with another

The change is tiny but touches a published behaviour (the forge
faithfulness numbers in the report). Bundling it with the larger
cascade-host-shim or per-encoding-A-B changes would conflate
"projection quality fix" with "new measurement infrastructure" — they
shouldn't be reasoned about together. Separate landings make it easy
to bisect if a forge faithfulness regression appears.

## Risks

- **`scale_boost="auto"` might land below `1.0` and make the random-
  init host's faithfulness numbers *worse*** (less amplification →
  smaller residuals → less label coincidence). That's not actually
  bad — it's a more honest signal — but the report's narrative might
  need a slight update to acknowledge the move.
- **The resolved attribute name might not match my guess**. Task 1.2
  confirms it before plumbing.
- **sae-forge's auto behaviour could change across versions**. The
  `[forge]` / `[forge-intel]` extras pin different transformers
  versions; if auto's computation depends on something that ships with
  sae-forge, we should pin the sae-forge git ref too. Task 1.3
  captures this.
