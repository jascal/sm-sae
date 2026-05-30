# scoreboard-forge-pipeline-runs Specification (delta)

## MODIFIED Requirements

### Requirement: scoreboard host column renders aux-supervised hosts distinctly

`_format_forge_pipeline_results` SHALL inspect each row's
`forge.host.aux_supervision` field and render the host cell as
follows:

- When `aux_supervision == "off"` or the field is absent: render
  the existing `🎓 trained (loss=X.XXX)` cell unchanged.
- When `aux_supervision != "off"`: render the cell as `🎓+aux
  trained (loss=X.XXX, aux=Y.YYY)`, where `Y.YYY` is the formatted
  `forge.host.aux_loss_final` value.
- When the host is not trained at all (random init): render the
  pre-change non-trained cell unchanged.

#### Scenario: aux-supervised host gets the `🎓+aux` marker

- **GIVEN** a row with `forge.host.aux_supervision == "pooled"`,
  `forge.host.loss_final == 0.123`, and `forge.host.aux_loss_final
  == 0.456`
- **WHEN** the scoreboard renders the row's host cell
- **THEN** the cell text is `🎓+aux trained (loss=0.123, aux=0.456)`

#### Scenario: legacy host gets the plain `🎓` marker

- **GIVEN** a row whose `forge.host` payload lacks
  `aux_supervision` (a host trained before this change)
- **WHEN** the scoreboard renders the row's host cell
- **THEN** the cell text is `🎓 trained (loss=X.XXX)` — identical
  to the pre-change rendering

#### Scenario: `aux_supervision == "off"` also gets the plain marker

- **GIVEN** a row with `forge.host.aux_supervision == "off"`
  (explicit, not absent)
- **WHEN** the scoreboard renders the row's host cell
- **THEN** the cell text is `🎓 trained (loss=X.XXX)` — the
  `+aux` suffix is NOT appended

### Requirement: scoreboard aside explains the `🎓+aux` marker

The scoreboard's host-column aside SHALL gain one sentence
explaining that rows marked `🎓+aux` were trained with an auxiliary
supervision head, linking to the archived
`aux-supervise-cascade-host` change for the specifics. The original
sentence about plain `🎓` remains.

#### Scenario: aside text covers both markers

- **WHEN** the scoreboard's host-column aside is rendered
- **THEN** the aside contains a sentence about plain `🎓` (unchanged
  from pre-change) AND a sentence about `🎓+aux` that names the
  source change directory in the openspec archive
