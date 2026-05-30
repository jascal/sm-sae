# tasks — reframe-gate-7.3-absolute-forge-score

## 1. Gate definition (code)

- [x] 1.1 Add `GATE_7_3_ABS_FORGE_TARGET = 0.76` to
      `scripts/cascade_host_capacity_sweep.py` with the rationale comment.
- [x] 1.2 Reframe the `C.2` / `B.2` gate evaluation to absolute `forge_score`;
      demote Δ-vs-random to `delta_vs_random_diagnostic`; print the cell
      forge_score + a note that the canonical gate is the scoreboard's
      strongest-family value.
- [x] 1.3 Update `tests/test_budget_sweep_config.py` to the reframed gate
      (`cell_meets_target`, `gate_7_3_target`, diagnostic retained).

## 2. Docs

- [x] 2.1 `README.md` — gate-7.3 quickstart wording → absolute forge_score.
- [x] 2.2 `scripts/train_cascade_host.py` — gate-7.3 docstring wording.
- [x] 2.3 `openspec/README.md` — lineage note: closed, gate reframed + MET.

## 3. Decision close-out

- [x] 3.1 Resolve `cascade-host-nonautoregressive` (option B taken); archive it
      with a resolution note; keep option A as an optional future experiment.

## 4. Verify

- [x] 4.1 Full `pytest` green.
