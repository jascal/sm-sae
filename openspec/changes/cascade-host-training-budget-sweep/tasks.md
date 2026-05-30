# tasks — cascade-host-training-budget-sweep

## 1. Driver: per-config training budget

- [x] 1.1 Add `BUDGET_GRID` (4 dicts at fixed `n_embd=61, n_layer=6`,
      varying `n_trajectories`/`epochs`).
- [x] 1.2 Add `_normalize_configs()` — turn 2-tuples (capacity/depth)
      and dicts (budget) into a uniform list of config dicts carrying
      `n_trajectories`/`epochs`/`lr`/`dir_name`. Capacity/depth inherit
      the global defaults; their `dir_name` stays `sweep_NE{ne}_L{nl}`
      (byte-identical host dirs, so re-runs reuse them).
- [x] 1.3 Thread `lr` through `_train_one()`.
- [x] 1.4 Rewrite the `run_sweep()` loop to read per-config budget +
      `dir_name`; add `n_trajectories`/`epochs`/`lr`/`n_train_steps`
      to each summary row.

## 2. Driver: CLI + gate reporting

- [x] 2.1 Add `budget` to `--grid` choices; default `--out`
      `runs/budget_sweep` when selected; add `--lr` flag.
- [x] 2.2 Make `--smoke` grid-aware (budget/depth → first 2 configs of
      the selected grid; capacity keeps its documented `(61,2)+(96,2)`).
- [x] 2.3 Add the `budget_trend` block to `gate_summary` (color:r +
      spotlight-median AUC vs `n_train_steps`; gradient-step rows vs the
      corpus-scaling control; gate-7.3 closure).

## 3. Run

- [x] 3.1 `--smoke` budget run — verify wiring end to end.
- [x] 3.2 Full 4-config budget sweep → `runs/budget_sweep/summary.{json,csv}`.
- [x] 3.3 Record gate B.1/B.2/B.3 verdicts (see proposal `## Result`):
      B.1 PASS, B.2 FAIL (best Δ +0.0239 at the lowest budget), B.3
      REFUTED (color:r 0.877 → 0.814 → 0.779, monotonically *down*).

## 4. Tests + close-out

- [x] 4.1 Unit test: `_normalize_configs` round-trips tuples and dicts;
      budget dirs disambiguate; capacity/depth dirs unchanged.
      (`tests/test_budget_sweep_config.py`)
- [x] 4.2 Full `pytest` green.
- [x] 4.3 File the follow-up per the acceptance-summary branch the result
      landed on (B.2 miss + color:r *dropping*): filed
      `cascade-host-nonautoregressive`. Lineage-wide archival
      (this change + the 7 sibling landed changes) is deferred to a
      dedicated `archive-gate-7.3-lineage` cleanup so the whole arc moves
      together.
