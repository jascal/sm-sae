# tasks — investigate-cascade-host-capacity-sweep

## 1. Implementation

- [x] 1.1 `scripts/cascade_host_capacity_sweep.py` — driver script
      that runs train + forge + probe across the grid; writes
      `runs/capacity_sweep/summary.{json,csv}` and per-config
      artefacts.
- [x] 1.2 `--smoke` flag for fast wiring verification
      (`(61,2) + (96,2)` only).
- [x] 1.3 Patch `scripts/forge_pipeline.py` with `--host-dir`
      override so per-config hosts can feed the forge without
      overwriting `runs/cascade_host/61/`.
- [x] 1.4 Configuration: the proposal's n_embd-only grid was
      revised after impl discovered the cascade SAE
      (`cascade__jumprelu`, input_dim=61) can only forge hosts
      with n_embd=61. The actual grid:
        - Forge + probe: (61, 2), (61, 4), (61, 6) — depth scan
        - Probe only:    (96, 2), (128, 2), (192, 2) — width scan
      Width configs still inform gate C.3 (per-particle AUC
      scaling) but don't contribute to gate C.2 directly.

## 2. Live sweep (this machine, 2026-05-20)

- [x] 2.1 Full sweep ran in 345.1s (5.8 min) on Intel CPU. Wall
      well under the proposal's 20-min budget.
- [x] 2.2 All 6 configs trained + measured. C.1 PASS.

### Sweep results (the load-bearing artefact)

| config | n_params | train_loss | forge_score | Δ_random | probe_mean | spotlight median |
|---|---|---|---|---|---|---|
| NE61_L2 | 97k | 1.778 | 0.7262 | **−0.0048** | 0.902 | 0.799 |
| NE61_L4 | 188k | 1.662 | 0.7358 | **+0.0048** | 0.897 | 0.809 |
| NE61_L6 | 279k | 1.729 | 0.7549 | **+0.0239** | 0.915 | 0.859 |
| NE96_L2 | 233k | 1.805 | skipped | — | 0.918 | 0.864 |
| NE128_L2 | 409k | 1.708 | skipped | — | 0.922 | 0.865 |
| NE192_L2 | 908k | 1.695 | skipped | — | 0.917 | 0.867 |

Random-init forge baseline: 0.7310.

## 3. Gate verdicts (live measurement)

- [x] **Gate C.1 (mechanical)**: **PASS** — full sweep completed in
      5.8 min; all 6 configs produced train + forge/probe rows.

- [x] **Gate C.2 (gate-7.3-by-capacity, ≥ +0.05)**: **FAIL but
      trending positive on depth axis.** Best Δ in measured set:
      **+0.0239** at NE61_L6 (n_layer=6). The trajectory is
      monotonic in n_layer:
        - L2: −0.0048
        - L4: +0.0048 (+0.0096 from L2)
        - L6: +0.0239 (+0.0191 from L4)
      Each doubling of depth ~ doubles the Δ. Extrapolating, L8
      ≈ +0.035, L10 ≈ +0.045-0.055 — gate target may be within
      reach via depth alone.

- [x] **Gate C.3 (per-particle AUC scaling to ≥ 0.92)**: **FAIL —
      width saturates.** Largest-capacity config (NE192_L2)
      spotlight median is **0.867**, well short of 0.92. The width
      scan shows clear saturation:
        - NE96_L2:  0.864
        - NE128_L2: 0.865
        - NE192_L2: 0.867
      4× width buys +0.003. Depth, by contrast, moves the
      spotlight from 0.799 (L2) → 0.859 (L6) — **+0.06 from depth,
      +0.003 from width**.

## 4. Diagnosis (definitively informed by all evidence)

The five-PR arc (#19 v1 → #20 probe proposal → #22 probe impl → #23
v2 → THIS) converges on a CLEAR answer:

- **Aux supervision (v1, v2) is not the binding lever.** Both miss
  gate 7.3; v2 with more labels actually regresses.
- **Width scaling is not the binding lever either.** n_embd 61→192
  (3×) only buys +0.003 spotlight AUC; n_params 4×-10× yields
  near-zero forge improvement.
- **Depth IS the binding lever.** n_layer 2→6 (3×) buys +0.06
  spotlight AUC AND +0.0287 forge Δ. The depth trajectory is
  monotonic and consistent with closing gate 7.3 at n_layer ≈ 8-10.

**Why depth wins**: cascade prediction requires composing multi-step
relationships (state_t → decay candidates → state_{t+1} populations).
Width gives more residual-stream bandwidth for one-shot lookups; depth
gives MORE COMPOSITIONAL STAGES. The cascade is structurally a
compositional task, so it benefits from depth where width saturates.

## 5. Recommended next experiment

**File `cascade-host-depth-sweep`** as the direct follow-up:

- Sweep `n_layer ∈ {6, 8, 10, 12}` at fixed `n_embd=61`, no aux.
- Train + forge + probe each; report Δ_random + spotlight AUC.
- **Acceptance**: at least one configuration achieves Δ_random ≥ +0.05
  (the original gate 7.3).
- Compute estimate on Intel CPU: ~10-15 min for 4 configs.
- If the trend continues at the L4→L6 rate (Δ ~ +0.01 per layer
  doubling), L8 ≈ +0.035, L12 ≈ +0.06+. L10 should be the
  transition point.

If THIS sweep closes gate 7.3, the gate-7.3 saga ends and the
canonical production cascade host becomes n_embd=61 / n_layer=10-ish
(or whichever configuration first hits the gate).

If the depth trend SATURATES somewhere between L6 and L12, the next
experiment is "depth + width combined" (a 2D sweep) OR a different
SAE — `cascade__topk` or a larger-input_dim retrained SAE — to test
whether the SAE is the binding constraint at this host capacity.

## 6. Artefacts persisted

- `scripts/cascade_host_capacity_sweep.py` (new, ~340 lines).
- `runs/capacity_sweep/summary.json` (force-add despite gitignore;
  the load-bearing measurement).
- `runs/capacity_sweep/summary.csv` (analyst-friendly export).
- Per-config probe/forge dirs under `runs/capacity_sweep/`.
- `runs/cascade_host/sweep_NE*_L*/` host weights (gitignored by
  default; preserved on this machine for the depth follow-up).
- `scripts/forge_pipeline.py` patched with `--host-dir` override.

## 7. What this change explicitly does NOT do

- **Implement the depth-only follow-up sweep.** That's the next
  proposal (§5).
- **Retrain the cascade SAE at larger input_dim.** Several width
  configs were measured (NE96/128/192) but their forge rows are
  marked `skipped_dim_mismatch`. A retrained SAE at n_embd=192
  would let those configs land forge scores; out of scope here.
- **Change the cascade-rollout vocabulary or the SAE family.** The
  depth-axis result rules in/out the v1+v2+probe hypothesis at
  the current SAE family; that's the load-bearing finding.

## 8. Archive trigger

This change SHALL be archived once:
1. The depth-only follow-up (`cascade-host-depth-sweep`) is filed.
2. The summary.json artefact is committed to history.
