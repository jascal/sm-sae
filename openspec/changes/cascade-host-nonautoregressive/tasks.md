# tasks — cascade-host-nonautoregressive

> Proposal-only close-out branch of `cascade-host-training-budget-sweep`.
> Implementation is deferred until the maintainer picks a path (see the
> proposal's "Decision teed up for the maintainer").

## 1. Decision (do first)

- [ ] 1.1 Maintainer picks: (A) build the non-AR host, or (B) re-frame
      gate 7.3 to absolute `forge_score` per PR #27 (lower effort).
- [ ] 1.2 If (B): update README gate-7.3 definition + the openspec
      gate-7.3 lineage close-out; the saga ends here.

## 2. If (A) — non-AR host (only if 1.1 chooses it)

- [ ] 2.1 Add a non-AR host (DeepSets/mean-pool MLP or non-causal
      encoder) behind the existing forge host seam (residual width 61,
      saved under `<host-dir>/host/`).
- [ ] 2.2 Masked/denoising particle objective (predict held-out
      `state_t` particles) replacing next-token CE.
- [ ] 2.3 Add `--host-kind {ar,nonar}` to `cascade_host_capacity_sweep.py`
      (default `ar` preserves all existing grids).
- [ ] 2.4 Run; report forge Δ-vs-random + `color:r`/spotlight AUC vs the
      AR baseline (gates N.1/N.2/N.3).

## 3. Close-out

- [ ] 3.1 Record the verdict; archive the gate-7.3 lineage together.
