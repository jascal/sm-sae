# tasks — add-cascade-host-shim

## 1. Trajectory dataset

- [x] 1.1 Add `smsae.sae.data.cascade_transitions(n_trajectories: int, seed: int)`
      yielding `(state_t, state_t+1)` pairs. Reuses `cascade()` from
      `smsae.sm.cascade`; one trajectory contributes ~5–30 transition
      pairs depending on parent particle mass.
- [x] 1.2 Encode each `state_t` as int64 input_ids by listing particles
      `count` times in canonical (alphabetical) order. Truncate to
      `max_seq` (default 32); pad with a reserved `PAD` token.
- [x] 1.3 Encode each `state_t+1` as a per-position categorical target
      over the 62-symbol vocab (61 particles + PAD). Positions
      corresponding to PAD inputs get `ignore_index=-100`.
- [x] 1.4 Unit test: round-trip a known cascade trajectory through the
      encoder and confirm the input_ids reconstruct the multiset.

## 2. Tiny transformer architecture

- [x] 2.1 Add `smsae.host.tiny_gpt2(n_embd, n_layer=2, n_head=None,
      vocab_size=62)` returning a HuggingFace `GPT2LMHeadModel` with
      the requested dimensions. Default `n_head = max(1, n_embd // 8)`
      with a divisor check so `n_embd % n_head == 0`.
- [x] 2.2 Verify total parameter count lands in a non-degenerate range
      for the shapes sm-sae uses. **Measured**: `n_embd=16` → 8.1k
      params, `n_embd=61` → 97k. The original `~30k / ~110k` forecast
      didn't account for the tied LM head + small (62-token) vocab.
      Test thresholds adjusted to `[5k, 200k]` / `[50k, 200k]` to
      catch genuinely-degenerate models rather than the inflated
      forecast.
- [x] 2.3 Smoke test: forward pass on a (4, 32) int64 batch produces a
      `(4, 32, 62)` logits tensor with no NaNs.

## 3. Training loop

- [x] 3.1 Add `scripts/train_cascade_host.py` with CLI:
      `--n-embd INT [--n-layer INT] [--n-head INT] [--n-trajectories INT]
      [--epochs INT] [--batch-size INT] [--lr FLOAT] [--seed INT]
      [--out PATH]`.
- [x] 3.2 Implement the training loop: AdamW + cosine LR schedule with
      10% warmup, gradient clipping at 1.0, log per-100-step. Default
      hyperparameters converge well under 5 minutes on CPU for
      `n_embd=16` (measured 14s wall for 5 epochs / 2k trajectories).
- [x] 3.3 Save the trained model via `model.save_pretrained(out / "host")`
      and a `config.json` recording: `n_embd`, `n_layer`, `n_head`,
      `vocab_size`, `n_trajectories`, `n_train_steps`, `train_loss_final`,
      `seed`, `git_commit` (best-effort).
- [x] 3.4 Smoke test: `python scripts/train_cascade_host.py --n-embd 16
      --n-trajectories 100 --epochs 1` completes in 8.5s and writes
      both files.

## 4. forge_pipeline integration

- [x] 4.1 Refactor `_build_synthetic_host(input_dim)` to:
      (a) check for `runs/cascade_host/<input_dim>/host/` (HF
      save_pretrained directory),
      (b) if found, load via `GPT2LMHeadModel.from_pretrained(...)` and
      return that;
      (c) otherwise emit a `UserWarning` (with the exact training
      command to fix) and return the random-init version.
- [x] 4.2 Have the function also return a small dict describing the
      host (`kind`, `path`, `n_embd`, `n_layer`, `n_head`, `n_params`,
      `train_loss_final`, `n_train_trajectories`, `n_train_steps`,
      `seed`) so the pipeline can record it in `forge_results.json`.
- [x] 4.3 Update `forge()` to thread the host-info dict into the result
      payload under the `host` key (replacing the free-text string).
- [x] 4.4 Integration test: after running `train_cascade_host.py --n-embd
      61`, `forge_pipeline.py cascade__jumprelu` records
      `host.kind == "trained"` with `train_loss_final` and
      `n_train_trajectories` populated.

## 5. Scoreboard rendering

- [x] 5.1 In `scripts/visualize.py:_format_forge_pipeline_results`, add
      a "host" column between "selection" and "dict features". Render
      "🎓 trained (loss=X.XXX)" for trained hosts and "🎲 random" for
      fallback hosts (italics).
- [x] 5.2 Update the surrounding prose to flag the distinction: rows
      with 🎲 are wiring sanity checks; only 🎓 rows belong in Axis-C
      comparisons.
- [ ] 5.3 Conditional `_sae_forge_section` "What's still synthetic"
      pivot. Deferred — minor copy edit, schedule for after the gate
      gap is resolved.

## 6. Documentation

- [ ] 6.1 README quickstart step: mention `python scripts/train_cascade_host.py
      --n-embd 16` before running `forge_pipeline.py`. Deferred.
- [ ] 6.2 Add a `docs/forge_pipeline.md` (new) that walks through the
      cascade-host shim's design, why it's needed, and how to interpret
      the resulting faithfulness numbers. Deferred.
- [ ] 6.3 Move openspec change to `archive/` once gate 7.3 is closed.

## 7. Acceptance gate

- [x] 7.1 `python scripts/train_cascade_host.py --n-embd 16` runs to
      completion in 14s on CPU and writes the expected files. (Gate
      was 5 min; we're 20× under.)
- [x] 7.2 `python scripts/forge_pipeline.py cascade__jumprelu` picks up
      the trained host (`forge_results.json.host.kind == "trained"`,
      `train_loss_final` and `n_train_trajectories` both populated).
- [ ] 7.3 The forge faithfulness with the trained host differs from the
      random-init baseline by ≥ 0.05 in absolute AUC terms on at least
      one of `embedded__topk` / `cascade__jumprelu`.
      **Missed:** measured deltas:

      | run_id | random | trained (5 ep, 2k traj) | trained (20 ep, 4k traj) | Δ vs random |
      |---|---|---|---|---|
      | `embedded__topk` (n_embd=16)     | 0.8834 | 0.8835 | — | +0.0001 |
      | `cascade__jumprelu` (n_embd=61)  | 0.7448 | 0.7367 | 0.7499 | +0.0051 |

      Magnitude is ~10× below the gate even with 4× more training. The
      most likely cause is downstream: the SubspaceProjector compresses
      the 96k-param trained host into a 3-feature basis (because the
      Compressor only produces 3 clusters from 20 confirmed pairs —
      the same over-consolidation tracked in
      [[diagnose-compressor-over-consolidation]]). Almost all of the
      host's learned cascade signal is discarded at the projection
      stage, so trained-vs-random becomes indistinguishable. Gate 7.3
      will be re-evaluated after the Compressor sweep lands and
      cluster counts grow.
- [x] 7.4 The scoreboard's "Forge pipeline runs" table has the new
      host column populated and the surrounding prose updated.

## 8. Follow-up

Gate 7.3 is blocked on the same Compressor over-consolidation that
PR #1's gate 8.4 hit. Once [[diagnose-compressor-over-consolidation]]
lands a tuned config and cluster counts rise (e.g. 3 → 8 on
`cascade__jumprelu`), the trained-host faithfulness delta should
become measurable. Re-run this gate then, before archiving.
