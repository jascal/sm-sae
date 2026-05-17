# tasks — add-cascade-host-shim

## 1. Trajectory dataset

- [ ] 1.1 Add `smsae.sae.data.cascade_transitions(n_trajectories: int, seed: int)`
      yielding `(state_t, state_t+1)` pairs. Reuses `cascade()` from
      `smsae.sm.cascade`; one trajectory contributes ~5–30 transition
      pairs depending on parent particle mass.
- [ ] 1.2 Encode each `state_t` as int64 input_ids by listing particles
      `count` times in canonical (alphabetical) order. Truncate to
      `max_seq` (default 32); pad with a reserved `PAD` token.
- [ ] 1.3 Encode each `state_t+1` as a per-position categorical target
      over the 62-symbol vocab (61 particles + PAD). Positions
      corresponding to PAD inputs get `ignore_index=-100`.
- [ ] 1.4 Unit test: round-trip a known cascade trajectory through the
      encoder and confirm the input_ids reconstruct the multiset.

## 2. Tiny transformer architecture

- [ ] 2.1 Add `smsae.host.tiny_gpt2(n_embd, n_layer=2, n_head=None,
      vocab_size=62)` returning a HuggingFace `GPT2LMHeadModel` with
      the requested dimensions. Default `n_head = max(1, n_embd // 8)`
      with a divisor check so `n_embd % n_head == 0`.
- [ ] 2.2 Verify total parameter count lands in [10k, 200k] for the
      shapes sm-sae actually uses (`n_embd=16` → ~30k, `n_embd=61` →
      ~110k).
- [ ] 2.3 Smoke test: forward pass on a (4, 32) int64 batch produces a
      `(4, 32, 62)` logits tensor with no NaNs.

## 3. Training loop

- [ ] 3.1 Add `scripts/train_cascade_host.py` with CLI:
      `--n-embd INT [--n-layer INT] [--n-head INT] [--n-trajectories INT]
      [--epochs INT] [--batch-size INT] [--lr FLOAT] [--seed INT]
      [--out PATH]`.
- [ ] 3.2 Implement the training loop: AdamW + cosine LR schedule with
      10% warmup, gradient clipping at 1.0, log per-100-step. Default
      hyperparameters target convergence in ≤ 5 minutes on CPU for
      `n_embd=16`.
- [ ] 3.3 Save the trained model via `model.save_pretrained(out / "host")`
      and a `config.json` recording: `n_embd`, `n_layer`, `n_head`,
      `vocab_size`, `n_trajectories`, `n_train_steps`, `train_loss_final`,
      `seed`, `git_commit` (best-effort).
- [ ] 3.4 Smoke test: `python scripts/train_cascade_host.py --n-embd 16
      --n-trajectories 100 --epochs 1` completes in < 60s and writes
      both files.

## 4. forge_pipeline integration

- [ ] 4.1 Refactor `_build_synthetic_host(input_dim)` to:
      (a) check for `runs/cascade_host/<input_dim>/host.safetensors`,
      (b) if found, load via `GPT2LMHeadModel.from_pretrained(...)` and
      return that;
      (c) otherwise emit a `UserWarning` and return the random-init
      version.
- [ ] 4.2 Have the function also return a small dict describing the
      host (kind / n_embd / n_layer / n_head / train_loss_final if
      trained / `None` if random) so the pipeline can record it in
      `forge_results.json`.
- [ ] 4.3 Update `forge()` to thread the host-info dict into the result
      payload under the `host` key (replacing the current free-text
      string).
- [ ] 4.4 Integration test: after running `train_cascade_host.py --n-embd
      16`, `forge_pipeline.py embedded__topk` records `host.kind ==
      "trained"` in its `forge_results.json`.

## 5. Scoreboard rendering

- [ ] 5.1 In `scripts/visualize.py:_format_forge_pipeline_results`, add
      a "host" column between "encoding" and "dict features". Render
      "🎓 trained (loss=0.42)" for trained hosts and "🎲 random" for
      fallback hosts (italics).
- [ ] 5.2 Update the surrounding prose to flag the distinction: "rows
      with a 🎲 host are wiring sanity checks; rows with 🎓 are the ones
      whose forge_score actually reflects cascade-structure preservation".
- [ ] 5.3 Update the `_sae_forge_section` "What's still synthetic" block
      to be conditional: when at least one trained-host row exists,
      pivot the prose from "everything is still synthetic" to "trained
      hosts now available; the cascade-host shim has landed".

## 6. Documentation

- [ ] 6.1 README quickstart step: mention `python scripts/train_cascade_host.py
      --n-embd 16` before running `forge_pipeline.py`.
- [ ] 6.2 Add a `docs/forge_pipeline.md` (new) that walks through the
      cascade-host shim's design, why it's needed, and how to interpret
      the resulting faithfulness numbers.
- [ ] 6.3 Update the openspec `archive/` move once this change has
      landed and the acceptance criteria are met.

## 7. Acceptance gate

- [ ] 7.1 `python scripts/train_cascade_host.py --n-embd 16` runs to
      completion in < 5 minutes on CPU and writes the expected files.
- [ ] 7.2 `python scripts/forge_pipeline.py embedded__topk` picks up the
      trained host (verify via `forge_results.json.host.kind`).
- [ ] 7.3 The forge faithfulness with the trained host differs from the
      random-init baseline by ≥ 0.05 in absolute AUC terms on at least
      one of `embedded__topk` / `cascade__jumprelu`. (Direction not
      asserted — a *decrease* would also be informative; the point is
      the trained host is doing something.)
- [ ] 7.4 The scoreboard's "Forge pipeline runs" table has the new
      host column populated and the surrounding prose updated.
