# tasks — aux-supervise-cascade-host

## 1. Aux-label module

- [ ] 1.1 Create `smsae/host/aux_labels.py` defining
      `aux_label_names() -> list[str]` (returns the 5 label names in
      a stable order) and
      `compute_aux_labels(state: dict[str, int],
      initial_parent: str | None) -> np.ndarray` (returns a
      `(5,)` float32 array of 0/1 labels for the given cascade state
      and the originating particle the rollout started from).
- [ ] 1.2 Implement the five labels per `design.md` table:
      `total_charge_neutral`, `total_baryon_neutral`,
      `originated_from_top`, `state_has_higgs`, `state_has_top`.
      Use `smsae.sm.embeddings.build_sm()` for the per-particle
      charge/baryon values; cache the SM dict at module scope to
      avoid rebuild cost per call.
- [ ] 1.3 Unit test: every label is 0 or 1 for every test state;
      `originated_from_top` requires `initial_parent` to be one of
      `{"t_r","t_g","t_b"}` to fire; `state_has_higgs` requires
      `"H" in state`; conservation labels match
      hand-computed values on three fixture cascade rollouts.

## 2. `cascade_transitions` yields aux labels

- [ ] 2.1 Extend
      `smsae.sae.data.cascade_transitions(n_trajectories, seed,
      max_seq)` to optionally yield
      `(input_ids, target_ids, aux_labels)` triples when called
      with `with_aux=True` (default `False` so existing callers
      see the old shape). The label vector is computed from
      `state_t+1` and the trajectory's `initial_parent`; same row
      count as the existing pair stream.
- [ ] 2.2 Unit test: `with_aux=True` returns 3-tuples; aux vector
      shape is `(5,)`; row counts match `with_aux=False`.

## 3. `tiny_gpt2` optional aux head

- [ ] 3.1 Extend `smsae.host.tiny_gpt2.tiny_gpt2(n_embd, n_layer=2,
      n_head=None, vocab_size=62, n_positions=64, n_inner=None,
      aux_heads: int = 0)`. When `aux_heads > 0`, attach
      `model.aux_head = nn.Linear(n_embd, aux_heads)` after
      construction; the model is otherwise byte-identical to
      `aux_heads=0`.
- [ ] 3.2 Smoke test: `aux_heads=5` adds exactly
      `n_embd * 5 + 5` parameters (one Linear with bias);
      `aux_heads=0` leaves param count unchanged from the v1
      shape; forward-pass shape is unchanged regardless (aux head
      is invoked separately by the trainer).

## 4. Trainer integration

- [ ] 4.1 Add CLI flag `--aux-supervision {off, pooled,
      per_channel, dual}` to `scripts/train_cascade_host.py`,
      default `off`. Only `pooled` is implemented in v1; the other
      two raise `NotImplementedError` with a pointer at this
      proposal's follow-up changes.
- [ ] 4.2 Add CLI flag `--aux-lambda FLOAT`, default `1.0`. Ignored
      when `--aux-supervision off`.
- [ ] 4.3 When `--aux-supervision pooled`, the training loop:
      (a) builds the model with `aux_heads=len(aux_label_names())`;
      (b) consumes the 3-tuple stream from `cascade_transitions(..., with_aux=True)`;
      (c) computes pooled hidden state (`out.hidden_states[-1].mean(dim=1)`
      or equivalent via `output_hidden_states=True`);
      (d) loss = `CE(token_logits, target_ids)` + `λ * BCE_with_logits(aux_logits, aux_labels)`;
      (e) logs both losses separately every 100 steps;
      (f) saves the model via `save_pretrained` as before (aux head
      ships with the rest of the state dict).
- [ ] 4.4 Extend `runs/cascade_host/<n_embd>/config.json` with
      `aux_supervision`, `aux_labels` (the names list),
      `aux_lambda`, `aux_loss_final` fields. Missing-key tolerant
      when reading old configs.
- [ ] 4.5 Smoke test: `python scripts/train_cascade_host.py
      --n-embd 16 --n-trajectories 100 --epochs 1
      --aux-supervision pooled` finishes in < 60s, writes a
      config.json that includes the new fields, and the saved model
      has an `aux_head` submodule.

## 5. forge_pipeline integration

- [ ] 5.1 `_build_synthetic_host`: when loading a trained host,
      surface the new config fields on the returned `host_info`
      dict (`aux_supervision`, `aux_loss_final`). Missing keys
      default to `aux_supervision="off"`, `aux_loss_final=None`
      for back-compat with pre-aux-trained hosts.
- [ ] 5.2 `forge()`: thread the new fields into
      `forge_results.json.forge.host`. No other changes — sae-forge
      doesn't use the aux head; only the projected token-side
      weights matter for `run_synthetic`.
- [ ] 5.3 Integration test: train a host with
      `--aux-supervision pooled`; run `forge_pipeline.py
      embedded__topk`; confirm
      `forge_results.json.forge.host.aux_supervision == "pooled"`.

## 6. Scoreboard rendering

- [ ] 6.1 `_format_forge_pipeline_results`: when
      `host.aux_supervision != "off"`, render the host cell as
      "🎓+aux trained (loss=X.XXX, aux=Y.YYY)" instead of the
      current "🎓 trained (loss=X.XXX)". Plain 🎓 stays for hosts
      with `aux_supervision="off"` or missing.
- [ ] 6.2 The host-column aside gains one sentence: "🎓+aux rows
      were trained with an auxiliary supervision head; see
      `openspec/changes/archive/aux-supervise-cascade-host/` for
      what's being supervised."

## 7. Acceptance gate

- [ ] 7.1 `python scripts/train_cascade_host.py --n-embd 61
      --aux-supervision pooled` runs to completion in
      < 5 minutes on CPU.
- [ ] 7.2 `python scripts/forge_pipeline.py cascade__jumprelu`
      with the aux-trained host present records
      `host.aux_supervision == "pooled"` and
      `host.aux_loss_final` is a positive float.
- [ ] 7.3 The trained-vs-random faithfulness delta on
      `cascade__jumprelu` is ≥ 0.05 (the gate 7.3 metric from
      `add-cascade-host-shim`). Closes that gate retroactively if
      met.
- [ ] 7.4 If 7.3 is missed: file the appropriate follow-up
      (`per-channel-cascade-host-supervision`,
      `dual-head-cascade-host-supervision`) with the measured
      delta in the proposal's "Why" section.

## 8. Documentation

- [ ] 8.1 README quickstart: add a `--aux-supervision pooled` mention
      to the cascade-host training step with a one-line note on
      what it's for.
- [ ] 8.2 Refresh the
      `_sae_forge_section` cascade-host paragraph to note that
      `🎓+aux` runs are an option and link to this change's
      archive after it lands.

## 9. Diagnostic (recommended, not gating)

- [ ] 9.1 Add `scripts/probe_host_aux_recoverability.py`: trains a
      tiny linear classifier from a host's pooled hidden state to
      each of the 5 aux labels, with the host's weights frozen.
      Reports per-label AUC. Run on both a non-aux-trained host
      (PR #10 baseline) and a `--aux-supervision pooled` host.
      Headline question: were the aux labels already recoverable
      without supervision? If so, that explains a flat gate 7.3
      outcome; if not, aux supervision is doing genuine work.
