# tasks — sae-forge-world-model-adapter

> Tasks are split across two repos. **Upstream tasks** land in
> [jascal/sae-forge](https://github.com/jascal/sae-forge) and ship in
> a `saeforge` release. **Downstream tasks** land in sm-sae's
> `retire-cascade-host-shim` follow-up change once upstream is
> available.

## 1. Upstream: protocol + wrappers (sae-forge)

- [ ] 1.1 Add `saeforge.adapters.WorldModelAdapter` as a
      `typing.Protocol` with `n_features: int`,
      `extract_features(input_ids) -> Tensor`, and
      `project_into(basis) -> WorldModelAdapter`.
- [ ] 1.2 Add `saeforge.adapters.TransformerHostAdapter` that wraps a
      HuggingFace model and implements the protocol via the existing
      family-adapter machinery + `SubspaceProjector`.
- [ ] 1.3 Smoke test: a `GPT2LMHeadModel` wrapped in
      `TransformerHostAdapter` produces the same `(batch, seq,
      n_features)` extraction as today's `host.transformer(input_ids)`
      path.

## 2. Upstream: pipeline integration

- [ ] 2.1 `ForgePipeline.run_synthetic(host_model=None,
      world_model=None, ...)` — when `world_model` is None and
      `host_model` is not None, wrap in `TransformerHostAdapter`;
      otherwise use `world_model` directly. Both passed → warn +
      prefer `world_model`.
- [ ] 2.2 Refactor the internal pipeline to call
      `world_model.extract_features` instead of touching
      `host.transformer` directly. Single code path post-entrypoint.
- [ ] 2.3 Regression test: existing `host_model=...` call site produces
      byte-identical `ForgeResult` to the pre-change baseline on at
      least one fixture (sm-sae's `embedded__topk` is a good choice
      — short to run).

## 3. Upstream: result schema

- [ ] 3.1 `ForgeResult.host` becomes a `dict` with required key
      `family: Literal["transformer", "world_model"]` and family-
      specific extras. Transformer rows carry `model_id`,
      `n_params`, etc.; world-model rows carry `adapter_id`,
      `n_features`, and any adapter-supplied metadata.
- [ ] 3.2 Document the migration: any consumer reading the legacy
      free-text `host` field needs to parse the dict and switch on
      `family`.

## 4. Upstream: FaithfulnessTarget compatibility

- [ ] 4.1 Resolve the open question from `design.md` (Risks): does
      `FaithfulnessTarget.score` take `features=` directly, or does
      `WorldModelAdapter` expose a `torch_module`-compatible shim?
      Recommendation in the design is option (a); needs upstream
      sign-off.
- [ ] 4.2 If (a): bump the `FaithfulnessTarget` protocol minor version
      and document the new kwarg. If (b): document the
      `torch_module` requirement on `WorldModelAdapter`.
- [ ] 4.3 Update the bundled targets (`KLTarget`, `CosineTarget`,
      future `GroundTruthTarget` from the sm-sae upstream proposal)
      to use whichever path lands.

## 5. Upstream: release

- [ ] 5.1 Bump `saeforge` to v0.5. Changelog entry calls out the new
      kwarg, the result-schema change, and the backward-compat
      promise for `host_model=`.
- [ ] 5.2 README / docs update: a one-paragraph "WorldModelAdapter"
      section with a minimal example showing a non-transformer
      substrate.

## 6. Downstream: file `retire-cascade-host-shim` in sm-sae

- [ ] 6.1 New openspec change `retire-cascade-host-shim` once
      `saeforge>=0.5` is pinnable. Spec covers:
      - Adding `smsae.world_model.CascadeWorldModel` implementing
        `WorldModelAdapter`.
      - Switching `scripts/forge_pipeline.py:forge` to the
        `world_model=` entry point.
      - Deleting `smsae/host/`, `scripts/train_cascade_host.py`, and
        the `_build_synthetic_host` machinery (~700 LOC).
      - Removing the `runs/cascade_host/` artifact convention.
      - Adding a third class to the scoreboard's `host` column
        (🌐 `world_model`); flagging archived 🎓/🎲 runs as
        "legacy".
- [ ] 6.2 Acceptance gate for `retire-cascade-host-shim`: forge
      faithfulness on `cascade__jumprelu` with the WorldModel
      substrate is higher than the trained-shim baseline by ≥ 0.05
      absolute AUC (this is the gate
      [[add-cascade-host-shim]] 7.3 set but missed; it should be
      reachable once projection-loss is removed).

## 7. Cross-repo coordination

- [ ] 7.1 File a sae-forge issue linking this proposal; agree on the
      protocol shape (especially the FaithfulnessTarget question)
      before upstream implementation begins.
- [ ] 7.2 Once upstream lands, update this openspec README index to
      reflect the v0.5 release and unblock the
      `retire-cascade-host-shim` follow-up.

## 8. Acceptance gate (for this proposal — spec-only)

- [ ] 8.1 Proposal/design/tasks reviewed and approved.
- [ ] 8.2 sae-forge issue filed and linked.
- [ ] 8.3 Open question on FaithfulnessTarget compatibility (4.1) has
      an upstream decision recorded in `design.md`.
- [ ] 8.4 Move this change to `openspec/changes/archive/` once
      upstream v0.5 ships *or* the proposal is explicitly rejected
      (with reason recorded).
