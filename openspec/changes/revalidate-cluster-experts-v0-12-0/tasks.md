# tasks — revalidate-cluster-experts-v0-12-0

## 1. Re-validate on polygram v0.12.0

- [x] 1.1 Install pinned polygram v0.12.0 into the venv (was 0.10.0).
- [x] 1.2 Confirm `cluster_experts` / `ExpertDictionary` API unchanged (no fix).
- [x] 1.3 Re-run the demo on cascade__jumprelu + embedded__topk; confirm the
      PR #16 validation reproduces.

## 2. Durable capture

- [x] 2.1 Add a consolidated `summary.json` builder to
      `scripts/cluster_experts_demo.py` (`build_validation_summary` +
      `write_consolidated_summary`: per-cell headline + overall verdict +
      polygram version, reading the per-cell results.json).
- [x] 2.2 Force-add `runs/cluster_experts/summary.json` (compact), matching the
      sweep convention.
- [x] 2.3 Update the demo's module docstring (0.10.0 → v0.12.0 re-validation).
- [x] 2.4 Record the v0.12.0 validation result in `README.md`.

## 3. Tests + close-out

- [x] 3.1 Unit test for the summary builder (`tests/test_cluster_experts_summary.py`;
      pure dict→dict; no training).
- [x] 3.2 Full `pytest` green (83 passed).
- [ ] 3.3 PR; archive on landing.
