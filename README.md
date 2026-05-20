# sm-sae

**The Standard Model as a tensor bundle, used as ground-truth substrate for SAE polysemanticity research.**

sm-sae bundles the Standard Model of particle physics into ~130 KB of named
tensors and a JSON labels sidecar, then uses that bundle as a synthetic
ground-truth fixture for sparse-autoencoder experiments. Because the SM's
feature factorization (charge ⊗ color ⊗ flavor ⊗ generation ⊗ chirality)
is exactly known, a trained SAE can be scored against truth without
post-hoc interpretation — making it an unusually clean benchmark for
polysemanticity and feature-discovery work.

The double meaning is intentional: **sm-sae** = "small SAE" *and* "Standard Model SAE".

## Status

Pre-alpha. The SM bundle and the Phase A/B/C SAE training-and-eval pipeline
are working; numbers below come from `scripts/train_all.py` →
`scripts/evaluate.py` → `scripts/polygram_demo.py`. Treat the design choices
as experimental.

## Why this exists

Two threads:

1. **Mechanistic interpretability needs ground-truth benchmarks.** Training
   SAEs on real LLM activations gives you features that "look interpretable"
   but you have no oracle to grade them against. The SM has exact
   ground-truth features at multiple granularities (per-charge, per-color,
   per-generation, per-particle, per-vertex-type) — so you can ask "did the
   SAE recover the known structure?" and get a numerical answer.

2. **Tensor representations as a neutral computational substrate.** If the
   rules of physics are computational, packaging the SM as a self-contained
   tensor file lets you study compressibility, low-rank structure, and
   feature factorization independently of any specific implementation.

The SM bundle is independently useful: it's the entire minimal SM
(19 free parameters, 168 vertex types, 11 representations of SU(2)×SU(3),
all Lie-algebra structure constants, CKM/PMNS, Yukawa matrices) in one
loadable file.

## Layout

```
sm-sae/
├── smsae/                   # the package
│   ├── sm/                  # Standard Model bundle (the substrate)
│   │   ├── embeddings.py    # 61 particles as dense vectors in R^9
│   │   ├── groups.py        # SU(2)/SU(3) generators and structure constants
│   │   ├── reps.py          # higher representations (1, 3, 3̄, 6, 8, 10)
│   │   ├── parameters.py    # PDG values: couplings, masses, CKM, PMNS
│   │   ├── lagrangian.py    # SM Lagrangian as 25 tensor-contraction records
│   │   ├── cascade.py       # vertex catalog + decay-cascade Monte Carlo
│   │   ├── checks.py        # GMN, anomaly cancellation, vertex closure tests
│   │   ├── export.py        # write sm_data.{npz,safetensors}
│   │   └── torch_model.py   # write sm.safetensors + sm_labels.json
│   ├── sae/                 # SAE training and evaluation
│   │   ├── data.py          # three feeds: raw / embedded / cascade
│   │   ├── models.py        # TopK, L1, JumpReLU SAEs
│   │   ├── train.py         # training loop with dead-neuron resampling
│   │   └── evaluation.py    # AUC-based ground-truth alignment scoring
│   └── polygram_bridge.py   # SM → polygram.Dictionary + cancellation runs
├── scripts/                 # entry points
│   ├── build_data.py        # generate data/*.{npz,safetensors,json}
│   ├── run_checks.py        # SM bundle consistency tests
│   ├── train_all.py         # Phase A: train 3 variants × 3 feeds
│   ├── evaluate.py          # Phase B: ground-truth alignment
│   └── polygram_demo.py     # Phase C: polygram fixture experiments
├── data/                    # generated artifacts (checked in)
│   ├── sm.safetensors       # PyTorch nn.Module state_dict (128 KB)
│   ├── sm_labels.json       # axis labels + layer categorization (12 KB)
│   ├── sm_data.npz          # flat numpy bundle (17 KB gzip)
│   └── sm_data.safetensors  # flat safetensors equivalent
├── runs/                    # SAE checkpoints + eval summaries (gitignored)
├── tests/                   # pytest (placeholder)
├── pyproject.toml
├── LICENSE                  # Apache 2.0
└── README.md
```

## Install

```bash
git clone https://github.com/jascal/sm-sae.git
cd sm-sae
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For the Phase C polygram demo, also install the polygram extra:

```bash
pip install -e ".[dev,polygram]"
```

### Intel macOS (x86_64) caveat

PyTorch's last x86_64 macOS wheels are torch 2.2.2 (CPython 3.10/3.11,
numpy<2). Use the `[intel]` extra in place of plain torch:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,intel,polygram]"
```

Apple Silicon and Linux/CUDA are unaffected.

## Quickstart

```bash
# Generate the SM data files (data/sm.safetensors, sm_data.npz, etc.)
python scripts/build_data.py

# Run all SM-bundle consistency checks (anomaly cancellation, vertex closure, ...)
python scripts/run_checks.py

# Phase A: train all 9 SAEs (3 variants × 3 data feeds)
python scripts/train_all.py

# Phase B: score every SAE against the ground-truth feature dictionary
python scripts/evaluate.py

# Phase C: convert SM bundle to a polygram Dictionary + run cancellation
python scripts/polygram_demo.py

# Optional: train a cascade-host shim for the sae-forge synthetic-host path.
# `--aux-supervision pooled` adds a supervised head over 5 per-state binary
# labels (charge / baryon conservation, top-quark lineage / existence, Higgs
# presence). Closes gate 7.3 if the trained-vs-random faithfulness delta on
# cascade__jumprelu reaches ≥ 0.05. Defaults to `off` (legacy LM-only loss).
python scripts/train_cascade_host.py --n-embd 61 --aux-supervision pooled
```

## What's in the SM bundle

| tensor | shape | meaning |
|---|---|---|
| `particle_vectors` | (61, 9) | dense vectors for every SM particle |
| `vertex_incidence` | (168, 59) | signed vertex-particle incidence; null space = conservation algebra (7-dim) |
| `su2_f`, `su3_f` | (3,3,3), (8,8,8) | Lie algebra structure constants (Jacobi-verified to ~1e-16) |
| `rep_su3_{1,3,3bar,6,8,10}` | various | irreducible representations |
| `ckm`, `pmns` | (3,3) complex | quark and lepton mixing matrices (unitary) |
| `yukawa_{up,down,lep}` | (3,3) diagonal | Yukawa eigenvalues |
| **trainable parameters** | 19 (or 26 with neutrinos) | the only "weights" of the SM |

Internal consistency checks (`scripts/run_checks.py`) verify:
- Gell-Mann–Nishijima `Q = T_3 + Y/2` for every Weyl spinor (7/7)
- All 5 gauge anomalies cancel per generation (~1e-15)
- 168/168 SM vertices close on (Q, B, L_e, L_μ, L_τ, C₃, C₈)
- Conservation algebra recovered exactly: `nullity(vertex_incidence) == 7`
- CKM unitarity to machine precision
- M_W, M_Z predicted from `g v / 2` and `√(g²+g'²) v / 2` to 0.4%

## Phase A/B/C results

**Phase A — Training (9 SAEs):** All converge to ≥96% variance explained.
TopK enforces exact L0; L1 lands moderate; JumpReLU runs hot at the default
`l0_coeff`. Runtime: ~45 seconds total on CPU.

**Phase B — Ground-truth alignment (AUC of SAE feature vs GT feature):**

| feed     | variant   | n_sae | active | cov≥0.95 | cov≥0.90 | mean_best | monosem |
|----------|-----------|-------|--------|----------|----------|-----------|---------|
| raw      | topk      | 32    | 32     | 56.4%    | 65.5%    | 0.898     | 100.0%  |
| raw      | l1        | 32    | 32     | 56.4%    | 66.4%    | 0.891     | 100.0%  |
| raw      | jumprelu  | 32    | 32     | 47.3%    | 58.2%    | 0.875     | 100.0%  |
| embedded | topk      | 64    | 63     | 70.0%    | 72.7%    | 0.902     | 100.0%  |
| embedded | l1        | 64    | 64     | 71.8%    | 72.7%    | 0.894     | 100.0%  |
| embedded | jumprelu  | 64    | 64     | 69.1%    | 74.5%    | 0.917     | 100.0%  |
| cascade  | topk      | 128   | 113    | 20.7%    | 30.6%    | 0.706     | 48.7%   |
| cascade  | l1        | 128   | 128    | 28.9%    | 33.9%    | 0.726     | 42.2%   |
| **cascade**  | **jumprelu**  | **128**   | **128**    | **37.2%**    | **45.5%**    | **0.767**     | **80.5%**   |

Notable: on the cascade feed, the SAE recovers features like `origin:t_b`,
`origin:mu+` at AUC=1.0 — it's inferring *which heavy particle started the
cascade* from the final-state bag of stable particles. That's genuine causal
inference, not memorization.

**Phase C — Polygram fixture (4 cancellation runs on an 8-particle slice):**

| pair | before | after | structural floor | met? |
|---|---|---|---|---|
| e⁻ / e⁺            | 0.085 | 0.085 | 0.085 | False |
| u_r / ū_r          | 0.382 | 0.382 | 0.382 | False |
| e⁻ / u_r           | 0.204 | 0.204 | 0.204 | False |
| photon / W⁺        | 0.593 | 0.593 | 0.593 | False |

All four cancellations hit `before == structural_floor` — meaning the
SM-derived dictionary is already at its phase-bound minimum overlap.
Further reduction requires the amplitude knobs (`alpha`, `gamma`,
`psi_aux`, `theta_amp`), not phase. This is a meaningful structural
statement about the encoding geometry, not a failure.

Against `polygram>=0.11.0` each result also carries
`at_structural_floor=True` (and a `UserWarning` from `Cancellation.run`)
so the at-floor outcome is observable programmatically — not just as
`before ≈ floor ≈ after` numerically. The bridge prints `cancellation
efficiency: N/A` for these cases rather than the misleading `0.00%`.

## Relationship to polygram / sae-forge

This project is a sandbox for ideas that feed into
[jascal/polygram](https://github.com/jascal/polygram) (quantum-inspired
analysis lab for SAE feature dictionaries) and
[jascal/sae-forge](https://github.com/jascal/sae-forge) (forges
polygram-compressed SAEs into small interpretable transformers).

What sm-sae provides that those projects don't:

- **A ground-truth-rich fixture** for evaluating SAE/polygram primitives
  beyond the toy animals example
- **A test bed for the SAE end-to-end workflow** (training → eval → polygram
  integration) on data small enough to iterate in seconds
- **An algebraic representation** of polysemanticity (the canonical
  multi-feature concept "u_r quark" = "up flavor" ⊗ "red color" ⊗
  "left-handed" ⊗ "first generation") with exact ground truth for whether
  decompositions recovered it

## Contributing

Pre-alpha. The most useful contributions right now are:
- More SAE variants (gated SAE, anthropic-style top-k-with-aux-loss)
- Larger SM-derived data distributions (multi-cascade simulation, deep
  decay trees)
- A polygram Compressor pass that operates on a trained SAE here
- Architecture variations of the SM (SO(10), SU(5)) to test bundle
  modularity

## License

Apache 2.0. See [LICENSE](LICENSE).
