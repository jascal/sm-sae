"""End-to-end forge benchmark pipeline (scaffolding).

Stages:

  1.  load_sae(ckpt)                  load a runs/{feed}__{variant}.pt SAE
  2.  convert_to_safetensors(sae, …)  write W_dec/W_enc/b_enc/b_dec as safetensors
                                       (polygram's load_sae_safetensors expects this)
  3.  build_records(sae_safetensors)  → dict[int, SAEFeatureRecord]
  4.  build_dictionary(records, encoding)  → polygram.Dictionary
  5.  build_validation_report(sae, feed)   synthesize a ValidationReport from
                                            cascade-activation pairwise stats
  6.  compress(dictionary, vreport, sae)   run polygram.Compressor → *.compressed.safetensors
  7.  forge(compressed, host, faithfulness)  ← STUB; blocked on sae-forge release
  8.  score_against_gt(features, feed)      AUC of features vs sm-sae GT labels
  9.  write_report(run_dir)                 forge_results.json the scoreboard reads

Stages 1–4 and 8–9 are implemented now. Stages 5–6 are partially implemented
(see notes) and may need polygram-side adjustments. Stage 7 is intentionally
stubbed pending the upstream pluggable-Faithfulness release.

Usage:
    python scripts/forge_pipeline.py <feed>__<variant>
        [--encoding rung5] [--n-amp-qubits 4]
        [--select-by firing_rate|gt_alignment|head] [--out runs/sae_forge]

Encoding choices (stage 4):
    mps_rung1   cap=8     — tightest; only suitable for SAEs with very
                            few useful features.
    rung3       cap=16    — pre-2026-05 default; throws away most of
                            the cascade SAE's 128 features.
    rung4       cap=32
    rung5       cap=2**(n_amp_qubits + 3)  — default. With
                --n-amp-qubits 4 (the new default) cap=128, matching
                the cascade SAE width and exceeding the 110-feature
                GT vocabulary.

Selector choices (stage 4):
    firing_rate   keep the cap-many features that fire most often on the feed
                  (default — captures features the SAE actually uses)
    gt_alignment  keep features with highest |max AUC - 0.5| against GT labels
                  (most useful for the benchmark)
    head          keep the cap-many features with the lowest feature_ids
                  (legacy behaviour; reproducible but arbitrary)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Stage 1: load SAE
# ---------------------------------------------------------------------------
def load_sae(ckpt_path: str):
    """Load one of our runs/{feed}__{variant}.pt SAE checkpoints."""
    from smsae.sae.evaluation import load_sae as _load
    sae, meta = _load(ckpt_path)
    return sae, meta


# ---------------------------------------------------------------------------
# Stage 2: convert to safetensors (polygram-compatible layout)
# ---------------------------------------------------------------------------
def convert_to_safetensors(sae, out_path: str) -> str:
    """Write the SAE's W_dec, W_enc, b_enc, b_dec to a safetensors file using
    the canonical SAE-Lens key naming that polygram.load_sae_safetensors picks
    up (W_dec is the preferred decoder key).
    """
    from safetensors.torch import save_file
    state = {
        "W_dec": sae.W_dec.detach().contiguous(),   # (input_dim, n_features)
        "W_enc": sae.W_enc.detach().contiguous(),   # (n_features, input_dim)
        "b_enc": sae.b_enc.detach().contiguous(),
        "b_dec": sae.b_dec.detach().contiguous(),
    }
    save_file(state, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Stage 3: build SAEFeatureRecord dict from the safetensors
# ---------------------------------------------------------------------------
def build_records(safetensors_path: str) -> dict:
    """Use polygram's loader to get the dict[int, SAEFeatureRecord] form."""
    from polygram import load_sae_safetensors
    # W_dec is (input_dim, n_features) so polygram needs to know which axis is
    # features. Per its docstring, "W_dec" is row-as-feature (i.e. it expects
    # transpose). Our W_dec layout has features as columns of a (D, F) matrix,
    # so we need to fix this before/after.
    # Workaround: write the transpose under the W_dec key so the loader picks
    # the right rows.
    import safetensors.torch as st
    from safetensors.torch import save_file
    sd = st.load_file(safetensors_path)
    if sd["W_dec"].shape[0] != sd["W_enc"].shape[0]:
        # W_dec is (input_dim, n_features) but loader wants (n_features, input_dim)
        sd["W_dec"] = sd["W_dec"].T.contiguous()
        save_file(sd, safetensors_path)
    return load_sae_safetensors(safetensors_path)


# ---------------------------------------------------------------------------
# Stage 3.5: feature selectors — pick the top-cap SAE features for the encoding
# ---------------------------------------------------------------------------
# A selector maps (sae, feed, records) -> list[int] returning ALL feature IDs
# in preferred-keep order (best first). The caller applies `[:cap]` after, so
# selectors don't need to know the encoding cap. Returned IDs must be a
# permutation of `records.keys()` so the cap-slice is meaningful.

def select_by_head(sae, feed, records) -> list[int]:
    """Current behaviour: feature IDs in ascending numeric order.

    Useful as an explicit opt-in to reproduce pre-change runs.
    """
    return sorted(records.keys())


def select_by_firing_rate(sae, feed, records) -> list[int]:
    """Rank features by mean firing rate on the feed (descending).

    Captures features that actually fire on real inputs. Tie-break on
    feature_id ascending for determinism.
    """
    ids = sorted(records.keys())
    with torch.no_grad():
        z = sae(feed.X).z.detach().cpu()        # (N, F_full)
    rate = (z.abs() > 1e-9).float().mean(dim=0).numpy()  # (F_full,)
    # Restrict to the feature IDs that exist in records (defensive: usually
    # records covers 0..F_full-1, but don't assume).
    scores = [(-float(rate[i]), i) for i in ids]
    scores.sort()  # ascending by -rate (i.e. descending rate); ties → id asc
    return [i for _, i in scores]


def select_by_gt_alignment(sae, feed, records) -> list[int]:
    """Rank features by max AUC across GT labels (descending).

    Captures features most useful for the benchmark. Tie-break on
    feature_id ascending for determinism.
    """
    from smsae.sae.evaluation import auc_matrix, build_gt_matrix
    ids = sorted(records.keys())
    with torch.no_grad():
        z = sae(feed.X).z.detach().cpu().numpy().astype(np.float32)
    Y = build_gt_matrix(feed)
    A = auc_matrix(z, Y)                         # (F_full, M)
    # Per-feature "usefulness" = best AUC across any GT label. Center around
    # 0.5 so a feature with no signal scores 0 (random AUC), and a perfectly
    # anti-correlated feature also scores high — we care about discriminative
    # power, not sign.
    best = np.abs(A.max(axis=1) - 0.5) if A.size else np.zeros(len(ids))
    scores = [(-float(best[i]), i) for i in ids]
    scores.sort()
    return [i for _, i in scores]


SELECTORS = {
    "head":          select_by_head,
    "firing_rate":   select_by_firing_rate,
    "gt_alignment":  select_by_gt_alignment,
}


def _resolve_selector(arg):
    """Accept either a registry key (str) or a user-supplied callable.

    Returns the resolved callable. Raises ValueError for unknown keys.
    """
    if callable(arg):
        return arg
    if isinstance(arg, str):
        if arg not in SELECTORS:
            raise ValueError(
                f"unknown selector {arg!r}; choose from {sorted(SELECTORS)} "
                f"or pass a callable (sae, feed, records) -> list[int]")
        return SELECTORS[arg]
    raise TypeError(f"selector must be str or callable, got {type(arg).__name__}")


# ---------------------------------------------------------------------------
# Stage 4: SAEFeatureRecord dict → polygram.Dictionary
# ---------------------------------------------------------------------------
def build_dictionary(records: dict, encoding_name: str = "rung5",
                     selector="firing_rate", sae=None, feed=None,
                     n_amp_qubits: int = 4):
    """Wrap records as a polygram Dictionary using the chosen encoding.

    The selector decides which `cap` of the SAE's features the Dictionary
    sees (cap = encoding.max_features). `selector` is a SELECTORS key or
    a callable (sae, feed, records) -> list[int]. `sae`/`feed` are
    required for any selector other than `"head"`.

    `n_amp_qubits` only matters for the `rung5` encoding and sets its
    cap = 2 ** (n_amp_qubits + 3) (so n_amp_qubits=4 → cap=128). The
    default of 4 matches the cascade SAE's 128 features end-to-end and
    is sized to cover the full 110-feature GT vocabulary with room to
    spare.

    Returns (dictionary, sel_report, selection_meta) where selection_meta
    is `{method, n_candidates, n_kept, kept_ids, ordered_ids}`.
    """
    from polygram import from_sae_lens, MPSRung1, Rung3, Rung4, Rung5
    builders = {
        "mps_rung1": lambda: MPSRung1(bond_dim=2, phase_knobs=True),
        "rung3":     lambda: Rung3(bond_dim=2),
        "rung4":     lambda: Rung4(bond_dim=2),
        "rung5":     lambda: Rung5(bond_dim=2, n_amp_qubits=n_amp_qubits),
    }
    if encoding_name not in builders:
        raise ValueError(f"unknown encoding {encoding_name!r}; "
                         f"choose from {list(builders)}")
    encoding = builders[encoding_name]()

    selector_fn = _resolve_selector(selector)
    method_name = selector if isinstance(selector, str) else getattr(
        selector, "__name__", "custom")
    if method_name != "head" and (sae is None or feed is None):
        raise ValueError(
            f"selector={method_name!r} needs sae and feed; only 'head' "
            f"works without them.")

    ordered_ids = list(selector_fn(sae, feed, records))
    if sorted(ordered_ids) != sorted(records.keys()):
        raise ValueError(
            f"selector {method_name!r} did not return a permutation of "
            f"records.keys(); got {len(ordered_ids)} ids "
            f"(expected {len(records)}).")

    encoding_cap = getattr(encoding, "max_features", len(ordered_ids))
    # Polygram's Compressor indexes into arrays of shape (..., input_dim)
    # with feature IDs, so it raises IndexError when cap > sae.input_dim.
    # Clamp to keep the pipeline robust; downstream gates the actual
    # feature ceiling on min(encoding cap, SAE feature count, SAE input
    # dim). The Compressor's overcomplete-basis support is tracked as a
    # follow-up upstream.
    sae_n_features = len(records)
    sae_input_dim = (
        int(getattr(sae, "input_dim", sae_n_features))
        if sae is not None else sae_n_features
    )
    cap = min(encoding_cap, sae_n_features, sae_input_dim)
    if cap < encoding_cap:
        import warnings
        warnings.warn(
            f"encoding {encoding_name!r} cap={encoding_cap} clamped to "
            f"{cap} (= min(encoding cap, n_records={sae_n_features}, "
            f"sae.input_dim={sae_input_dim})). The Compressor cannot "
            f"index into overcomplete bases; this is a polygram-side "
            f"limit, not an sm-sae policy.",
            UserWarning, stacklevel=2,
        )
    kept_ids = ordered_ids[:cap]

    dictionary, sel_report = from_sae_lens(
        records,
        feature_ids=kept_ids,
        name=f"sm_sae_{encoding_name}",
        encoding=encoding,
        assign_amp_knobs=True,
        assign_phase_knobs=True,
    )
    selection_meta = {
        "method":        method_name,
        "n_candidates":  len(records),
        "n_kept":        len(kept_ids),
        "kept_ids":      [int(i) for i in kept_ids],
        "ordered_ids":   [int(i) for i in ordered_ids],
        "encoding_cap":  int(encoding_cap),
        "sae_input_dim": int(sae_input_dim),
        "cap_applied":   int(cap),
    }
    return dictionary, sel_report, selection_meta


# ---------------------------------------------------------------------------
# Stage 5: synthesize a ValidationReport from cascade activations
# ---------------------------------------------------------------------------
def synthesize_validation_report(sae, dictionary, feed,
                                 polygram_threshold: float = 0.3,
                                 jaccard_threshold: float = 0.05,
                                 min_firing_rate: float = 0.001,
                                 min_both_fire: int = 1):
    """Build a ValidationReport by computing pairwise stats on the feed's
    activations (no LLM/host model required).

    Caveat: BehaviouralValidator normally fills `kl_ablate_*` and
    `kl_log_ratio_abs` from a host-model ablation, which we don't have.
    We set them to 0.0 — Compressor's `merge` strategy primarily uses
    `polygram_overlap` and `jaccard`, so this should still produce a
    usable plan. If Compressor errors, this is the first place to look.
    """
    from polygram import (BucketStats, CandidatePair, ValidationReport,
                          ValidationSummary)

    with torch.no_grad():
        Z = sae(feed.X).z.detach().cpu().numpy()  # (N, F)

    N, F_full = Z.shape
    n_features = len(dictionary.features)
    Z = Z[:, :n_features]  # restrict to the features the Dictionary chose

    # Firing matrix
    fires = (Z > 1e-9).astype(np.int64)
    n_fires = fires.sum(axis=0).astype(int)  # (n_features,)

    # Gram from polygram dictionary (squared overlaps |⟨a|b⟩|² in [0,1])
    gram = np.abs(np.asarray(dictionary.gram())) ** 2

    # Decoder overlap (cosine of decoder columns)
    Wd = sae.W_dec.detach().cpu().numpy()  # (D, F_full)
    Wd = Wd[:, :n_features]
    Wd_norm = Wd / (np.linalg.norm(Wd, axis=0, keepdims=True) + 1e-9)
    decoder_overlap = np.abs(Wd_norm.T @ Wd_norm)  # (n, n)

    pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            both = int(((fires[:, i] == 1) & (fires[:, j] == 1)).sum())
            either = int(((fires[:, i] == 1) | (fires[:, j] == 1)).sum())
            jac = both / either if either > 0 else 0.0
            # Pearson on continuous activations
            zi = Z[:, i]; zj = Z[:, j]
            mi, mj = zi.mean(), zj.mean()
            num = ((zi - mi) * (zj - mj)).sum()
            den = (np.sqrt(((zi - mi) ** 2).sum() * ((zj - mj) ** 2).sum())
                   + 1e-12)
            pearson = float(num / den)
            gate_pass = (float(gram[i, j]) >= polygram_threshold
                         and jac >= jaccard_threshold
                         and n_fires[i] / N >= min_firing_rate
                         and n_fires[j] / N >= min_firing_rate
                         and both >= min_both_fire)
            pairs.append(CandidatePair(
                i=i, j=j,
                polygram_overlap=float(gram[i, j]),
                decoder_overlap=float(decoder_overlap[i, j]),
                jaccard=jac,
                pearson_activation=pearson,
                kl_ablate_i=0.0, kl_ablate_j=0.0,
                kl_ratio_paired=0.0, kl_log_ratio_abs=0.0,
                n_fires_i=int(n_fires[i]),
                n_fires_j=int(n_fires[j]),
                n_both_fire=both,
                n_either_fire=either,
                gate_pass=gate_pass,
            ))

    confirmed = tuple((p.i, p.j) for p in pairs if p.gate_pass)
    summary = ValidationSummary(
        spearman_polygram_jaccard=0.0,
        spearman_decoder_jaccard=0.0,
        spearman_polygram_log_kl_abs=0.0,
        pearson_polygram_jaccard=0.0,
        pearson_decoder_jaccard=0.0,
        buckets={},
        outcome=f"synthesized from {N} feed activations",
    )
    return ValidationReport(
        schema_version=1,
        dictionary_name=dictionary.name,
        model_name="smsae.cascade",      # not a real LLM
        layer=0,
        n_prompts=int(N),
        n_tokens=int(N),
        polygram_overlap_threshold=polygram_threshold,
        jaccard_threshold=jaccard_threshold,
        min_firing_rate=min_firing_rate,
        min_both_fire=min_both_fire,
        feature_ids=tuple(range(n_features)),
        pairs=tuple(pairs),
        summary=summary,
        confirmed=confirmed,
    )


# ---------------------------------------------------------------------------
# Stage 6: run polygram.Compressor
# ---------------------------------------------------------------------------
def compress(vreport, sae_safetensors_path: str, out_path: str,
             strategy: str = "merge"):
    """Run polygram.Compressor end-to-end; emit *.compressed.safetensors."""
    from polygram import Compressor
    from pathlib import Path
    compressor = Compressor(
        validation_report=vreport,
        sae_checkpoint=Path(sae_safetensors_path),
        strategy=strategy,
    )
    result = compressor.run(Path(out_path))
    return result


# ---------------------------------------------------------------------------
# Stage 7: forge into a small transformer (sae-forge >= 0.4 with pluggable
# Faithfulness)
# ---------------------------------------------------------------------------
class GroundTruthAlignment:
    """FaithfulnessTarget that scores forged-model residual activations vs
    sm-sae ground-truth labels via AUC. Conforms to
    saeforge.eval.FaithfulnessTarget (Protocol).

    Holds the GT label matrix internally; reads sample input_ids from
    ctx['_eval_input_ids']. Returns (mean_best_auc, 1 - mean_best_auc) so
    that with better_when='higher' the perplexity-analog the FSM consumes
    is a positive-real *decreasing* function of the score, per the protocol
    contract.
    """
    name = "gt_alignment"
    better_when = "higher"

    def __init__(self, labels: np.ndarray, layer: int = -1):
        self.labels = labels.astype(np.float32)
        self.layer = layer

    def score(self, *, forged, host, ctx):
        input_ids = ctx.get("_eval_input_ids")
        if input_ids is None:
            raise KeyError(
                "GroundTruthAlignment.score requires ctx['_eval_input_ids']; "
                "pass eval_input_ids= to ForgePipeline.run_synthetic().")
        device = ctx.get("device", "cpu")
        torch_mod = forged.torch_module.to(device).eval()
        with torch.no_grad():
            # sae-forge's ForgedGPT2.transformer(ids) returns the residual
            # stream directly as a Tensor of shape (batch, seq, n_features).
            # That's exactly what we want — the residual width equals the
            # SAE's kept-feature count, so each coordinate is a feature.
            if hasattr(torch_mod, "transformer"):
                hidden = torch_mod.transformer(input_ids.to(device))
                if not isinstance(hidden, torch.Tensor):
                    # HuggingFace-shaped return (older code paths)
                    hidden = (hidden.last_hidden_state
                              if hasattr(hidden, "last_hidden_state")
                              else hidden[0])
            else:
                # No transformer attribute — fall back to logits and warn
                hidden = torch_mod(input_ids.to(device))
        # Mean-pool over the sequence dim: (N, T, F) -> (N, F)
        feats = hidden.mean(dim=1).cpu().numpy().astype(np.float32)
        # AUC of each feature vs each GT label
        from smsae.sae.evaluation import auc_matrix
        # Trim labels to match the number of samples we actually have
        N = min(feats.shape[0], self.labels.shape[0])
        A = auc_matrix(feats[:N], self.labels[:N])
        best_per_label = A.max(axis=0) if A.size else np.zeros(self.labels.shape[1])
        mean_best = float(best_per_label.mean())
        # better_when='higher' → perplexity-analog must be decreasing; the
        # canonical 1 - score (clamped) is fine here.
        perplexity_analog = max(1.0 - mean_best, 1e-6)
        return mean_best, perplexity_analog


def _build_synthetic_host(input_dim: int, vocab_size: int = 64,
                          num_layers: int = 1, num_heads: int = 2,
                          seed: int = 0):
    """Return `(model, info)` for the sae-forge synthetic host.

    First tries to load a cascade-trained host from
    `runs/cascade_host/<input_dim>/host/`; if not found, emits a
    UserWarning and falls back to a random-init tiny GPT-2.

    With a trained host the forged model's residuals carry real cascade
    signal and GroundTruthAlignment scores are interpretable. With the
    random-init fallback they reflect projection noise; the warning
    points users at scripts/train_cascade_host.py.
    """
    import warnings

    canonical_dir = os.path.join(REPO_ROOT, "runs", "cascade_host",
                                 str(input_dim))
    host_dir = os.path.join(canonical_dir, "host")
    cfg_path = os.path.join(canonical_dir, "config.json")
    if os.path.isdir(host_dir):
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(host_dir)
        train_meta: dict = {}
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    train_meta = json.load(f)
            except Exception:
                train_meta = {}
        info = {
            "kind":                "trained",
            "path":                host_dir,
            "n_embd":              int(model.config.n_embd),
            "n_layer":             int(model.config.n_layer),
            "n_head":              int(model.config.n_head),
            "n_params":            int(sum(p.numel() for p in model.parameters())),
            "train_loss_final":    train_meta.get("train_loss_final"),
            "n_train_trajectories": train_meta.get("n_trajectories"),
            "n_train_steps":       train_meta.get("n_train_steps"),
            "seed":                train_meta.get("seed"),
        }
        return model, info

    warnings.warn(
        f"no trained cascade host at {host_dir}; falling back to "
        f"random-init tiny GPT-2. Forge faithfulness scores will not "
        f"reflect cascade structure. To train one: "
        f"`python scripts/train_cascade_host.py --n-embd {input_dim}`",
        UserWarning, stacklevel=2,
    )
    from transformers import GPT2Config, GPT2LMHeadModel
    # input_dim must be divisible by num_heads
    if input_dim % num_heads != 0:
        num_heads = 1
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_embd=input_dim,
        n_layer=num_layers,
        n_head=num_heads,
        n_inner=input_dim * 2,
        n_positions=64,
    )
    model = GPT2LMHeadModel(cfg)
    info = {
        "kind":      "random_init",
        "path":      None,
        "n_embd":    int(input_dim),
        "n_layer":   int(num_layers),
        "n_head":    int(num_heads),
        "n_params":  int(sum(p.numel() for p in model.parameters())),
        "seed":      int(seed),
    }
    return model, info


def _encode_feed_as_input_ids(feed, vocab_size: int = 64,
                              max_seq: int = 16) -> "torch.Tensor":
    """Turn each feed sample into an int64 input_ids row.

    cascade feed (bag-of-particles counts): each particle appears `count`
    times in the sequence, sorted, truncated/padded to max_seq.
    other feeds: argmax per sample (single token).
    """
    X = feed.X.numpy()
    N = X.shape[0]
    rows = np.zeros((N, max_seq), dtype=np.int64)
    if X.shape[1] == 1 or X.dtype.kind == "f" and X.min() < 0:
        # continuous-valued feed; fall back to argmax per sample
        idx = X.argmax(axis=1) % vocab_size
        rows[:, 0] = idx
    else:
        # count-vector feed
        for i in range(N):
            counts = X[i]
            tokens = []
            for j in range(min(len(counts), vocab_size)):
                tokens.extend([j] * int(counts[j]))
            tokens = tokens[:max_seq]
            for k, t in enumerate(tokens):
                rows[i, k] = t
    return torch.from_numpy(rows)


def forge(compressed_path: str, sae, feed, run_dir: str) -> dict:
    """Stage 7: actually run sae-forge's ForgePipeline.run_synthetic with our
    GroundTruthAlignment target."""
    from saeforge import FeatureBasis, SubspaceProjector, ForgePipeline
    from smsae.sae.evaluation import build_gt_matrix

    basis = FeatureBasis.from_polygram_checkpoint(compressed_path)
    if basis.n_features == 0:
        return {
            "status":  "skipped",
            "reason":  "compressed basis has 0 kept features; nothing to forge",
        }
    projector = SubspaceProjector(basis, scale_boost=1.0)

    host, host_info = _build_synthetic_host(input_dim=sae.input_dim)
    input_ids = _encode_feed_as_input_ids(feed)
    Y = build_gt_matrix(feed)
    target = GroundTruthAlignment(labels=Y)

    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        host_model_id=None,                     # synthetic-host path
        eval_prompts=[],
        faithfulness=target,
        orchestrator="imperative",
        device="cpu",
        dtype="float32",
    )
    out_dir = os.path.join(run_dir, "forge")
    os.makedirs(out_dir, exist_ok=True)
    try:
        result = pipeline.run_synthetic(host, output_dir=out_dir,
                                         eval_input_ids=input_ids)
    except Exception as e:
        return {
            "status":  "error",
            "reason":  f"{type(e).__name__}: {e}",
            "host":    host_info,
        }
    return {
        "status":                   "ok",
        "faithfulness":             float(result.faithfulness),
        "faithfulness_target_name": result.faithfulness_target_name,
        "n_params":                 int(result.n_params),
        "host":                     host_info,
    }


# ---------------------------------------------------------------------------
# Stage 8: score features against sm-sae ground truth
# ---------------------------------------------------------------------------
def score_against_gt(activations: np.ndarray, feed) -> dict:
    """AUC of each feature column vs each GT label column. Returns a small
    summary dict shaped like the entries the scoreboard already consumes."""
    from smsae.sae.evaluation import auc_matrix, build_gt_matrix
    Y = build_gt_matrix(feed)
    A = auc_matrix(activations.astype(np.float32), Y)
    best_per_gt = A.max(axis=0) if A.size else np.zeros(Y.shape[1])
    return {
        "n_features":      int(activations.shape[1]),
        "n_gt_features":   int(Y.shape[1]),
        "coverage_0.95":   float((best_per_gt >= 0.95).mean()),
        "coverage_0.90":   float((best_per_gt >= 0.90).mean()),
        "mean_best_auc":   float(best_per_gt.mean()),
    }


# ---------------------------------------------------------------------------
# Stage 9: write final report
# ---------------------------------------------------------------------------
def write_report(run_dir: str, payload: dict) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "forge_results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id",
                    help="e.g. embedded__topk; reads runs/<run_id>.pt")
    ap.add_argument("--encoding", default="rung5",
                    choices=["mps_rung1", "rung3", "rung4", "rung5"],
                    help="polygram encoding family. Default rung5 (cap "
                         "= 2 ** (n_amp_qubits + 3); with the default "
                         "--n-amp-qubits 4 that is 128 features, "
                         "matching the cascade SAE width and exceeding "
                         "the 110-feature GT vocabulary).")
    ap.add_argument("--n-amp-qubits", type=int, default=4, dest="n_amp_qubits",
                    help="number of amplitude qubits for rung5; cap = "
                         "2 ** (n_amp_qubits + 3). Choices 1..4 give "
                         "caps 16/32/64/128. Ignored for other "
                         "encodings.")
    ap.add_argument("--select-by", default="firing_rate",
                    choices=sorted(SELECTORS.keys()),
                    dest="select_by",
                    help="how to pick the cap-many SAE features for the "
                         "polygram Dictionary (default: firing_rate).")
    ap.add_argument("--feed", default=None,
                    help="override feed name; default inferred from run_id")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "sae_forge"))
    ap.add_argument("--n-cascade-events", type=int, default=2000)
    args = ap.parse_args()

    run_dir = os.path.join(args.out, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    feed_name = args.feed or args.run_id.split("__")[0]
    print(f"forge_pipeline: run_id={args.run_id}  feed={feed_name}  "
          f"encoding={args.encoding}  out={run_dir}")

    # Build feed
    from smsae.sae.data import feed_cascade, feed_embedded, feed_raw
    feed_builders = {
        "raw":      feed_raw,
        "embedded": lambda: feed_embedded(embed_dim=16, seed=0),
        "cascade":  lambda: feed_cascade(n_events=args.n_cascade_events, seed=0),
    }
    feed = feed_builders[feed_name]()

    # Stage 1
    ckpt = os.path.join(REPO_ROOT, "runs", f"{args.run_id}.pt")
    print(f"  [1] load SAE  ←  {ckpt}")
    sae, meta = load_sae(ckpt)

    # Stage 2
    safetensors_path = os.path.join(run_dir, "sae.safetensors")
    print(f"  [2] write safetensors  →  {safetensors_path}")
    convert_to_safetensors(sae, safetensors_path)

    # Stage 3
    print(f"  [3] build SAEFeatureRecord dict")
    records = build_records(safetensors_path)
    print(f"      {len(records)} features loaded")

    # Stage 4
    print(f"  [4] wrap as polygram.Dictionary "
          f"(encoding={args.encoding}, select-by={args.select_by})")
    dictionary, sel_report, selection_meta = build_dictionary(
        records, args.encoding,
        selector=args.select_by, sae=sae, feed=feed,
        n_amp_qubits=args.n_amp_qubits,
    )
    print(f"      {len(dictionary.features)} features kept "
          f"(encoding cap = {getattr(dictionary.encoding, 'max_features', '?')}; "
          f"selection = {selection_meta['method']})")

    # Stage 5
    print(f"  [5] synthesize ValidationReport from {feed.N} feed activations")
    vreport = synthesize_validation_report(sae, dictionary, feed)
    print(f"      {len(vreport.pairs)} candidate pairs, "
          f"{len(vreport.confirmed)} confirmed")
    vreport_path = os.path.join(run_dir, "validation_report.json")
    vreport.to_json(vreport_path)
    print(f"      saved → {vreport_path}")

    # Stage 6
    compressed_path = os.path.join(run_dir, "sae.compressed.safetensors")
    print(f"  [6] run Compressor  →  {compressed_path}")
    try:
        comp_result = compress(vreport, safetensors_path, compressed_path)
        comp_summary = {
            "n_clusters":       comp_result.report.n_clusters,
            "n_features_kept":  comp_result.report.n_features_kept,
            "n_features_zeroed": comp_result.report.n_features_zeroed,
        }
        print(f"      clusters={comp_summary['n_clusters']}  "
              f"kept={comp_summary['n_features_kept']}  "
              f"zeroed={comp_summary['n_features_zeroed']}")
    except Exception as e:
        comp_summary = {"error": f"{type(e).__name__}: {e}"}
        print(f"      Compressor failed: {comp_summary['error']}")

    # Stage 7 — real sae-forge call (>= 0.4)
    print(f"  [7] sae-forge ForgePipeline.run_synthetic with "
          f"GroundTruthAlignment")
    forge_summary = forge(compressed_path, sae, feed, run_dir)
    if forge_summary.get("status") == "ok":
        print(f"      faithfulness={forge_summary['faithfulness']:.4f} "
              f"({forge_summary['faithfulness_target_name']})  "
              f"n_params={forge_summary['n_params']}")
    else:
        print(f"      {forge_summary['status']}: "
              f"{forge_summary.get('reason', '?')}")

    # Stage 8 — score the *uncompressed* SAE features against GT as a baseline
    # so we have a number to compare the forged model against once stage 7 is real
    print(f"  [8] score SAE features against ground truth (baseline)")
    with torch.no_grad():
        Z_sae = sae(feed.X).z.detach().cpu().numpy()
    baseline_scores = score_against_gt(Z_sae, feed)
    print(f"      baseline: cov≥0.95 = {baseline_scores['coverage_0.95']:.1%}  "
          f"mean_best_auc = {baseline_scores['mean_best_auc']:.3f}")

    # Stage 9
    payload = {
        "run_id":      args.run_id,
        "feed":        feed_name,
        "encoding":    args.encoding,
        "n_features":  int(sae.n_features),
        "input_dim":   int(sae.input_dim),
        "dictionary":  {
            "name": dictionary.name,
            "n_features": len(dictionary.features),
            "encoding_max": getattr(dictionary.encoding, "max_features", None),
            "selection": selection_meta,
        },
        "compress":    comp_summary,
        "forge":       forge_summary,
        "baseline_score": baseline_scores,
        "forge_score":    (forge_summary.get("faithfulness")
                           if forge_summary.get("status") == "ok"
                           else None),
    }
    out_path = write_report(run_dir, payload)
    print(f"  [9] wrote {out_path}")


if __name__ == "__main__":
    main()
