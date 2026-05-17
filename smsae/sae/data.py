"""Data feeds for the sm-sae project.

Three feeds, all returning a common (X, ground_truth) interface where
  X              : torch.Tensor of shape (N, D) -- SAE training inputs
  ground_truth   : dict mapping each row of X to a multi-label set of
                   physically meaningful features it "should" decompose into.

Ground-truth feature dictionary (used in Phase B alignment scoring):
  - "particle:<name>"          -- identity (e.g. "particle:u_r")
  - "kind:<kind>"              -- quark/antiquark/lepton/.../gauge/higgs/gluon
  - "flavor:<flavor>"          -- u/d/s/c/b/t/e/mu/tau/nu_e/...
  - "color:<r|g|b>"             -- color for quarks
  - "chirality:<L|R|-|none>"   -- chirality (for fermions); '-' for bosons
  - "generation:<1|2|3>"       -- generation index for fermions
  - "charge_sign:<+|-|0>"       -- sign of Q
  - "is_charged"                -- |Q| > 0
  - "is_colored"                -- carries color
  - "is_antiparticle"           -- starts with "~" or ends with "+"

Feeds:
  feed_raw      : 61 particle vectors, D = 9
  feed_embedded : 61 particles -> learned random embedding, D = embed_dim (default 16)
  feed_cascade  : N cascade-generated events as bag-of-particles vectors, D = 61
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from smsae.sm.cascade import build_decay_catalog, cascade
from smsae.sm.embeddings import build_sm
from smsae.sm.cascade import particle_info


# ----------------------------------------------------------------------------
# Ground-truth feature dictionary
# ----------------------------------------------------------------------------
def particle_features(name: str) -> set[str]:
    """Return the ground-truth feature labels for a particle."""
    feats: set[str] = {f"particle:{name}"}
    kind, flavor, color = particle_info(name)
    feats.add(f"kind:{kind}")
    if flavor:
        feats.add(f"flavor:{flavor}")
    if color:
        feats.add(f"color:{color}")
        feats.add("is_colored")

    is_anti = name.startswith("~") or name.endswith("+")
    if is_anti:
        feats.add("is_antiparticle")

    # Charge sign (from particle_vectors[0])
    sm = build_sm()
    q = float(sm[name].vec[0])
    if q > 1e-9:
        feats.add("charge_sign:+")
        feats.add("is_charged")
    elif q < -1e-9:
        feats.add("charge_sign:-")
        feats.add("is_charged")
    else:
        feats.add("charge_sign:0")

    # Generation (for fermions only)
    gen_map = {
        "u": 1, "d": 1, "e": 1, "nu_e": 1,
        "c": 2, "s": 2, "mu": 2, "nu_mu": 2,
        "t": 3, "b": 3, "tau": 3, "nu_tau": 3,
    }
    if flavor in gen_map:
        feats.add(f"generation:{gen_map[flavor]}")

    # Chirality: pre-EWSB the SM is chiral, but post-EWSB names are Dirac.
    # Mark fermions generically without claiming L/R.
    if kind in ("quark", "antiquark", "lepton", "antilepton", "neutrino", "antineutrino"):
        feats.add("is_fermion")
    elif kind in ("gauge", "gluon"):
        feats.add("is_boson")
    elif kind == "higgs":
        feats.add("is_scalar")

    return feats


def all_ground_truth_features() -> list[str]:
    """The complete vocabulary of ground-truth feature labels across all particles."""
    sm = build_sm()
    feats: set[str] = set()
    for name in sm:
        feats |= particle_features(name)
    return sorted(feats)


# ----------------------------------------------------------------------------
# Feed dataclass
# ----------------------------------------------------------------------------
@dataclass
class Feed:
    name: str
    X: torch.Tensor                              # (N, D) training inputs
    sample_names: list[str | tuple[str, ...]]    # length-N labels
    sample_features: list[set[str]]              # length-N ground-truth feature sets
    feature_vocab: list[str]                     # sorted vocabulary of features
    notes: str = ""

    @property
    def N(self) -> int: return self.X.shape[0]
    @property
    def D(self) -> int: return self.X.shape[1]


# ----------------------------------------------------------------------------
# Feed 1: Raw particle vectors
# ----------------------------------------------------------------------------
def feed_raw() -> Feed:
    """61 particle vectors directly. Tiny -- SAE will mostly memorize, but
    the alignment-to-ground-truth eval still works."""
    sm = build_sm()
    names = sorted(sm.keys())
    X = torch.tensor(np.array([sm[n].vec for n in names]), dtype=torch.float32)
    feats = [particle_features(n) for n in names]
    return Feed(
        name="raw",
        X=X,
        sample_names=names,
        sample_features=feats,
        feature_vocab=all_ground_truth_features(),
        notes="61 particle vectors, 9 SM quantum-number coords. Tiny; meant as a sanity-check feed.",
    )


# ----------------------------------------------------------------------------
# Feed 2: Particle one-hots through a random embedding
# ----------------------------------------------------------------------------
def feed_embedded(embed_dim: int = 16, seed: int = 0) -> Feed:
    """Each particle is a one-hot; pass through a random linear embedding.

    This simulates how features get superimposed in a real model's residual
    stream: a vocabulary of P "concepts" is squashed into a D < P dimensional
    space, forcing polysemantic features.
    """
    sm = build_sm()
    names = sorted(sm.keys())
    P = len(names)
    rng = np.random.default_rng(seed)
    # Random projection with orthonormal-ish columns
    W = rng.standard_normal((P, embed_dim)).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    # Each row of X is the embedding of one particle
    one_hot = np.eye(P, dtype=np.float32)
    X = torch.tensor(one_hot @ W)
    feats = [particle_features(n) for n in names]
    return Feed(
        name=f"embedded_d{embed_dim}",
        X=X,
        sample_names=names,
        sample_features=feats,
        feature_vocab=all_ground_truth_features(),
        notes=(f"61 particle one-hots projected through a random {P}x{embed_dim} embedding. "
               f"Superposed features (P={P} > D={embed_dim}); SAE should disentangle."),
    )


# ----------------------------------------------------------------------------
# Feed 3: Cascade events as bag-of-particles vectors
# ----------------------------------------------------------------------------
def feed_cascade(n_events: int = 2000, seed: int = 0,
                 start_distribution: dict[str, float] | None = None) -> Feed:
    """Run the cascade simulator many times; encode each final state as a
    bag-of-particles count vector over the 61-particle vocabulary.

    This gives an unbounded synthetic training distribution. Ground-truth
    features = the union of per-particle features across the bag.
    """
    import random
    if start_distribution is None:
        # Default: sample heavy parents that produce interesting cascades
        start_distribution = {"H": 1.0, "Z": 1.0, "W+": 1.0, "W-": 1.0,
                              "t_r": 1.0, "t_g": 1.0, "t_b": 1.0,
                              "tau-": 0.5, "tau+": 0.5, "mu-": 0.3, "mu+": 0.3}
    start_keys = list(start_distribution.keys())
    start_weights = np.array(list(start_distribution.values()), dtype=float)
    start_weights /= start_weights.sum()

    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    sm = build_sm()
    names = sorted(sm.keys())
    name_to_col = {n: i for i, n in enumerate(names)}
    catalog = build_decay_catalog(sm)

    X = np.zeros((n_events, len(names)), dtype=np.float32)
    sample_names: list[tuple[str, ...]] = []
    sample_features: list[set[str]] = []
    for ev in range(n_events):
        parent = rng_np.choice(start_keys, p=start_weights)
        history = cascade(sm, {parent: 1}, catalog, max_steps=30, rng=rng_py)
        final = history[-1][0]
        feats: set[str] = set()
        labels: list[str] = []
        for p, n in final.items():
            X[ev, name_to_col[p]] = n
            labels.extend([p] * n)
            feats |= particle_features(p)
        # also tag the originating parent (interesting "lineage" feature)
        feats.add(f"origin:{parent}")
        sample_names.append(tuple(sorted(labels)))
        sample_features.append(feats)

    # Extra vocab tokens for origin: features
    extra = sorted({f"origin:{s}" for s in start_keys})
    vocab = sorted(set(all_ground_truth_features()) | set(extra))

    return Feed(
        name=f"cascade_n{n_events}",
        X=torch.tensor(X),
        sample_names=sample_names,
        sample_features=sample_features,
        feature_vocab=vocab,
        notes=(f"{n_events} cascade events. Each row is a 61-dim bag-of-particles count vector. "
               f"Initial parents sampled from {sorted(start_distribution.keys())}."),
    )


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("Feed: raw")
    print("=" * 78)
    f = feed_raw()
    print(f"  X shape: {tuple(f.X.shape)}    vocab: {len(f.feature_vocab)} features")
    print(f"  sample[0]: {f.sample_names[0]}")
    print(f"  features[0]: {sorted(f.sample_features[0])}")
    print(f"  notes: {f.notes}")

    print("\n" + "=" * 78)
    print("Feed: embedded")
    print("=" * 78)
    f = feed_embedded(embed_dim=16)
    print(f"  X shape: {tuple(f.X.shape)}    vocab: {len(f.feature_vocab)} features")
    print(f"  sample[0]: {f.sample_names[0]}")
    print(f"  ||X[0]||: {float(torch.linalg.norm(f.X[0])):.4f}")
    print(f"  notes: {f.notes}")

    print("\n" + "=" * 78)
    print("Feed: cascade")
    print("=" * 78)
    f = feed_cascade(n_events=500)
    print(f"  X shape: {tuple(f.X.shape)}    vocab: {len(f.feature_vocab)} features")
    print(f"  sample[0]: {f.sample_names[0]}")
    print(f"  features[0]: {sorted(f.sample_features[0])}")
    print(f"  avg # particles per event: {float(f.X.sum(axis=1).mean()):.2f}")
    print(f"  sparsity: {float((f.X == 0).float().mean()):.3f}")
    print(f"  notes: {f.notes}")
