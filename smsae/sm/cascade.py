"""SM cascade simulator: Monte Carlo on particle multi-sets.

State: a multi-set (Counter) of particle names.
Transition: a 1->N decay rule from a decay catalog. At each step we enumerate
applicable decays (whose parent is present and whose products are kinematically
allowed via m_parent > sum(m_daughters)), weight them by physical couplings
(Yukawa^2, |V_CKM|^2, gauge couplings), and sample.

The cascade halts when no parent in the current state has a kinematically
allowed decay. Stable particles emerge as a consequence (e, nu, gamma, light
quarks/gluons -- which in reality hadronize, but we stop at the parton level).

Branching fractions for H, Z, W, top, mu, tau are computed analytically from
the same weights as a sanity check against PDG.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

import smsae.sm.parameters as P
from smsae.sm.embeddings import (CONSERVED, COORDS, build_sm,
                            kinematic_threshold, vertex_residual)
from smsae.sm.checks import vertex_catalog


# ============================================================================
# Particle naming and classification
# ============================================================================
UP_FLAVORS     = ["u", "c", "t"]
DOWN_FLAVORS   = ["d", "s", "b"]
LEPTON_FLAVORS = ["e", "mu", "tau"]


def anti_name(name: str) -> str:
    """Antiparticle name."""
    fixed = {
        "photon": "photon", "Z": "Z", "H": "H",
        "W+": "W-", "W-": "W+",
        "e-": "e+", "e+": "e-",
        "mu-": "mu+", "mu+": "mu-",
        "tau-": "tau+", "tau+": "tau-",
        "nu_e": "~nu_e", "~nu_e": "nu_e",
        "nu_mu": "~nu_mu", "~nu_mu": "nu_mu",
        "nu_tau": "~nu_tau", "~nu_tau": "nu_tau",
        "g_rgbar": "g_grbar", "g_grbar": "g_rgbar",
        "g_gbbar": "g_bgbar", "g_bgbar": "g_gbbar",
        "g_rbbar": "g_brbar", "g_brbar": "g_rbbar",
        "g_diag3": "g_diag3", "g_diag8": "g_diag8",
    }
    if name in fixed:
        return fixed[name]
    return name[1:] if name.startswith("~") else "~" + name


def particle_info(name: str) -> tuple[str, str | None, str | None]:
    """Return (kind, flavor, color). kind in
    {quark, antiquark, lepton, antilepton, neutrino, antineutrino, gauge, higgs, gluon}.
    """
    if name in ("photon", "Z"):       return ("gauge", name, None)
    if name == "H":                    return ("higgs", "H", None)
    if name in ("W+", "W-"):           return ("gauge", name, None)
    if name.startswith("g_"):          return ("gluon", name, None)

    is_anti = name.startswith("~") or name.endswith("+")
    base = name.lstrip("~").rstrip("+-")

    if base in ("nu_e", "nu_mu", "nu_tau"):
        return ("antineutrino" if is_anti else "neutrino", base, None)
    if base in ("e", "mu", "tau"):
        return ("antilepton" if is_anti else "lepton", base, None)
    if "_" in base:
        flavor, color = base.split("_")
        if flavor in UP_FLAVORS + DOWN_FLAVORS:
            return ("antiquark" if is_anti else "quark", flavor, color)
    return ("unknown", None, None)


def particle_mass(sm, name: str) -> float:
    return float(sm[name].vec[8])


# ============================================================================
# Decay catalog
# ============================================================================
def build_decay_catalog(sm) -> list[tuple[str, str, list[str]]]:
    """Return [(label, parent, products), ...] of all 1->N decays we'll allow.

    Sources:
      - Filter sm_tests.vertex_catalog() to 1-incoming, kinematically allowed,
        and excluding radiation (parent in products) since that's mass-conserving.
      - Top quark decays t -> q W+ for q in (d, s, b), per color (not in catalog
        because catalog only wrote W -> qq' direction).
      - Muon decay mu -> e nu nu (effective 4-fermion via virtual W).
      - Tau decays: leptonic (-> e nu nu, mu nu nu) and hadronic (-> ud nu).
    """
    decays: list[tuple[str, str, list[str]]] = []

    # 1-to-N tree decays from the vertex catalog.
    for desc, inc, out in vertex_catalog():
        if len(inc) != 1 or len(out) < 2:
            continue
        parent = inc[0]
        if parent in out:
            continue  # mass-conserving radiation
        mp, md, ok = kinematic_threshold(sm, parent, out)
        if not ok or mp <= md + 1e-9:
            continue
        decays.append((desc.strip(), parent, list(out)))

    # Top decays (3 colors x 3 CKM channels each, for t and ~t)
    for col in ("r", "g", "b"):
        for q in DOWN_FLAVORS:
            decays.append((f"top   t_{col} -> {q}_{col} W+",
                           f"t_{col}", [f"{q}_{col}", "W+"]))
            decays.append((f"top  ~t_{col} -> ~{q}_{col} W-",
                           f"~t_{col}", [f"~{q}_{col}", "W-"]))

    # Muon decay (effective; in reality mu -> e nu_mu ~nu_e via virtual W)
    decays.append(("mu    mu- -> e- ~nu_e nu_mu", "mu-", ["e-", "~nu_e", "nu_mu"]))
    decays.append(("mu    mu+ -> e+ nu_e ~nu_mu", "mu+", ["e+",  "nu_e", "~nu_mu"]))

    # Tau decays (leptonic)
    decays.append(("tau   tau- -> e-  ~nu_e  nu_tau",  "tau-", ["e-",  "~nu_e",  "nu_tau"]))
    decays.append(("tau   tau- -> mu- ~nu_mu nu_tau",  "tau-", ["mu-", "~nu_mu", "nu_tau"]))
    decays.append(("tau   tau+ -> e+   nu_e ~nu_tau",  "tau+", ["e+",   "nu_e",  "~nu_tau"]))
    decays.append(("tau   tau+ -> mu+  nu_mu ~nu_tau", "tau+", ["mu+",  "nu_mu", "~nu_tau"]))
    # Tau decays (hadronic: tau- -> d_c ~u_c nu_tau, color-diagonal)
    for col in ("r", "g", "b"):
        decays.append((f"tau   tau- -> ~u_{col} d_{col} nu_tau",
                       "tau-", [f"~u_{col}", f"d_{col}", "nu_tau"]))
        decays.append((f"tau   tau+ -> u_{col} ~d_{col} ~nu_tau",
                       "tau+", [f"u_{col}", f"~d_{col}", "~nu_tau"]))

    # Verify every decay conserves all 7 SM charges (sanity).
    valid = []
    for desc, parent, products in decays:
        r = vertex_residual(sm, [parent], products)
        if np.max(np.abs(r)) < 1e-9:
            valid.append((desc, parent, products))
        else:
            nz = [(COORDS[i], float(r[i])) for i in range(7) if abs(r[i]) > 1e-9]
            print(f"  DROPPED (non-conserved): {desc}  {nz}")
    return valid


# ============================================================================
# Decay weights (physical rates, relative within each parent)
# ============================================================================
def decay_weight(sm, parent: str, products: list[str]) -> float:
    """Relative rate.  Sufficient for sampling and for branching fractions."""
    mp = particle_mass(sm, parent)
    md = sum(particle_mass(sm, p) for p in products)
    if md >= mp:
        return 0.0
    beta = float(np.sqrt(max(0.0, 1.0 - (md / mp) ** 2)))

    # Higgs Yukawa: H -> f f-bar (each color is a separate channel)
    if parent == "H":
        fname = next((p for p in products if not p.startswith("~")), None)
        if fname is None:
            return 0.0
        mf = particle_mass(sm, fname)
        y2 = (float(np.sqrt(2)) * mf / P.HIGGS_VEV) ** 2
        beta_f = float(np.sqrt(max(0.0, 1 - 4 * mf**2 / mp**2)))
        return y2 * beta_f ** 3

    # Z -> f f-bar: weight = g_L^2 + g_R^2 (per color)
    if parent == "Z":
        fname = next((p for p in products if not p.startswith("~")), None)
        if fname is None:
            return 0.0
        T3, Q = _z_couplings(fname)
        if T3 is None:
            return 0.0
        sw2 = P.SIN2_THETA_W
        gL = T3 - Q * sw2
        gR = -Q * sw2
        return gL ** 2 + gR ** 2

    # W^+- decays
    if parent in ("W+", "W-"):
        kinds = [particle_info(p) for p in products]
        # Leptonic (one lepton + one neutrino): weight = 1 per (leptonic) channel
        if any(k[0] in ("lepton", "antilepton") for k in kinds):
            return 1.0
        # Quark CC: weight = |V_{ij}|^2 (color-diagonal, each color is a channel)
        up = down = None
        for p in products:
            inf = particle_info(p)
            if inf[1] in UP_FLAVORS:    up = inf[1]
            if inf[1] in DOWN_FLAVORS:  down = inf[1]
        if up and down:
            i = UP_FLAVORS.index(up)
            j = DOWN_FLAVORS.index(down)
            return float(abs(P.CKM[i, j]) ** 2)
        return 0.0

    # Top quark decay t -> q W+, ~t -> ~q W-
    pinfo = particle_info(parent)
    if pinfo[1] == "t" and pinfo[0] in ("quark", "antiquark"):
        for p in products:
            inf = particle_info(p)
            if inf[1] in DOWN_FLAVORS:
                j = DOWN_FLAVORS.index(inf[1])
                return float(abs(P.CKM[2, j]) ** 2)
        return 0.0

    # mu / tau effective 4-fermion: phase space ~ m^5 with same kinematic factor
    # for all channels of a given parent.  Hadronic tau gets a color factor of 1
    # per color (3 colors summed = 3x leptonic, which matches PDG ~65% hadronic).
    if pinfo[0] in ("lepton", "antilepton"):
        return 1.0

    return 1.0


def _z_couplings(name: str) -> tuple[float | None, float | None]:
    """Return (T_3, Q) for the Z coupling, treating L chirality (rate uses both)."""
    info = particle_info(name)
    kind, flavor, _ = info
    if kind in ("quark", "antiquark") and flavor in UP_FLAVORS:    return (+0.5, +2/3)
    if kind in ("quark", "antiquark") and flavor in DOWN_FLAVORS:  return (-0.5, -1/3)
    if kind in ("lepton", "antilepton"):                            return (-0.5, -1.0)
    if kind in ("neutrino", "antineutrino"):                        return (+0.5,  0.0)
    return (None, None)


# ============================================================================
# Cascade engine
# ============================================================================
def available_transitions(state: Counter, decay_catalog) -> list[tuple[str, str, list[str]]]:
    return [(d, p, prod) for d, p, prod in decay_catalog if state.get(p, 0) > 0]


def step(sm, state: Counter, decay_catalog, rng) -> tuple[Counter, tuple | None]:
    trans = available_transitions(state, decay_catalog)
    if not trans:
        return state, None
    weights = [decay_weight(sm, p, prod) for _, p, prod in trans]
    total = sum(weights)
    if total <= 0:
        return state, None
    r = rng.uniform(0, total)
    cum = 0.0
    for (desc, parent, products), w in zip(trans, weights):
        cum += w
        if r <= cum:
            new_state = Counter(state)
            new_state[parent] -= 1
            if new_state[parent] == 0:
                del new_state[parent]
            for p in products:
                new_state[p] += 1
            return new_state, (desc, parent, products)
    return state, None


def cascade(sm, initial_state, decay_catalog, max_steps=100, rng=None):
    if rng is None:
        rng = random.Random()
    state = Counter(initial_state)
    history = [(dict(state), None)]
    for _ in range(max_steps):
        new_state, decay = step(sm, state, decay_catalog, rng)
        if decay is None:
            break
        state = new_state
        history.append((dict(state), decay))
    return history


# ============================================================================
# Branching fractions & summary helpers
# ============================================================================
def branching_fractions(sm, parent, decay_catalog) -> list[tuple[str, float]]:
    channels = [(d, p, prod) for d, p, prod in decay_catalog if p == parent]
    weights = [decay_weight(sm, p, prod) for _, p, prod in channels]
    total = sum(weights)
    if total == 0:
        return []
    return [(d, w / total) for (d, _, _), w in zip(channels, weights)]


def _canonicalize_final_state(products: list[str]) -> str:
    """Reduce a product list to a flavor-only multiset (drop color, drop sign)."""
    parts = []
    for p in products:
        inf = particle_info(p)
        if inf[0] in ("quark", "antiquark"):
            parts.append(inf[1] + "qbar" if inf[0] == "antiquark" else inf[1])
        elif inf[0] in ("lepton", "antilepton"):
            parts.append(inf[1])
        elif inf[0] in ("neutrino", "antineutrino"):
            parts.append("nu")
        else:
            parts.append(p)
    return " ".join(sorted(parts))


def aggregated_branching(sm, parent, decay_catalog) -> dict[str, float]:
    out = Counter()
    for desc, frac in branching_fractions(sm, parent, decay_catalog):
        # Reconstruct products from the catalog entry that matches this desc
        for d, p, prod in decay_catalog:
            if d == desc and p == parent:
                key = _canonicalize_final_state(prod)
                out[key] += frac
                break
    return dict(out)


def _format_state(state: dict) -> str:
    return " + ".join(f"{n}*{p}" if n > 1 else p for p, n in sorted(state.items()))


# ============================================================================
# Demo
# ============================================================================
if __name__ == "__main__":
    sm = build_sm()
    catalog = build_decay_catalog(sm)
    print(f"Decay catalog: {len(catalog)} channels\n")

    print("=" * 76)
    print("Branching fractions (aggregated by flavor, summed over color)")
    print("=" * 76)
    for parent, pdg_note in [
        ("H",    "PDG ~58% bb, ~21% WW*, ~9% gg(loop), ~6% ττ, ~3% cc, ~2.6% ZZ*"),
        ("Z",    "PDG ~20% nunu (3 flavors), ~10% ll (3 flavors), ~70% hadrons"),
        ("W+",   "PDG ~32% leptonic (3 flavors), ~67% hadronic"),
        ("t_r",  "PDG ~99.9% bW, ~0.1% sW, negligible dW"),
        ("tau-", "PDG ~17.8% eνν, ~17.4% μνν, ~64% hadronic"),
        ("mu-",  "PDG ~100% eνν"),
    ]:
        print(f"\n{parent}  ({pdg_note})")
        agg = aggregated_branching(sm, parent, catalog)
        if not agg:
            print("    (no kinematically allowed decays)")
            continue
        for key, frac in sorted(agg.items(), key=lambda x: -x[1]):
            print(f"  {frac * 100:7.3f}%  -> {key}")

    print("\n" + "=" * 76)
    print("Example cascades")
    print("=" * 76)
    rng = random.Random(42)
    for start in [{"H": 1}, {"t_r": 1}, {"Z": 1}, {"W+": 1, "W-": 1}, {"tau-": 1}]:
        print(f"\nstart:  {_format_state(start)}")
        hist = cascade(sm, start, catalog, max_steps=20, rng=rng)
        for state, decay in hist:
            tag = "" if decay is None else f"   <- {decay[0]}"
            print(f"  {_format_state(state)}{tag}")

    print("\n" + "=" * 76)
    print("Higgs cascade statistics (2000 trials)")
    print("=" * 76)
    rng = random.Random(0)
    final_particles = Counter()
    multiplicities = Counter()
    for _ in range(2000):
        hist = cascade(sm, {"H": 1}, catalog, max_steps=30, rng=rng)
        final = hist[-1][0]
        multiplicities[sum(final.values())] += 1
        for p, n in final.items():
            final_particles[p] += n

    print("\n  Final-state multiplicity distribution:")
    for k in sorted(multiplicities):
        bar = "#" * int(40 * multiplicities[k] / 2000)
        print(f"    {k:>2d} particles:  {multiplicities[k]:>5d}  {bar}")

    print("\n  Most common final particles (summed over 2000 cascades):")
    for p, n in sorted(final_particles.items(), key=lambda x: -x[1])[:15]:
        print(f"    {p:<14s}  {n}")
