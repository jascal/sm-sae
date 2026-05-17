"""Dense vector embeddings of Standard Model particles.

Each particle is a point in R^d where the coordinates are the conserved
additive quantum numbers. Gauge vertices and allowed decays become exact
linear-algebra statements: at any SM vertex, the (signed) sum of particle
vectors is zero on the conserved-charge subspace.

Coordinate layout (CONSERVED at every SM vertex):
    0  Q        electric charge
    1  B        baryon number
    2  L_e      electron family number
    3  L_mu     muon family number
    4  L_tau    tau family number
    5  C3       color SU(3) Cartan T3 (lambda_3 / 2 eigenvalue)
    6  C8       color SU(3) Cartan T8 (lambda_8 / 2 eigenvalue)

Side coordinates (NOT linearly conserved, kept for reference):
    7  spin
    8  mass_GeV (PDG central values; 0 for massless / unstable to define)

Antiparticles are obtained by negating the conserved block (indices 0..6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CONSERVED = slice(0, 7)
DIM = 9
COORDS = ["Q", "B", "L_e", "L_mu", "L_tau", "C3", "C8", "spin", "mass_GeV"]

# --- Color SU(3) Cartan vectors --------------------------------------------
# Fundamental rep weights (quark colors): three vertices of a triangle.
COLOR_R = (0.5, 1 / (2 * math.sqrt(3)))
COLOR_G = (-0.5, 1 / (2 * math.sqrt(3)))
COLOR_B = (0.0, -1 / math.sqrt(3))
COLORLESS = (0.0, 0.0)


def _v(Q, B, Le, Lmu, Ltau, color, spin, mass) -> np.ndarray:
    c3, c8 = color
    return np.array([Q, B, Le, Lmu, Ltau, c3, c8, spin, mass], dtype=float)


@dataclass(frozen=True)
class Particle:
    name: str
    vec: np.ndarray = field(repr=False)

    def anti(self, name: str | None = None) -> "Particle":
        v = self.vec.copy()
        v[CONSERVED] = -v[CONSERVED]
        return Particle(name or f"~{self.name}", v)


# --- Quark builder ----------------------------------------------------------
QUARK_DATA = {
    # flavor: (Q, mass_GeV)  --  B = 1/3, no lepton numbers
    "u": (+2 / 3, 0.0022),
    "d": (-1 / 3, 0.0047),
    "c": (+2 / 3, 1.27),
    "s": (-1 / 3, 0.095),
    "t": (+2 / 3, 172.76),
    "b": (-1 / 3, 4.18),
}


def _quark(flavor: str, color_label: str) -> Particle:
    Q, mass = QUARK_DATA[flavor]
    color = {"r": COLOR_R, "g": COLOR_G, "b": COLOR_B}[color_label]
    return Particle(
        f"{flavor}_{color_label}",
        _v(Q, 1 / 3, 0, 0, 0, color, 0.5, mass),
    )


# --- Lepton builder ---------------------------------------------------------
LEPTON_DATA = {
    # name: (Q, family_index, mass_GeV)
    "e-":     (-1, 0, 0.000511),
    "nu_e":   ( 0, 0, 0.0),
    "mu-":    (-1, 1, 0.10566),
    "nu_mu":  ( 0, 1, 0.0),
    "tau-":   (-1, 2, 1.77686),
    "nu_tau": ( 0, 2, 0.0),
}


def _lepton(name: str) -> Particle:
    Q, fam, mass = LEPTON_DATA[name]
    fam_vec = [0, 0, 0]
    fam_vec[fam] = 1
    return Particle(name, _v(Q, 0, *fam_vec, COLORLESS, 0.5, mass))


# --- Gluon builder ----------------------------------------------------------
# Six off-diagonal gluons carry color-anticolor charge (a root of SU(3)).
# Two diagonal gluons live at the Cartan origin (the lambda_3, lambda_8 modes).
def _gluon_offdiag(c_from, c_to, label: str) -> Particle:
    c3 = c_from[0] - c_to[0]
    c8 = c_from[1] - c_to[1]
    return Particle(f"g_{label}", _v(0, 0, 0, 0, 0, (c3, c8), 1.0, 0.0))


def _gluon_diag(label: str) -> Particle:
    return Particle(f"g_{label}", _v(0, 0, 0, 0, 0, COLORLESS, 1.0, 0.0))


# --- Build the particle table ----------------------------------------------
def build_sm() -> dict[str, Particle]:
    p: dict[str, Particle] = {}

    # Quarks, three colors each
    for flavor in QUARK_DATA:
        for c in ("r", "g", "b"):
            q = _quark(flavor, c)
            p[q.name] = q
            anti = q.anti(f"~{flavor}_{c}")  # antiquark carries anticolor automatically
            p[anti.name] = anti

    # Leptons + antileptons
    for name in LEPTON_DATA:
        lep = _lepton(name)
        p[lep.name] = lep
        # Standard antilepton naming
        anti_name = {
            "e-": "e+", "mu-": "mu+", "tau-": "tau+",
            "nu_e": "~nu_e", "nu_mu": "~nu_mu", "nu_tau": "~nu_tau",
        }[name]
        p[anti_name] = lep.anti(anti_name)

    # Electroweak bosons (post-EWSB physical states)
    p["photon"] = Particle("photon", _v(0, 0, 0, 0, 0, COLORLESS, 1.0, 0.0))
    p["Z"]      = Particle("Z",      _v(0, 0, 0, 0, 0, COLORLESS, 1.0, 91.1876))
    p["W+"]     = Particle("W+",     _v(+1, 0, 0, 0, 0, COLORLESS, 1.0, 80.379))
    p["W-"]     = Particle("W-",     _v(-1, 0, 0, 0, 0, COLORLESS, 1.0, 80.379))
    p["H"]      = Particle("H",      _v(0, 0, 0, 0, 0, COLORLESS, 0.0, 125.10))

    # Gluons: six off-diagonal + two diagonal (Cartan)
    colors = [("r", COLOR_R), ("g", COLOR_G), ("b", COLOR_B)]
    for (a, ca), (b, cb) in [(x, y) for x in colors for y in colors if x[0] != y[0]]:
        p[f"g_{a}{b}bar"] = _gluon_offdiag(ca, cb, f"{a}{b}bar")
    p["g_diag3"] = _gluon_diag("diag3")  # ~ lambda_3 mode, (rr̄ - gḡ)/√2
    p["g_diag8"] = _gluon_diag("diag8")  # ~ lambda_8 mode, (rr̄ + gḡ - 2bb̄)/√6

    return p


# --- Vertex closure check ---------------------------------------------------
def vertex_residual(particles: dict[str, Particle], incoming, outgoing) -> np.ndarray:
    """Return sum(in) - sum(out) on the conserved-charge subspace.

    A valid SM gauge vertex has residual == 0 on every coordinate.
    """
    inc = sum(particles[n].vec[CONSERVED] for n in incoming)
    out = sum(particles[n].vec[CONSERVED] for n in outgoing)
    return inc - out


def is_allowed(particles, incoming, outgoing, tol=1e-12) -> bool:
    return bool(np.all(np.abs(vertex_residual(particles, incoming, outgoing)) < tol))


def kinematic_threshold(particles, parent, daughters):
    """For a 1->N decay at rest, check m_parent >= sum(m_daughters).

    Returns (m_parent, sum_m_daughters, allowed). Off-shell intermediates can
    of course violate this — but for an on-shell decay this is necessary.
    """
    m_p = float(particles[parent].vec[8])
    m_d = float(sum(particles[d].vec[8] for d in daughters))
    return m_p, m_d, m_p >= m_d


def solve_missing(particles, incoming, outgoing, missing_side="out", top_k=3):
    """Given a vertex with one leg unspecified, compute the required charge
    vector and return the top_k closest known particles.

    `missing_side` is "in" or "out" — which side the unknown leg sits on.
    """
    inc = sum(particles[n].vec[CONSERVED] for n in incoming)
    out = sum(particles[n].vec[CONSERVED] for n in outgoing)
    needed = (inc - out) if missing_side == "out" else (out - inc)
    scored = sorted(
        particles.items(),
        key=lambda kv: float(np.linalg.norm(kv[1].vec[CONSERVED] - needed)),
    )
    return needed, scored[:top_k]


# --- Demo / smoke tests ----------------------------------------------------
if __name__ == "__main__":
    sm = build_sm()
    print(f"Built {len(sm)} particle states in R^{DIM}.")
    print(f"Conserved-charge subspace: coords {list(range(7))} = {COORDS[:7]}\n")

    cases = [
        # Description, incoming, outgoing, expected
        ("beta- decay  d -> u + e- + ~nu_e",
            ["d_r"], ["u_r", "e-", "~nu_e"], True),
        ("muon decay   mu- -> e- + ~nu_e + nu_mu",
            ["mu-"], ["e-", "~nu_e", "nu_mu"], True),
        ("QED vertex   e- -> e- + photon",
            ["e-"], ["e-", "photon"], True),
        ("QCD vertex   u_r -> u_g + g_rgbar",
            ["u_r"], ["u_g", "g_rgbar"], True),
        ("Z decay      Z -> e- + e+",
            ["Z"], ["e-", "e+"], True),
        ("Higgs decay  H -> b_r + ~b_r",
            ["H"], ["b_r", "~b_r"], True),
        ("FORBIDDEN    mu- -> e- + photon (LFV)",
            ["mu-"], ["e-", "photon"], False),
        ("FORBIDDEN    proton decay  u_r + u_g + d_b -> e+ + photon",
            ["u_r", "u_g", "d_b"], ["e+", "photon"], False),
    ]

    for desc, inc, out, expected in cases:
        r = vertex_residual(sm, inc, out)
        ok = bool(np.all(np.abs(r) < 1e-12))
        mark = "OK " if ok == expected else "!! "
        print(f"{mark}{desc}")
        if not ok:
            nz = [(COORDS[i], r[i]) for i in range(7) if abs(r[i]) > 1e-12]
            print(f"     non-conserved: {nz}")
