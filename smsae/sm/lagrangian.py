"""The Standard Model Lagrangian as an enumerated list of tensor contractions.

L_SM = L_gauge + L_fermion_kin + L_gauge_coupling + L_Higgs + L_Yukawa

Every term is one tensor contraction.  This module makes the structure explicit:
each term is a `LTerm` record carrying
  - a symbolic coefficient (e.g. "-1/4", "g_s", "-y^u_{ij}")
  - the field operators it contracts
  - the gauge-tensor factors involved (T^a of color/weak, hypercharge Y, gamma^mu, ...)
  - a human-readable index contraction
  - a category tag

Together with the SM field table (sm_lagrangian.SM_FIELDS), this is essentially
the *entire data of the SM Lagrangian* in ~25 records.

Independent real parameters:  19  (minimal SM, no neutrino masses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# --- Field declarations (pre-EWSB) -----------------------------------------
Chirality = Literal["L", "R", "-"]
Lorentz   = Literal["scalar", "spinor", "vector"]


@dataclass(frozen=True)
class Field:
    name: str
    color_rep: str        # SU(3) irrep label: '1', '3', '8 (adjoint)'
    n_color: int          # dim of color rep
    weak_rep: str         # SU(2) irrep label: '1', '2', '3 (adjoint)'
    n_weak: int           # dim of weak rep
    Y: float              # U(1)_Y hypercharge  (so Q = T3 + Y/2)
    n_gen: int            # number of generations (3 for SM fermions, 1 for bosons/Higgs)
    chirality: Chirality
    lorentz: Lorentz

    def Q_range(self) -> list[float]:
        """Electric charges of the isospin components, via Q = T3 + Y/2."""
        if self.n_weak == 1:
            return [self.Y / 2]
        elif self.n_weak == 2:
            return [+0.5 + self.Y / 2, -0.5 + self.Y / 2]
        elif self.n_weak == 3:
            return [+1 + self.Y / 2, 0 + self.Y / 2, -1 + self.Y / 2]
        else:
            raise ValueError(f"weak rep dim {self.n_weak} not supported")


# SM field content (pre-EWSB)
Q_L = Field("Q_L", "3",  3, "2", 2,  1/3, 3, "L", "spinor")
u_R = Field("u_R", "3",  3, "1", 1,  4/3, 3, "R", "spinor")
d_R = Field("d_R", "3",  3, "1", 1, -2/3, 3, "R", "spinor")
L_L = Field("L_L", "1",  1, "2", 2, -1.0, 3, "L", "spinor")
e_R = Field("e_R", "1",  1, "1", 1, -2.0, 3, "R", "spinor")
H   = Field("H",   "1",  1, "2", 2, +1.0, 1, "-", "scalar")
G   = Field("G",   "8",  8, "1", 1,  0.0, 1, "-", "vector")  # gluon
W   = Field("W",   "1",  1, "3", 3,  0.0, 1, "-", "vector")  # SU(2)_L gauge boson
B   = Field("B",   "1",  1, "1", 1,  0.0, 1, "-", "vector")  # U(1)_Y gauge boson

SM_FIELDS = [Q_L, u_R, d_R, L_L, e_R, H, G, W, B]
FERMIONS  = [Q_L, u_R, d_R, L_L, e_R]
GAUGE_BOSONS = [G, W, B]


# --- Lagrangian term records ----------------------------------------------
@dataclass(frozen=True)
class LTerm:
    name: str
    coefficient: str
    factors: tuple[str, ...]       # field operators (with conjugate marks: "Q_L^dag")
    tensors: tuple[str, ...]       # gauge-tensor / metric factors used in the contraction
    contraction: str               # human-readable index spec
    category: str
    notes: str = ""


def _conj(field_name: str) -> str:
    return f"{field_name}^dag"


# --- L_gauge:  -1/4 F^{a, mu nu} F^a_{mu nu}  for each group ---------------
L_GAUGE = [
    LTerm("L_G", "-1/4",
          factors=(G.name, G.name),
          tensors=("eta^{mu rho} eta^{nu sigma}", "delta^{ab}"),
          contraction="G^a_{mu nu} G^{a, mu nu};  G^a_{mu nu} = d_mu G^a_nu - d_nu G^a_mu + g_s f^{abc}_{SU3} G^b_mu G^c_nu",
          category="gauge_kinetic",
          notes="3g and 4g vertices hidden in F^2 via f^{abc} of SU(3)"),
    LTerm("L_W", "-1/4",
          factors=(W.name, W.name),
          tensors=("eta^{mu rho} eta^{nu sigma}", "delta^{ab}"),
          contraction="W^a_{mu nu} W^{a, mu nu};  W^a_{mu nu} = d_mu W^a_nu - d_nu W^a_mu + g epsilon^{abc} W^b_mu W^c_nu",
          category="gauge_kinetic",
          notes="WWZ, WW-gamma, WWWW vertices hidden in F^2 via eps^{abc} of SU(2)"),
    LTerm("L_B", "-1/4",
          factors=(B.name, B.name),
          tensors=("eta^{mu rho} eta^{nu sigma}",),
          contraction="B_{mu nu} B^{mu nu};  B_{mu nu} = d_mu B_nu - d_nu B_mu",
          category="gauge_kinetic",
          notes="abelian, no self-coupling"),
]


# --- L_fermion_kin:  i psi-bar gamma^mu d_mu psi --------------------------
def _fermion_kinetic_terms():
    out = []
    for psi in FERMIONS:
        out.append(LTerm(
            name=f"kin_{psi.name}",
            coefficient="i",
            factors=(_conj(psi.name), psi.name),
            tensors=("gamma^mu",),
            contraction=f"{psi.name}^dag gamma^mu d_mu {psi.name}",
            category="fermion_kinetic",
            notes=f"3 generations, color rep {psi.color_rep}, weak rep {psi.weak_rep}",
        ))
    return out


# --- L_gauge_coupling:  expand i psi-bar gamma^mu D_mu psi (gauge pieces) -
def _gauge_coupling_terms():
    out = []
    for psi in FERMIONS:
        # SU(3): color-triplet fermions couple to gluon
        if psi.n_color > 1:
            out.append(LTerm(
                name=f"g_{psi.name}_G",
                coefficient="g_s",
                factors=(_conj(psi.name), psi.name, G.name),
                tensors=("gamma^mu", "T^a (SU3 fund)"),
                contraction=f"{psi.name}^dag gamma^mu (T^a_C) {psi.name} G^a_mu",
                category="gauge_coupling",
                notes=f"strong coupling to {psi.name} (color {psi.color_rep})",
            ))
        # SU(2): weak-doublet fermions couple to W
        if psi.n_weak > 1:
            out.append(LTerm(
                name=f"g_{psi.name}_W",
                coefficient="g",
                factors=(_conj(psi.name), psi.name, W.name),
                tensors=("gamma^mu", "tau^a / 2 (SU2 fund)"),
                contraction=f"{psi.name}^dag gamma^mu (tau^a/2) {psi.name} W^a_mu",
                category="gauge_coupling",
                notes="charged-current + neutral-current pieces both live here",
            ))
        # U(1)_Y: any nonzero-Y fermion couples to B
        if psi.Y != 0:
            out.append(LTerm(
                name=f"g_{psi.name}_B",
                coefficient=f"g' * ({psi.Y:+g}/2)",
                factors=(_conj(psi.name), psi.name, B.name),
                tensors=("gamma^mu",),
                contraction=f"{psi.name}^dag gamma^mu {psi.name} B_mu",
                category="gauge_coupling",
                notes=f"hypercharge Y = {psi.Y:+g}",
            ))
    return out


# --- L_Higgs:  (D_mu H)^dag (D^mu H) - mu^2 H^dag H - lambda (H^dag H)^2 --
L_HIGGS = [
    LTerm("kin_H", "1",
          factors=(_conj(H.name), H.name),
          tensors=("eta_{mu nu}",),
          contraction="(D_mu H)^dag (D^mu H)",
          category="higgs_kinetic",
          notes="expands into H kinetic + HW + HB + HWW + HBB + HWB; after EWSB gives M_W, M_Z, HVV vertices"),
    LTerm("mass_H", "-mu^2",
          factors=(_conj(H.name), H.name),
          tensors=(),
          contraction="H^dag H",
          category="higgs_potential",
          notes="mu^2 < 0 triggers EWSB; v = sqrt(-mu^2 / lambda)"),
    LTerm("quartic_H", "-lambda",
          factors=(_conj(H.name), _conj(H.name), H.name, H.name),
          tensors=(),
          contraction="(H^dag H)^2",
          category="higgs_potential",
          notes="lambda > 0 for vacuum stability; lambda = m_H^2 / (2 v^2)"),
]


# --- L_Yukawa:  -y^f_ij  fermion-bar  H  fermion  + h.c. -------------------
L_YUKAWA = [
    LTerm("Y_u", "-y^u_{ij}",
          factors=(_conj(Q_L.name), "H~", u_R.name),
          tensors=("epsilon^{alpha beta}",),
          contraction="Q_L^{i,dag} H~ u_R^j  + h.c.   (H~ = i sigma_2 H^*)",
          category="yukawa",
          notes="3x3 complex matrix y^u; after EWSB gives up-type quark masses + CKM"),
    LTerm("Y_d", "-y^d_{ij}",
          factors=(_conj(Q_L.name), H.name, d_R.name),
          tensors=(),
          contraction="Q_L^{i,dag} H d_R^j  + h.c.",
          category="yukawa",
          notes="3x3 complex matrix y^d; gives down-type quark masses"),
    LTerm("Y_e", "-y^e_{ij}",
          factors=(_conj(L_L.name), H.name, e_R.name),
          tensors=(),
          contraction="L_L^{i,dag} H e_R^j  + h.c.",
          category="yukawa",
          notes="3x3 complex matrix y^e (diagonal in minimal SM by lepton-flavor rotation); gives charged-lepton masses"),
]


# --- Optional CP-violating QCD theta term (set to 0 by observation) -------
L_THETA_QCD = [
    LTerm("theta_QCD", "theta_QCD * g_s^2 / (32 pi^2)",
          factors=(G.name, G.name),
          tensors=("epsilon^{mu nu rho sigma}", "delta^{ab}"),
          contraction="epsilon^{mu nu rho sigma} G^a_{mu nu} G^a_{rho sigma}",
          category="theta_term",
          notes="total derivative classically; observed bound |theta_QCD| < 1e-10 (strong CP problem)"),
]


SM_LAGRANGIAN: list[LTerm] = (
    L_GAUGE
    + _fermion_kinetic_terms()
    + _gauge_coupling_terms()
    + L_HIGGS
    + L_YUKAWA
    + L_THETA_QCD
)


# --- Parameter count -------------------------------------------------------
PARAMETERS = {
    "gauge couplings (g_s, g, g')":              3,
    "Higgs sector (mu^2, lambda)":               2,
    "quark masses (u, d, s, c, b, t)":           6,
    "CKM (3 angles + 1 CP phase)":               4,
    "charged-lepton masses (e, mu, tau)":        3,
    "theta_QCD (=0 within bounds)":              1,
}
N_PARAMETERS_MINIMAL = sum(PARAMETERS.values())  # 19

PARAMETERS_NU = {
    "neutrino masses (3)":                       3,
    "PMNS (3 angles + 1 Dirac phase)":           4,
    # +2 Majorana phases if neutrinos are Majorana
}
N_PARAMETERS_WITH_DIRAC_NU = N_PARAMETERS_MINIMAL + sum(PARAMETERS_NU.values())  # 26


# --- Pretty-printer --------------------------------------------------------
def print_lagrangian(show_notes: bool = True) -> None:
    print("L_SM  =  " + "  +  ".join(t.name for t in SM_LAGRANGIAN))
    print()

    # Group by category
    by_cat: dict[str, list[LTerm]] = {}
    for t in SM_LAGRANGIAN:
        by_cat.setdefault(t.category, []).append(t)

    order = ["gauge_kinetic", "fermion_kinetic", "gauge_coupling",
             "higgs_kinetic", "higgs_potential", "yukawa", "theta_term"]
    for cat in order:
        if cat not in by_cat:
            continue
        terms = by_cat[cat]
        print(f"--- {cat:<20s}  ({len(terms)} term{'s' if len(terms) != 1 else ''}) ---")
        for t in terms:
            print(f"  {t.name:<14s} {t.coefficient:>22s}  *  {t.contraction}")
            if show_notes and t.notes:
                print(f"  {'':14s} {'':22s}     ({t.notes})")
        print()


def print_field_table() -> None:
    print(f"{'field':<5s} {'color':>5s} {'weak':>5s} {'Y':>8s} {'Q':<14s} {'#gen':>5s}  chirality  lorentz")
    print("  " + "-" * 70)
    for fld in SM_FIELDS:
        qs = ", ".join(f"{q:+.3f}" for q in fld.Q_range())
        print(f"  {fld.name:<5s} {fld.color_rep:>5s} {fld.weak_rep:>5s} {fld.Y:>+8.3f} "
              f"{qs:<14s} {fld.n_gen:>5d}  {fld.chirality:<9s}  {fld.lorentz}")


def print_parameter_count() -> None:
    print(f"{'count':>5s}  parameter")
    print("  " + "-" * 50)
    for label, n in PARAMETERS.items():
        print(f"{n:>5d}  {label}")
    print(f"  {'-' * 5}  {'-' * 30}")
    print(f"{N_PARAMETERS_MINIMAL:>5d}  TOTAL (minimal SM, no neutrino mass)")
    print()
    print("  Extensions for observed neutrino oscillations (Dirac masses):")
    for label, n in PARAMETERS_NU.items():
        print(f"{n:>5d}  {label}")
    print(f"{N_PARAMETERS_WITH_DIRAC_NU:>5d}  TOTAL (SM + Dirac neutrino masses)")


# --- Cross-checks ----------------------------------------------------------
def consistency_checks() -> list[tuple[str, bool, str]]:
    """A few internal consistency checks on the Lagrangian / field content."""
    out = []
    # Charge assignments via Q = T3 + Y/2 should reproduce the known SM charges.
    expected = {
        "Q_L": [+2/3, -1/3], "u_R": [+2/3], "d_R": [-1/3],
        "L_L": [0.0, -1.0],  "e_R": [-1.0],
        "H":   [+1.0, 0.0],
    }
    for fld in [Q_L, u_R, d_R, L_L, e_R, H]:
        got = fld.Q_range()
        want = expected[fld.name]
        ok = all(abs(g - w) < 1e-12 for g, w in zip(got, want))
        out.append((f"Q_range({fld.name})", ok, f"got {got}, expected {want}"))
    # Every fermion has chirality L or R
    for f in FERMIONS:
        out.append((f"chirality({f.name})", f.chirality in ("L", "R"), f.chirality))
    # Hypercharge sums for anomaly: already tested in sm_tests; smoke-check here too.
    # sum_{LH Weyl, per generation} Y = 0
    grav = (3 * 2 * Q_L.Y          # Q_L:  3 colors x 2 isospin
          + 3 * (-u_R.Y)           # u_R^c: 3 colors (conjugate flips Y sign)
          + 3 * (-d_R.Y)
          + 1 * 2 * L_L.Y
          + 1 * (-e_R.Y))
    out.append(("sum_Y over generation = 0", abs(grav) < 1e-12, f"got {grav}"))
    return out


if __name__ == "__main__":
    print("=" * 78)
    print("Standard Model field content")
    print("=" * 78)
    print_field_table()

    print()
    print("=" * 78)
    print("Standard Model Lagrangian (one tensor contraction per row)")
    print("=" * 78)
    print_lagrangian(show_notes=True)

    print(f"Total terms in L_SM:  {len(SM_LAGRANGIAN)}")
    print()

    print("=" * 78)
    print("Independent real parameters")
    print("=" * 78)
    print_parameter_count()

    print()
    print("=" * 78)
    print("Internal consistency checks")
    print("=" * 78)
    for label, ok, detail in consistency_checks():
        mark = "OK" if ok else "!!"
        print(f"  {mark}  {label:<35s}  {detail}")
