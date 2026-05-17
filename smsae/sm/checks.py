"""Consistency tests on the Standard Model dense-vector encoding.

Tests performed:
  1. Gell-Mann-Nishijima:   Q = T_3 + Y/2 for every Weyl field.
  2. Anomaly cancellation:  Five gauge anomalies vanish per generation.
  3. Vertex catalog:        Every SM gauge / Yukawa vertex closes on conserved charges.
  4. Kinematic thresholds:  Selected decays satisfy m_parent >= sum(m_daughters).
  5. Embedding recovery:    Spectral embedding of the vertex graph spans the same
                            subspace as the true quantum-number vectors.
"""

from __future__ import annotations

import numpy as np

from smsae.sm.embeddings import (
    build_sm,
    vertex_residual,
    kinematic_threshold,
    solve_missing,
    CONSERVED,
    COORDS,
)


# --- (1) Pre-EWSB field table: left-handed Weyl spinors of one generation -
# Each entry: (label, n_color, n_isospin, Y, T3 list for each isospin component)
# All entries are LEFT-handed; right-handed singlets appear as charge conjugates.
GEN1_FIELDS = [
    # name,    n_color, n_isospin,  Y,    T3 per isospin component
    ("Q_L",    3,       2,           1/3,  [+0.5, -0.5]),  # (u_L, d_L)
    ("u_R^c",  3,       1,          -4/3,  [0.0]),         # antiquark of u_R
    ("d_R^c",  3,       1,           2/3,  [0.0]),         # antiquark of d_R
    ("L_L",    1,       2,          -1,    [+0.5, -0.5]),  # (nu_L, e_L)
    ("e_R^c",  1,       1,           2,    [0.0]),         # antilepton of e_R
]

# Expected Q for each component, from Q = T3 + Y/2
EXPECTED_Q = {
    ("Q_L", +0.5): +2/3, ("Q_L", -0.5): -1/3,
    ("u_R^c", 0.0): -2/3,
    ("d_R^c", 0.0): +1/3,
    ("L_L", +0.5):  0.0, ("L_L", -0.5): -1.0,
    ("e_R^c", 0.0): +1.0,
}


def gell_mann_nishijima_check() -> list[tuple[str, float, float, float, float, bool]]:
    """Verify Q = T_3 + Y/2 for each Weyl field component."""
    out = []
    for name, _, _, Y, t3_list in GEN1_FIELDS:
        for t3 in t3_list:
            Q_computed = t3 + Y / 2
            Q_expected = EXPECTED_Q[(name, t3)]
            ok = abs(Q_computed - Q_expected) < 1e-12
            out.append((name, t3, Y, Q_computed, Q_expected, ok))
    return out


# --- (2) Anomaly cancellation ----------------------------------------------
def anomaly_sums() -> dict[str, float]:
    """Compute the five SM gauge anomaly traces.  All should be zero per generation.

    Conventions: sum over LEFT-handed Weyl spinors.  T(fund SU(N)) = 1/2.
    SU(3) cubic anomaly counts fundamentals as +1, antifundamentals as -1.
    """
    y3 = grav = su2_u1 = su3_u1 = su3_3 = 0.0
    for name, nc, ni, Y, t3_list in GEN1_FIELDS:
        n_states = nc * ni
        y3   += n_states * Y ** 3                # [U(1)_Y]^3
        grav += n_states * Y                     # grav^2 * U(1)_Y
        if ni == 2:
            su2_u1 += nc * Y * 0.5               # [SU(2)]^2 * U(1)_Y, T(doublet) = 1/2
        if nc == 3:
            su3_u1 += ni * Y * 0.5               # [SU(3)]^2 * U(1)_Y
            sign = -1 if "^c" in name else +1
            su3_3 += ni * sign                   # [SU(3)]^3, fund=+1, antifund=-1
    return {
        "[U(1)_Y]^3        ": y3,
        "grav^2 * U(1)_Y   ": grav,
        "[SU(2)]^2 U(1)_Y  ": su2_u1,
        "[SU(3)]^2 U(1)_Y  ": su3_u1,
        "[SU(3)]^3         ": su3_3,
    }


# --- (3) Vertex catalog ----------------------------------------------------
QUARK_FLAVORS = ["u", "d", "c", "s", "t", "b"]
UP_FLAVORS    = ["u", "c", "t"]
DOWN_FLAVORS  = ["d", "s", "b"]
LEPTON_CHARGED = ["e", "mu", "tau"]
NEUTRINOS      = ["nu_e", "nu_mu", "nu_tau"]
COLORS        = ["r", "g", "b"]
OFF_DIAG_GLUONS = [  # (from_color, to_color, particle name)
    ("r", "g", "g_rgbar"), ("g", "b", "g_gbbar"), ("r", "b", "g_rbbar"),
    ("g", "r", "g_grbar"), ("b", "g", "g_bgbar"), ("b", "r", "g_brbar"),
]


def vertex_catalog() -> list[tuple[str, list[str], list[str]]]:
    """SM vertices (post-EWSB).  Programmatically generated to exhaustively
    cover every flavor x color combination of:
      - QED (charged-fermion + photon)
      - QCD (quark-gluon)
      - NC (Z + fermion pair)
      - Yukawa (H + fermion pair) -- diagonal in flavor
      - CC (W + lepton + neutrino,  W + up + anti-down with full 3x3 CKM mixing)
      - Triple/quartic gauge and Higgs self-couplings.
    """
    vs: list[tuple[str, list[str], list[str]]] = []

    # QED: every charged fermion couples to photon (both colors of quarks)
    for lep in LEPTON_CHARGED:
        vs.append((f"QED  {lep}- gamma",            [f"{lep}-"], [f"{lep}-", "photon"]))
        vs.append((f"QED  {lep}+ gamma",            [f"{lep}+"], [f"{lep}+", "photon"]))
    for f in QUARK_FLAVORS:
        for c in COLORS:
            vs.append((f"QED  {f}_{c} gamma",        [f"{f}_{c}"], [f"{f}_{c}", "photon"]))
            vs.append((f"QED  ~{f}_{c} gamma",       [f"~{f}_{c}"], [f"~{f}_{c}", "photon"]))

    # QCD: quark-gluon coupling for all flavors/colors via off-diagonal gluons
    for f in QUARK_FLAVORS:
        for ca, cb, g in OFF_DIAG_GLUONS:
            vs.append((f"QCD  {f}_{ca} -> {f}_{cb} {g}", [f"{f}_{ca}"], [f"{f}_{cb}", g]))
    # Triple-gluon (one example per cyclic orientation)
    vs.append(("QCD  3g  g_rgbar g_gbbar g_brbar", [], ["g_rgbar", "g_gbbar", "g_brbar"]))
    vs.append(("QCD  3g  g_grbar g_rbbar g_bgbar", [], ["g_grbar", "g_rbbar", "g_bgbar"]))

    # NC: Z couples to every fermion-antifermion pair (color-diagonal)
    for lep in LEPTON_CHARGED:
        vs.append((f"NC   Z -> {lep}- {lep}+",       ["Z"], [f"{lep}-", f"{lep}+"]))
    for nu in NEUTRINOS:
        vs.append((f"NC   Z -> {nu} ~{nu}",          ["Z"], [nu, f"~{nu}"]))
    for f in QUARK_FLAVORS:
        for c in COLORS:
            vs.append((f"NC   Z -> {f}_{c} ~{f}_{c}", ["Z"], [f"{f}_{c}", f"~{f}_{c}"]))

    # Yukawa: H -> ff (diagonal in mass basis, color-diagonal)
    for lep in LEPTON_CHARGED:
        vs.append((f"Yuk  H -> {lep}- {lep}+",       ["H"], [f"{lep}-", f"{lep}+"]))
    for f in QUARK_FLAVORS:
        for c in COLORS:
            vs.append((f"Yuk  H -> {f}_{c} ~{f}_{c}", ["H"], [f"{f}_{c}", f"~{f}_{c}"]))

    # CC leptonic: W+ -> l+ nu_l, W- -> l- ~nu_l   (lepton-flavor diagonal)
    for lep, nu in zip(LEPTON_CHARGED, NEUTRINOS):
        vs.append((f"CC   W+ -> {lep}+ {nu}",        ["W+"], [f"{lep}+", nu]))
        vs.append((f"CC   W- -> {lep}- ~{nu}",       ["W-"], [f"{lep}-", f"~{nu}"]))

    # CC quark: W+ -> u_i + ~d_j   (full 3x3 CKM mixing, color-diagonal)
    for u in UP_FLAVORS:
        for d in DOWN_FLAVORS:
            for c in COLORS:
                vs.append((f"CC   W+ -> {u}_{c} ~{d}_{c}", ["W+"], [f"{u}_{c}", f"~{d}_{c}"]))

    # Pure-bosonic vertices: triple/quartic gauge, Higgs self, Higgs-gauge.
    vs += [
        ("TGV   Z -> W+ W-",             ["Z"],         ["W+", "W-"]),
        ("TGV   gamma -> W+ W-",         ["photon"],    ["W+", "W-"]),
        ("QGV   W+ W- -> Z Z",           ["W+", "W-"],  ["Z", "Z"]),
        ("QGV   W+ W- -> gamma gamma",   ["W+", "W-"],  ["photon", "photon"]),
        ("QGV   W+ W- -> Z gamma",       ["W+", "W-"],  ["Z", "photon"]),
        ("HSC   H -> H H (off-shell)",   ["H"],         ["H", "H"]),
        ("HSC   H H -> H H",             ["H", "H"],    ["H", "H"]),
        ("HVV   H -> Z Z",               ["H"],         ["Z", "Z"]),
        ("HVV   H -> W+ W-",             ["H"],         ["W+", "W-"]),
        ("HVV   H H -> Z Z",             ["H", "H"],    ["Z", "Z"]),
    ]
    return vs


FORBIDDEN_VERTICES = [
    # Charged-lepton flavor violation
    ("FORB  mu- -> e- gamma  (LFV)",     ["mu-"],   ["e-", "photon"]),
    ("FORB  mu- -> e- e+ e-  (LFV)",     ["mu-"],   ["e-", "e+", "e-"]),
    # Baryon number violation
    ("FORB  proton-like  uud -> e+ photon",
        ["u_r", "u_g", "d_b"], ["e+", "photon"]),
    # Charge violation
    ("FORB  Z -> W+ W+",                 ["Z"],     ["W+", "W+"]),
    # Color violation: single colored particle in final state of a colorless decay
    ("FORB  H -> u_r d_g",               ["H"],     ["u_r", "d_g"]),
]


# --- (4) Kinematic thresholds ----------------------------------------------
KINEMATIC_CASES = [
    # (description, parent, daughters, kinematically_allowed_expected)
    ("muon decay      mu- -> e- ~nu_e nu_mu",          "mu-",    ["e-", "~nu_e", "nu_mu"],     True),
    ("tau decay       tau- -> mu- ~nu_mu nu_tau",      "tau-",   ["mu-", "~nu_mu", "nu_tau"],  True),
    ("Z -> e- e+",                                      "Z",      ["e-", "e+"],                  True),
    ("H -> b ~b",                                       "H",      ["b_r", "~b_r"],               True),
    ("H -> t ~t       (2 m_t > m_H, FORBIDDEN)",        "H",      ["t_r", "~t_r"],               False),
    ("W+ -> u ~d",                                      "W+",     ["u_r", "~d_r"],               True),
    ("photon -> e- e+ (m_gamma=0 < 2 m_e, FORBIDDEN)",  "photon", ["e-", "e+"],                  False),
]


# --- (5) Recovering the conservation algebra from the vertex list ---------
def signed_incidence_matrix(sm, vertices) -> tuple[list[str], np.ndarray]:
    """Build the signed vertex-particle incidence matrix.

    B[v, p] = (# times p appears as incoming) - (# times p appears as outgoing).

    A charge vector q (one entry per particle) is conserved at every listed
    vertex iff B @ q = 0. So the null space of B IS the algebra of conservation
    laws implied by this vertex set.
    """
    appearing: set[str] = set()
    for _, inc, out in vertices:
        appearing.update(inc); appearing.update(out)
    names = sorted(appearing)
    idx = {n: i for i, n in enumerate(names)}
    n, m = len(vertices), len(names)
    B = np.zeros((n, m))
    for v, (_, inc, out) in enumerate(vertices):
        for p in inc:
            B[v, idx[p]] += 1
        for p in out:
            B[v, idx[p]] -= 1
    return names, B


def conservation_algebra(sm, vertices, tol=1e-9) -> dict:
    """Compute the null space of the signed incidence matrix and check that
    the 7 true SM charges all lie inside it.

    Outputs:
      - rank, nullity of B
      - residual of projecting each true charge onto null(B)
      - any "extra" null directions beyond the 7 known charges, in particle-space

    If nullity == 7 and residuals are ~0, the vertex list exactly captures the
    SM conservation laws and nothing more. If nullity > 7 there are accidental
    symmetries of the vertex list (e.g. missing off-diagonal CKM vertices give
    per-flavor quark-number conservation).
    """
    names, B = signed_incidence_matrix(sm, vertices)
    # Null space via SVD
    U, s, Vt = np.linalg.svd(B, full_matrices=True)
    nullity = int(np.sum(s < tol)) + max(0, B.shape[1] - B.shape[0])
    # Robust nullity: cols - rank
    rank = int(np.sum(s > tol))
    nullity = B.shape[1] - rank
    null_basis = Vt[-nullity:].T if nullity > 0 else np.zeros((B.shape[1], 0))

    # True charges as a (n_particles x 7) matrix
    Q = np.array([sm[n].vec[CONSERVED] for n in names])
    # Project Q onto null(B); residual measures whether each true charge is conserved.
    proj = null_basis @ null_basis.T @ Q
    per_charge_residual = {
        COORDS[i]: float(np.linalg.norm(Q[:, i] - proj[:, i])
                          / max(np.linalg.norm(Q[:, i]), 1e-12))
        for i in range(Q.shape[1])
    }
    # Extra directions in null space orthogonal to span(true charges):
    # subtract the part of null_basis explained by Q, then describe what's left.
    extras: list[list[tuple[str, float]]] = []
    extra_dim = 0
    if null_basis.shape[1] > 0 and Q.shape[1] > 0:
        Q_orth, _ = np.linalg.qr(Q)
        residual_null = null_basis - Q_orth @ (Q_orth.T @ null_basis)
        U_e, s_e, _ = np.linalg.svd(residual_null, full_matrices=False)
        extra_dim = int(np.sum(s_e > 1e-6))
        for i in range(extra_dim):
            vec = U_e[:, i]
            # Largest-magnitude particle supports (these are the dominant carriers
            # of the unexplained conserved quantity).
            order = np.argsort(-np.abs(vec))
            supports = [(names[j], float(vec[j])) for j in order
                        if abs(vec[j]) > 1e-3][:6]
            extras.append(supports)

    return {
        "n_particles": B.shape[1],
        "n_vertices": B.shape[0],
        "rank_B": rank,
        "nullity_B": nullity,
        "expected_charges": Q.shape[1],
        "extra_symmetries": extra_dim,
        "per_charge_residual": per_charge_residual,
        "extra_directions": extras,
    }


# --- Runner ----------------------------------------------------------------
def _section(title):
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


if __name__ == "__main__":
    sm = build_sm()

    _section("(1) Gell-Mann-Nishijima:  Q = T_3 + Y/2")
    for name, t3, Y, Qc, Qe, ok in gell_mann_nishijima_check():
        mark = "OK" if ok else "!!"
        print(f"  {mark}  {name:<6s}  T3 = {t3:+.1f}  Y = {Y:+.3f}  =>  Q = {Qc:+.4f}  (expected {Qe:+.4f})")

    _section("(2) Anomaly cancellation (sum per generation; all should be 0)")
    for label, value in anomaly_sums().items():
        mark = "OK" if abs(value) < 1e-12 else "!!"
        print(f"  {mark}  {label}  =  {value:+.4e}")

    _section("(3) SM vertex catalog")
    by_type: dict[str, list[int]] = {}  # prefix -> [n_ok, n_total]
    n_ok = n_total = 0
    failures: list[tuple[str, list]] = []
    for desc, inc, out in vertex_catalog():
        r = vertex_residual(sm, inc, out)
        ok = bool(np.all(np.abs(r) < 1e-12))
        n_total += 1; n_ok += int(ok)
        prefix = desc.split()[0]  # QED/QCD/CC/NC/Yuk/TGV/QGV/HSC/HVV
        slot = by_type.setdefault(prefix, [0, 0])
        slot[1] += 1; slot[0] += int(ok)
        if not ok:
            nz = [(COORDS[i], float(r[i])) for i in range(7) if abs(r[i]) > 1e-12]
            failures.append((desc, nz))
    for prefix in ["QED", "QCD", "NC", "Yuk", "CC", "TGV", "QGV", "HSC", "HVV"]:
        if prefix in by_type:
            ok_n, tot_n = by_type[prefix]
            print(f"  {prefix:<5s}  {ok_n}/{tot_n} closed")
    print(f"  ----   ------")
    print(f"  total  {n_ok}/{n_total} closed")
    for desc, nz in failures:
        print(f"  !!  {desc}\n        non-conserved: {nz}")

    print("\n  Forbidden (should fail):")
    for desc, inc, out in FORBIDDEN_VERTICES:
        r = vertex_residual(sm, inc, out)
        closed = bool(np.all(np.abs(r) < 1e-12))
        mark = "OK" if not closed else "!!"
        nz = [(COORDS[i], float(r[i])) for i in range(7) if abs(r[i]) > 1e-12]
        print(f"  {mark}  {desc}")
        print(f"        violates: {nz}")

    _section("(4) Kinematic thresholds (m_parent >= sum m_daughters)")
    for desc, parent, daughters, expected in KINEMATIC_CASES:
        mp, md, allowed = kinematic_threshold(sm, parent, daughters)
        mark = "OK" if allowed == expected else "!!"
        verdict = "allowed" if allowed else "kinematically forbidden"
        print(f"  {mark}  {desc:<55s}  m_p={mp:9.4f}  sum_m_d={md:9.4f}  [{verdict}]")

    _section("(5) 'Solve for missing particle'")
    cases = [
        ("e- + e+ -> Z + ?",            ["e-", "e+"], ["Z"], "out"),
        ("W+ -> e+ + ?",                ["W+"],       ["e+"], "out"),
        ("? -> u_r + ~d_r",             [],            ["u_r", "~d_r"], "in"),
    ]
    for desc, inc, out, side in cases:
        needed, top = solve_missing(sm, inc, out, missing_side=side, top_k=3)
        print(f"\n  {desc}")
        print(f"    required charges: {dict(zip(COORDS[:7], np.round(needed, 4)))}")
        for name, particle in top:
            dist = float(np.linalg.norm(particle.vec[CONSERVED] - needed))
            print(f"    candidate: {name:<14s}  dist = {dist:.4e}")

    _section("(6) Recovering the conservation algebra from the vertex list")
    all_vertices = vertex_catalog()
    result = conservation_algebra(sm, all_vertices)
    print(f"  Signed incidence matrix B:  {result['n_vertices']} vertices x {result['n_particles']} particles")
    print(f"  rank(B) = {result['rank_B']},   nullity(B) = {result['nullity_B']}")
    print(f"  expected conserved charges = {result['expected_charges']}")
    print(f"  extra accidental symmetries = {result['extra_symmetries']}")
    print(f"  (extra > 0 means the vertex list under-determines the SM "
          f"-- e.g. missing CKM mixings)")
    print(f"\n  Each true charge q  satisfies  B q = 0  iff residual ~ 0:")
    for coord, r in result["per_charge_residual"].items():
        mark = "OK" if r < 1e-9 else "!!"
        print(f"    {mark}  {coord:<6s}  ||q - proj_null(q)|| / ||q|| = {r:.3e}")
    if result["extra_directions"]:
        print(f"\n  Accidental symmetries (top particle supports for each extra null direction):")
        for i, supports in enumerate(result["extra_directions"]):
            terms = "  ".join(f"{name}({val:+.2f})" for name, val in supports)
            print(f"    extra #{i+1}:  {terms}")
