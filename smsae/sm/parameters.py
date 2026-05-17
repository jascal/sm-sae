"""Measured numerical parameters of the Standard Model.

Together with the gauge group structure (sm_groups) and the field content
(sm_embeddings + sm_tests.GEN1_FIELDS), these inputs fully specify the SM
Lagrangian. Values are PDG 2024 central numbers where applicable.

The SM has 19 independent parameters (or 26-28 with neutrino masses/mixings,
depending on Majorana vs Dirac). They live in:
  - 3 gauge couplings (g_s, g, g')
  - 2 Higgs parameters (mu^2, lambda)  --  equivalently (v, m_H)
  - 9 fermion masses (or 9 Yukawa eigenvalues): u,c,t,d,s,b,e,mu,tau
  - 4 CKM parameters (3 angles + 1 CP phase)
  - 1 theta_QCD (set to 0 here; strong CP problem)
And with neutrino masses:
  - 3 neutrino masses
  - 4 PMNS parameters (3 angles + 1 Dirac phase; +2 Majorana phases if Majorana)
"""

from __future__ import annotations

import numpy as np

# --- Gauge sector (at M_Z) --------------------------------------------------
ALPHA_EM_MZ  = 1 / 127.951
ALPHA_S_MZ   = 0.1179
SIN2_THETA_W = 0.23122

E_CHARGE = float(np.sqrt(4 * np.pi * ALPHA_EM_MZ))
G_S      = float(np.sqrt(4 * np.pi * ALPHA_S_MZ))             # SU(3)_C
G_WEAK   = E_CHARGE / float(np.sqrt(SIN2_THETA_W))             # SU(2)_L  (g)
G_PRIME  = E_CHARGE / float(np.sqrt(1 - SIN2_THETA_W))         # U(1)_Y   (g')

# --- Higgs sector -----------------------------------------------------------
HIGGS_VEV    = 246.22       # GeV  (v = (sqrt(2) G_F)^(-1/2))
HIGGS_MASS   = 125.10       # GeV
HIGGS_LAMBDA = HIGGS_MASS ** 2 / (2 * HIGGS_VEV ** 2)
HIGGS_MU2    = -(HIGGS_MASS ** 2) / 2

# --- Boson masses (predicted, listed for cross-check) ----------------------
M_W = 80.379
M_Z = 91.1876

# --- Fermion masses (GeV; PDG central values) ------------------------------
QUARK_MASSES = {
    "u": 0.0022,  "c": 1.27,   "t": 172.76,
    "d": 0.0047,  "s": 0.095,  "b": 4.18,
}
LEPTON_MASSES = {
    "e":   0.000511,
    "mu":  0.10566,
    "tau": 1.77686,
}

# --- Neutrino mass-splittings (eV^2) ---------------------------------------
DM2_21 = 7.42e-5      # solar
DM2_31 = 2.515e-3     # atmospheric (normal ordering)


# --- Yukawa matrices in mass basis (diagonal) -------------------------------
# y_f = sqrt(2) * m_f / v
def yukawa(mass_GeV: float) -> float:
    return float(np.sqrt(2) * mass_GeV / HIGGS_VEV)


YUKAWA_UP   = np.diag([yukawa(QUARK_MASSES["u"]),
                       yukawa(QUARK_MASSES["c"]),
                       yukawa(QUARK_MASSES["t"])])
YUKAWA_DOWN = np.diag([yukawa(QUARK_MASSES["d"]),
                       yukawa(QUARK_MASSES["s"]),
                       yukawa(QUARK_MASSES["b"])])
YUKAWA_LEP  = np.diag([yukawa(LEPTON_MASSES["e"]),
                       yukawa(LEPTON_MASSES["mu"]),
                       yukawa(LEPTON_MASSES["tau"])])


# --- CKM matrix (Wolfenstein, lambda=0.22500, A=0.826, rho_bar=0.159, eta_bar=0.348) -
def ckm_wolfenstein(lam=0.22500, A=0.826, rho_bar=0.159, eta_bar=0.348) -> np.ndarray:
    """Wolfenstein parametrization of CKM to O(lambda^4). Returns 3x3 complex matrix."""
    s12 = lam
    s23 = A * lam ** 2
    # rho_bar + i eta_bar = (rho + i eta)(1 - lambda^2/2)
    # so (rho + i eta) ~ (rho_bar + i eta_bar) / (1 - lambda^2/2)
    z = (rho_bar + 1j * eta_bar) / (1 - lam ** 2 / 2)
    s13e = A * lam ** 3 * z  # V_ub* = s13 e^{-i delta}; here equals A lam^3 (rho - i eta) for one phase convention
    # Standard parametrization (PDG):
    c12, c23 = np.sqrt(1 - s12 ** 2), np.sqrt(1 - s23 ** 2)
    s13 = abs(s13e)
    c13 = np.sqrt(1 - s13 ** 2)
    delta = -np.angle(s13e) if s13 > 0 else 0.0
    e_id = np.exp(1j * delta)
    V = np.array([
        [ c12 * c13,                              s12 * c13,                              s13 * np.exp(-1j * delta)],
        [-s12 * c23 - c12 * s23 * s13 * e_id,     c12 * c23 - s12 * s23 * s13 * e_id,     s23 * c13],
        [ s12 * s23 - c12 * c23 * s13 * e_id,    -c12 * s23 - s12 * c23 * s13 * e_id,     c23 * c13],
    ], dtype=complex)
    return V


CKM = ckm_wolfenstein()

# --- PMNS matrix (NuFIT 5.3, normal ordering, best fit; Majorana phases = 0) -
# Mixing angles
THETA_12 = np.radians(33.41)
THETA_23 = np.radians(49.1)
THETA_13 = np.radians(8.54)
DELTA_CP_PMNS = np.radians(197)


def pmns(theta12, theta23, theta13, delta) -> np.ndarray:
    s12, c12 = np.sin(theta12), np.cos(theta12)
    s23, c23 = np.sin(theta23), np.cos(theta23)
    s13, c13 = np.sin(theta13), np.cos(theta13)
    e_id, e_mid = np.exp(1j * delta), np.exp(-1j * delta)
    U = np.array([
        [ c12 * c13,                  s12 * c13,                 s13 * e_mid],
        [-s12 * c23 - c12 * s13 * s23 * e_id,
          c12 * c23 - s12 * s13 * s23 * e_id,
          c13 * s23],
        [ s12 * s23 - c12 * s13 * c23 * e_id,
         -c12 * s23 - s12 * s13 * c23 * e_id,
          c13 * c23],
    ], dtype=complex)
    return U


PMNS = pmns(THETA_12, THETA_23, THETA_13, DELTA_CP_PMNS)


# --- Smoke / unitarity tests -----------------------------------------------
def unitarity_residual(M: np.ndarray) -> float:
    return float(np.max(np.abs(M @ M.conj().T - np.eye(M.shape[0]))))


if __name__ == "__main__":
    print("=== Gauge couplings at M_Z ===")
    print(f"  e (em)  = {E_CHARGE:.5f}    g (SU2) = {G_WEAK:.5f}")
    print(f"  g' (U1) = {G_PRIME:.5f}    g_s     = {G_S:.5f}")
    # Cross-check: M_W = g v / 2
    print(f"\n  Predicted M_W = g v / 2 = {G_WEAK * HIGGS_VEV / 2:.3f}  vs measured {M_W}")
    # M_Z = sqrt(g^2 + g'^2) v / 2
    mz_pred = float(np.sqrt(G_WEAK ** 2 + G_PRIME ** 2)) * HIGGS_VEV / 2
    print(f"  Predicted M_Z = sqrt(g^2+g'^2) v/2 = {mz_pred:.3f}  vs measured {M_Z}")

    print("\n=== Higgs sector ===")
    print(f"  v = {HIGGS_VEV} GeV,  m_H = {HIGGS_MASS} GeV")
    print(f"  lambda = {HIGGS_LAMBDA:.4f},  mu^2 = {HIGGS_MU2:.2f} GeV^2")

    print("\n=== Yukawa eigenvalues (mass-basis diagonal) ===")
    print(f"  y_u, y_c, y_t = {np.diag(YUKAWA_UP)}")
    print(f"  y_d, y_s, y_b = {np.diag(YUKAWA_DOWN)}")
    print(f"  y_e, y_mu, y_tau = {np.diag(YUKAWA_LEP)}")

    print("\n=== CKM unitarity ===")
    print(f"  max |V V^dagger - I| = {unitarity_residual(CKM):.2e}")
    print(f"  |V_us| = {abs(CKM[0,1]):.5f}   |V_cb| = {abs(CKM[1,2]):.5f}")
    print(f"  |V_ub| = {abs(CKM[0,2]):.5f}   Jarlskog J = "
          f"{float(np.imag(CKM[0,0] * CKM[1,1] * np.conj(CKM[0,1]) * np.conj(CKM[1,0]))):.3e}")

    print("\n=== PMNS unitarity ===")
    print(f"  max |U U^dagger - I| = {unitarity_residual(PMNS):.2e}")
