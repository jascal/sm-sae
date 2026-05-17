"""Lie algebra structure of the SM gauge group SU(3)_C x SU(2)_L x U(1)_Y.

Provides:
- Generator matrices in the fundamental representation (Pauli/2, Gell-Mann/2).
- Structure constants f^{abc} from [T^a, T^b] = i f^{abc} T^c.
- Symmetric d^{abc} from {T^a, T^b} = (1/N) delta^{ab} I + d^{abc} T^c.
- Jacobi identity check.
- Casimir / index values.

These are the rank-3 tensors that fully specify the non-abelian vertex factors
in the SM Lagrangian (gauge boson self-interactions and gauge-fermion couplings
are entirely fixed once these and the field representations are known).
"""

from __future__ import annotations

import numpy as np

# --- Pauli matrices and SU(2) ----------------------------------------------
PAULI = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]],
], dtype=complex)

SU2_GEN = PAULI / 2

# --- Gell-Mann matrices and SU(3) ------------------------------------------
GELL_MANN = np.zeros((8, 3, 3), dtype=complex)
GELL_MANN[0] = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
GELL_MANN[1] = [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]
GELL_MANN[2] = [[1, 0, 0], [0, -1, 0], [0, 0, 0]]
GELL_MANN[3] = [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
GELL_MANN[4] = [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]
GELL_MANN[5] = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
GELL_MANN[6] = [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]
GELL_MANN[7] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3)

SU3_GEN = GELL_MANN / 2


def structure_constants(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute f^{abc} and d^{abc} from generators T^a of SU(N).

    Conventions: T^a Hermitian, traceless, normalized so tr(T^a T^b) = (1/2) delta^{ab}.
    Then  [T^a, T^b] = i f^{abc} T^c   =>   f^{abc} = -2i tr([T^a,T^b] T^c)
    and   {T^a, T^b} = (1/N) delta^{ab} I + d^{abc} T^c   =>   d^{abc} = 2 tr({T^a,T^b} T^c)
    """
    n = T.shape[0]
    f = np.zeros((n, n, n), dtype=float)
    d = np.zeros((n, n, n), dtype=float)
    for a in range(n):
        for b in range(n):
            comm = T[a] @ T[b] - T[b] @ T[a]
            anti = T[a] @ T[b] + T[b] @ T[a]
            for c in range(n):
                f[a, b, c] = float(np.real(-2j * np.trace(comm @ T[c])))
                d[a, b, c] = float(np.real(2 * np.trace(anti @ T[c])))
    return f, d


SU2_F, SU2_D = structure_constants(SU2_GEN)
SU3_F, SU3_D = structure_constants(SU3_GEN)


def jacobi_residual(f: np.ndarray) -> float:
    """Max |f^{ade} f^{bcd} + f^{bde} f^{cad} + f^{cde} f^{abd}|.

    The Jacobi identity for a Lie algebra is equivalent to this tensor identity.
    """
    r = (
        np.einsum("ade,bcd->eabc", f, f)
        + np.einsum("bde,cad->eabc", f, f)
        + np.einsum("cde,abd->eabc", f, f)
    )
    return float(np.max(np.abs(r)))


def quadratic_casimir(T: np.ndarray) -> np.ndarray:
    """C2(R) for the representation: sum_a T^a T^a."""
    return sum(T[a] @ T[a] for a in range(T.shape[0]))


def dynkin_index(T: np.ndarray) -> float:
    """T(R): tr(T^a T^b) = T(R) delta^{ab}.  For fundamental of SU(N) this is 1/2."""
    return float(np.real(np.trace(T[0] @ T[0])))


# --- U(1)_Y: trivial structure, charges live per-field --------------------
# All hypercharge assignments are in sm_tests.GEN1_FIELDS (per-generation table).


if __name__ == "__main__":
    print("=== SU(2) ===")
    print(f"  Jacobi residual: {jacobi_residual(SU2_F):.2e}")
    print(f"  Dynkin index T(fund): {dynkin_index(SU2_GEN):.4f}   (expected 0.5)")
    C2 = quadratic_casimir(SU2_GEN)
    print(f"  C2(fund): {np.real(C2[0,0]):.4f}                   (expected 0.75)")
    print(f"  f^{{012}} = {SU2_F[0,1,2]:+.3f}                       (expected +1)")

    print("\n=== SU(3) ===")
    print(f"  Jacobi residual: {jacobi_residual(SU3_F):.2e}")
    print(f"  Dynkin index T(fund): {dynkin_index(SU3_GEN):.4f}   (expected 0.5)")
    C2 = quadratic_casimir(SU3_GEN)
    print(f"  C2(fund): {np.real(C2[0,0]):.4f}                   (expected 4/3 = 1.333)")
    # A few well-known SU(3) structure constants
    print(f"  f^{{123}} = {SU3_F[0,1,2]:+.3f}                       (expected +1)")
    print(f"  f^{{458}} = {SU3_F[3,4,7]:+.3f}                       (expected +sqrt(3)/2 = +0.866)")
    print(f"  d^{{118}} = {SU3_D[0,0,7]:+.3f}                       (expected +1/sqrt(3) = +0.577)")
    print(f"  d^{{888}} = {SU3_D[7,7,7]:+.3f}                       (expected -1/sqrt(3) = -0.577)")
