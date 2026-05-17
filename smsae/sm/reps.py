"""Higher representations of the SM gauge groups SU(2) and SU(3).

For each representation R we construct Hermitian generator matrices T^a_R
normalized so tr(T^a T^b) = T(R) delta^{ab}, then report:
  - dim(R)
  - Dynkin index T(R)         from tr(T^a T^b) = T(R) delta^{ab}
  - quadratic Casimir C_2(R)  from sum_a T^a T^a = C_2(R) * I

Constructions:
  - spin_rep(j):       SU(2) spin-j via J+, J-, Jz in the |j,m> basis.
  - sym_rep(T, p):     totally symmetric rank-p tensor power of any rep.
                       For SU(3) fundamental this gives (p,0):
                         p=1: 3,   p=2: 6,   p=3: 10,   p=4: 15'
  - conjugate_rep(T):  T_bar^a = -(T^a)^*. For SU(3) fundamental this gives 3-bar;
                       for SU(2) fundamental it's equivalent (pseudoreal).
  - adjoint_from_f(f): T^a_{bc} = -i f^{abc} (adjoint built from structure constants).

For each constructed rep we verify [T^a, T^b] = i f^{abc} T^c against the
structure constants of the parent algebra.
"""

from __future__ import annotations

from itertools import combinations_with_replacement, permutations
from math import sqrt

import numpy as np

import smsae.sm.groups as G


# --- SU(2) spin-j rep -------------------------------------------------------
def spin_rep(j: float) -> np.ndarray:
    """SU(2) generators in the spin-j irrep (dimension 2j+1).

    Returns array of shape (3, 2j+1, 2j+1): T^1, T^2, T^3.
    States are |j, m> for m = j, j-1, ..., -j; basis index k satisfies m = j - k.
    """
    dim = int(round(2 * j + 1))
    m_vals = np.array([j - k for k in range(dim)])
    Jz = np.diag(m_vals).astype(complex)
    Jp = np.zeros((dim, dim), dtype=complex)
    Jm = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        m = m_vals[k]
        # J+ |j,m> = sqrt((j-m)(j+m+1)) |j, m+1>  -- index k-1 since m increases as k decreases
        if m + 1 <= j:
            Jp[k - 1, k] = sqrt((j - m) * (j + m + 1))
        # J- |j,m> = sqrt((j+m)(j-m+1)) |j, m-1>  -- index k+1
        if m - 1 >= -j:
            Jm[k + 1, k] = sqrt((j + m) * (j - m + 1))
    Jx = (Jp + Jm) / 2
    Jy = (Jp - Jm) / (2j) if False else (Jp - Jm) / (2j)  # noqa: F841 -- keep clarity
    Jy = (Jp - Jm) / (2 * 1j)
    return np.stack([Jx, Jy, Jz])


# --- SU(N) symmetric tensor-product rep ------------------------------------
def sym_rep(T_parent: np.ndarray, p: int) -> np.ndarray:
    """Generators acting on the totally symmetric rank-p tensor power of T_parent.

    If T_parent are generators on V (dim n), this returns generators on Sym^p(V)
    of dimension C(n + p - 1, p).

    Construction: act on V^{⊗p} by the Leibniz rule (sum over slots), then
    project onto the symmetric subspace via an isometry P with orthonormal columns.
    """
    n_gen, n, _ = T_parent.shape
    if p == 1:
        return T_parent.copy()

    # Symmetric basis = sorted tuples (i_1 <= ... <= i_p)
    sym_tuples = list(combinations_with_replacement(range(n), p))
    d_sym = len(sym_tuples)

    # Build the isometry P: C^d_sym -> C^{n^p}.
    # Each symmetric basis vector is the normalized sum over distinct permutations
    # of the index tuple.
    P = np.zeros((n ** p, d_sym), dtype=complex)
    for col, tup in enumerate(sym_tuples):
        perms = set(permutations(tup))
        norm = sqrt(len(perms))
        for perm in perms:
            flat = 0
            for idx in perm:
                flat = flat * n + idx
            P[flat, col] = 1 / norm

    # Build T^a on V^{⊗p}: sum_slots I ⊗ ... ⊗ T^a (slot) ⊗ ... ⊗ I
    I = np.eye(n, dtype=complex)
    T_full = np.zeros((n_gen, n ** p, n ** p), dtype=complex)
    for a in range(n_gen):
        for slot in range(p):
            ops = [I] * p
            ops[slot] = T_parent[a]
            kron = ops[0]
            for op in ops[1:]:
                kron = np.kron(kron, op)
            T_full[a] += kron

    # Restrict to symmetric subspace.
    return np.einsum("ij,ajk,kl->ail", P.conj().T, T_full, P)


def conjugate_rep(T: np.ndarray) -> np.ndarray:
    """For SU(N): T_bar^a = -(T^a)^*. Hermitian preserved; trace zero preserved."""
    return -T.conj()


def adjoint_from_f(f: np.ndarray) -> np.ndarray:
    """Adjoint rep generators: (T^a_adj)_{bc} = -i f^{abc}.

    For SU(2) this is the spin-1 rep; for SU(3) this is the 8-dim octet.
    """
    n = f.shape[0]
    T = np.zeros((n, n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                T[a, b, c] = -1j * f[a, b, c]
    return T


# --- Rep diagnostics -------------------------------------------------------
def dim(T: np.ndarray) -> int:
    return T.shape[1]


def dynkin_index(T: np.ndarray) -> float:
    """T(R): tr(T^a T^b) = T(R) delta^{ab}. Reported as average diag value."""
    n_gen = T.shape[0]
    vals = [float(np.real(np.trace(T[a] @ T[a]))) for a in range(n_gen)]
    return float(np.mean(vals))


def quadratic_casimir(T: np.ndarray) -> float:
    """C_2(R): sum_a T^a T^a = C_2(R) * I."""
    if T.shape[1] == 0:
        return 0.0
    C = sum(T[a] @ T[a] for a in range(T.shape[0]))
    # Should be proportional to identity
    return float(np.real(C[0, 0]))


def verify_algebra(T: np.ndarray, f: np.ndarray) -> float:
    """Max |[T^a, T^b] - i f^{abc} T^c|, summed over (a, b)."""
    n_gen = T.shape[0]
    err = 0.0
    for a in range(n_gen):
        for b in range(n_gen):
            comm = T[a] @ T[b] - T[b] @ T[a]
            expected = 1j * sum(f[a, b, c] * T[c] for c in range(n_gen))
            err = max(err, float(np.max(np.abs(comm - expected))))
    return err


# --- Predefined SM-relevant reps -------------------------------------------
def all_su2_reps(spins=(0, 0.5, 1, 1.5, 2)) -> dict[str, np.ndarray]:
    out = {}
    for j in spins:
        label = f"j={j}" if int(2 * j) % 2 else f"j={int(j)}"
        if j == 0:
            out[label] = np.zeros((3, 1, 1), dtype=complex)
        else:
            out[label] = spin_rep(j)
    return out


def all_su3_reps_up_to(p_max: int = 3) -> dict[str, np.ndarray]:
    out = {"1 (singlet)": np.zeros((8, 1, 1), dtype=complex)}
    out["3 (fund)"]      = G.SU3_GEN.astype(complex)
    out["3bar"]          = conjugate_rep(G.SU3_GEN.astype(complex))
    for p in range(2, p_max + 1):
        out[f"{symbolic_dim(p, 0)} (sym^{p} fund, (p,0)={p},0)"] = sym_rep(G.SU3_GEN.astype(complex), p)
    out[f"8 (adjoint via f)"] = adjoint_from_f(G.SU3_F)
    return out


def symbolic_dim(p: int, q: int) -> int:
    """dim of SU(3) irrep with Dynkin labels (p, q)."""
    return (p + 1) * (q + 1) * (p + q + 2) // 2


# --- Demo ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("SU(2) representations")
    print("=" * 78)
    print(f"{'rep':<10s} {'dim':>5s} {'T(R)':>10s} {'C_2(R)':>10s} {'[T,T]=ifT error':>20s}")
    su2_f = G.SU2_F
    for label, T in all_su2_reps().items():
        d, tR, c2 = dim(T), dynkin_index(T), quadratic_casimir(T)
        err = verify_algebra(T, su2_f)
        print(f"{label:<10s} {d:>5d} {tR:>10.4f} {c2:>10.4f} {err:>20.2e}")

    print()
    print("=" * 78)
    print("SU(3) representations")
    print("=" * 78)
    print(f"{'rep':<40s} {'dim':>5s} {'T(R)':>10s} {'C_2(R)':>10s} {'algebra err':>15s}")
    su3_f = G.SU3_F
    for label, T in all_su3_reps_up_to(3).items():
        d, tR, c2 = dim(T), dynkin_index(T), quadratic_casimir(T)
        err = verify_algebra(T, su3_f)
        print(f"{label:<40s} {d:>5d} {tR:>10.4f} {c2:>10.4f} {err:>15.2e}")

    print()
    print("Expected:")
    print("  SU(2):  T(j) = j(j+1)(2j+1)/3,   C_2(j) = j(j+1)")
    print("  SU(3):  3:  T=1/2,    C_2=4/3")
    print("          6:  T=5/2,    C_2=10/3")
    print("          8:  T=3,      C_2=3")
    print("         10:  T=15/2,   C_2=6")
