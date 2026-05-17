"""Serialize the full SM tensor data to a single file in two formats.

Writes:
  - sm_data.npz           (numpy native; supports complex + object/string arrays directly)
  - sm_data.safetensors   (HuggingFace; complex split into _re/_im, strings JSON-encoded in metadata)

Both contain the same information: particle embeddings, field reps, gauge
structure constants, generator matrices, Yukawa / CKM / PMNS, signed vertex
incidence, higher SU(2)/SU(3) reps, Lagrangian-term metadata.

Run:
    python3 sm_export.py            -> writes both, prints schema
    python3 sm_export.py --verify   -> writes + reloads both, runs sanity checks
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import smsae.sm.groups as G
import smsae.sm.parameters as P
import smsae.sm.reps as R
import smsae.sm.checks as T
from smsae.sm.embeddings import CONSERVED, COORDS, build_sm
from smsae.sm.lagrangian import SM_FIELDS, SM_LAGRANGIAN


# ============================================================================
# Build the bundle
# ============================================================================
def build_bundle() -> dict[str, np.ndarray]:
    """Return all SM data as a flat name -> ndarray dict.

    Arrays may be float, complex, or object (strings); both saver paths handle
    the conversions needed for their target format.
    """
    sm = build_sm()

    # -- Particle embeddings -------------------------------------------------
    particle_names = sorted(sm.keys())
    particle_vectors = np.array([sm[n].vec for n in particle_names], dtype=float)

    # -- Field table (pre-EWSB) ---------------------------------------------
    field_table = np.array([
        [f.n_color, f.n_weak, f.Y, f.n_gen,
         {"L": 1, "R": -1, "-": 0}[f.chirality],
         {"scalar": 0, "spinor": 1, "vector": 2}[f.lorentz]]
        for f in SM_FIELDS
    ], dtype=float)

    # -- Vertex catalog as signed incidence matrix --------------------------
    vertices = T.vertex_catalog()
    appearing = sorted({p for _, inc, out in vertices for p in inc + out})
    name_to_idx = {n: i for i, n in enumerate(appearing)}
    incidence = np.zeros((len(vertices), len(appearing)), dtype=float)
    for v, (_, inc, out) in enumerate(vertices):
        for p in inc:  incidence[v, name_to_idx[p]] += 1
        for p in out:  incidence[v, name_to_idx[p]] -= 1

    # -- Per-generation Weyl-spinor table (for anomaly tests) ---------------
    gen1 = np.array([[nc, ni, Y] for _, nc, ni, Y, _ in T.GEN1_FIELDS], dtype=float)

    # -- Higher representations ---------------------------------------------
    rep_library: dict[str, np.ndarray] = {}
    rep_index: list[tuple[str, str]] = []
    for j in (0, 0.5, 1, 1.5, 2):
        key = f"rep_su2_j{j}"
        T_R = R.spin_rep(j) if j > 0 else np.zeros((3, 1, 1), dtype=complex)
        rep_library[key] = T_R.astype(complex)
        rep_index.append(("SU(2)", f"j={j}"))
    for label, T_R in R.all_su3_reps_up_to(3).items():
        key = "rep_su3_" + (
            label.split()[0].replace("(", "").replace(")", "").replace(",", "_")
        )
        rep_library[key] = T_R.astype(complex)
        rep_index.append(("SU(3)", label))

    bundle: dict[str, np.ndarray] = {
        # --- Particle embeddings -------------------------------------------
        "particle_names":           np.array(particle_names, dtype=object),
        "particle_vectors":         particle_vectors,
        "coord_labels":             np.array(COORDS, dtype=object),
        "conserved_coord_slice":    np.array([CONSERVED.start, CONSERVED.stop], dtype=np.int64),

        # --- Pre-EWSB field table ------------------------------------------
        "field_names":              np.array([f.name for f in SM_FIELDS], dtype=object),
        "field_table":              field_table,
        "field_table_columns":      np.array(
            ["n_color", "n_weak", "Y", "n_gen", "chirality_enc", "lorentz_enc"],
            dtype=object),

        # --- Anomaly-cancellation table ------------------------------------
        "gen1_field_names":         np.array([f[0] for f in T.GEN1_FIELDS], dtype=object),
        "gen1_table":               gen1,

        # --- Gauge structure -----------------------------------------------
        "su2_f":                    G.SU2_F.astype(float),
        "su2_d":                    G.SU2_D.astype(float),
        "su2_generators_fund":      G.SU2_GEN.astype(complex),
        "su3_f":                    G.SU3_F.astype(float),
        "su3_d":                    G.SU3_D.astype(float),
        "su3_generators_fund":      G.SU3_GEN.astype(complex),

        # --- SM physical parameters ----------------------------------------
        "gauge_couplings":          np.array([P.G_S, P.G_WEAK, P.G_PRIME], dtype=float),
        "gauge_couplings_labels":   np.array(["g_s", "g", "g_prime"], dtype=object),
        "sin2_theta_W":             np.array([P.SIN2_THETA_W], dtype=float),
        "higgs_parameters":         np.array(
            [P.HIGGS_VEV, P.HIGGS_MASS, P.HIGGS_LAMBDA, P.HIGGS_MU2], dtype=float),
        "higgs_parameters_labels":  np.array(["v", "m_H", "lambda", "mu_squared"], dtype=object),
        "yukawa_up":                P.YUKAWA_UP.astype(complex),
        "yukawa_down":              P.YUKAWA_DOWN.astype(complex),
        "yukawa_lepton":            P.YUKAWA_LEP.astype(complex),
        "ckm":                      P.CKM.astype(complex),
        "pmns":                     P.PMNS.astype(complex),
        "fermion_mass_labels":      np.array(
            ["m_u", "m_c", "m_t", "m_d", "m_s", "m_b", "m_e", "m_mu", "m_tau"],
            dtype=object),
        "fermion_masses":           np.array([
            P.QUARK_MASSES["u"], P.QUARK_MASSES["c"], P.QUARK_MASSES["t"],
            P.QUARK_MASSES["d"], P.QUARK_MASSES["s"], P.QUARK_MASSES["b"],
            P.LEPTON_MASSES["e"], P.LEPTON_MASSES["mu"], P.LEPTON_MASSES["tau"],
        ], dtype=float),

        # --- Vertex catalog (signed incidence) -----------------------------
        "vertex_descriptions":      np.array([v[0] for v in vertices], dtype=object),
        "vertex_particle_names":    np.array(appearing, dtype=object),
        "vertex_incidence":         incidence,

        # --- Lagrangian metadata -------------------------------------------
        "lagrangian_term_names":         np.array([t.name        for t in SM_LAGRANGIAN], dtype=object),
        "lagrangian_term_coefficients":  np.array([t.coefficient for t in SM_LAGRANGIAN], dtype=object),
        "lagrangian_term_contractions":  np.array([t.contraction for t in SM_LAGRANGIAN], dtype=object),
        "lagrangian_term_categories":    np.array([t.category    for t in SM_LAGRANGIAN], dtype=object),

        # --- Representation library index ----------------------------------
        "rep_index_group":          np.array([g for g, _ in rep_index], dtype=object),
        "rep_index_label":          np.array([l for _, l in rep_index], dtype=object),
    }
    bundle.update(rep_library)
    return bundle


# ============================================================================
# Schema reporting
# ============================================================================
def report_schema(bundle: dict[str, np.ndarray]) -> None:
    print(f"{'name':<40s} {'shape':>18s} {'dtype':>14s} {'bytes':>10s}")
    print("-" * 86)
    total = 0
    for name in sorted(bundle):
        arr = bundle[name]
        if arr.dtype == object:
            nbytes = sum(len(str(x).encode()) for x in arr.flatten())
            dt = "object"
        else:
            nbytes = arr.nbytes
            dt = str(arr.dtype)
        total += nbytes
        print(f"{name:<40s} {str(arr.shape):>18s} {dt:>14s} {nbytes:>10d}")
    print("-" * 86)
    print(f"{'TOTAL (in-memory approx)':<40s} {'':>18s} {'':>14s} {total:>10d}")


# ============================================================================
# Saver: npz (native; complex + object both fine)
# ============================================================================
def save_npz(bundle: dict[str, np.ndarray], path: str) -> None:
    np.savez_compressed(path, **bundle)


def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


# ============================================================================
# Saver: safetensors (only numeric; split complex into _re/_im; strings -> metadata JSON)
# ============================================================================
def save_safetensors(bundle: dict[str, np.ndarray], path: str) -> None:
    from safetensors.numpy import save_file
    tensors: dict[str, np.ndarray] = {}
    metadata: dict[str, str] = {}
    for k, v in bundle.items():
        if v.dtype == object:
            # JSON-encode the list of strings into the metadata blob
            metadata[k] = json.dumps([str(x) for x in v.tolist()])
        elif np.iscomplexobj(v):
            tensors[k + "_re"] = np.ascontiguousarray(v.real.astype(np.float64))
            tensors[k + "_im"] = np.ascontiguousarray(v.imag.astype(np.float64))
        else:
            tensors[k] = np.ascontiguousarray(v)
    # Record which keys are complex so the loader can re-pair them.
    metadata["__complex_keys__"] = json.dumps(
        sorted({k for k in bundle if np.iscomplexobj(bundle[k])})
    )
    metadata["__schema_version__"] = "1"
    save_file(tensors, path, metadata=metadata)


def load_safetensors(path: str) -> dict[str, np.ndarray]:
    from safetensors import safe_open
    from safetensors.numpy import load_file
    raw = load_file(path)
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata() or {}

    complex_keys = set(json.loads(metadata.get("__complex_keys__", "[]")))
    out: dict[str, np.ndarray] = {}
    consumed = set()
    for k in complex_keys:
        re_arr = raw.get(k + "_re")
        im_arr = raw.get(k + "_im")
        if re_arr is not None and im_arr is not None:
            out[k] = re_arr + 1j * im_arr
            consumed.add(k + "_re")
            consumed.add(k + "_im")
    for k, v in raw.items():
        if k in consumed:
            continue
        out[k] = v
    for k, json_str in metadata.items():
        if k.startswith("__"):
            continue
        out[k] = np.array(json.loads(json_str), dtype=object)
    return out


# ============================================================================
# Verification: re-derive key quantities purely from a reloaded bundle
# ============================================================================
def verify_bundle(bundle: dict[str, np.ndarray], label: str) -> None:
    print(f"\n--- verifying {label} ---")

    # 1. Vertex closure: signed incidence times particle charges == 0
    names = list(bundle["vertex_particle_names"])
    all_names = list(bundle["particle_names"])
    all_vecs = bundle["particle_vectors"]
    cons = slice(int(bundle["conserved_coord_slice"][0]),
                 int(bundle["conserved_coord_slice"][1]))
    name_to_vec = {n: all_vecs[i] for i, n in enumerate(all_names)}
    Q = np.array([name_to_vec[n][cons] for n in names])
    B = bundle["vertex_incidence"]
    print(f"  B Q max residual:            {float(np.max(np.abs(B @ Q))):.2e}")

    # 2. nullity(B) should equal 7 (= number of conserved charges)
    rank = int(np.linalg.matrix_rank(B, tol=1e-9))
    print(f"  rank(B), nullity(B):         {rank}, {B.shape[1] - rank} (expect nullity = 7)")

    # 3. SU(3) Jacobi from stored f^{abc}
    f_ = bundle["su3_f"]
    jac = (np.einsum("ade,bcd->eabc", f_, f_)
         + np.einsum("bde,cad->eabc", f_, f_)
         + np.einsum("cde,abd->eabc", f_, f_))
    print(f"  SU(3) Jacobi residual:       {float(np.max(np.abs(jac))):.2e}")

    # 4. CKM unitarity
    ckm = bundle["ckm"]
    unit = float(np.max(np.abs(ckm @ ckm.conj().T - np.eye(3))))
    print(f"  CKM unitarity residual:      {unit:.2e}")

    # 5. SU(3) sextet Casimir / Dynkin from stored rep matrices
    T6 = bundle["rep_su3_6"]
    tR = float(np.real(np.trace(T6[0] @ T6[0])))
    C2 = float(np.real((sum(T6[a] @ T6[a] for a in range(8)))[0, 0]))
    print(f"  SU(3) 6:  T(R) = {tR:.4f} (expect 2.5),  C2 = {C2:.4f} (expect 3.333)")

    # 6. Anomaly cancellation
    y3 = grav = su2_u1 = su3_u1 = su3_3 = 0.0
    for name, (nc, ni, Y) in zip(bundle["gen1_field_names"], bundle["gen1_table"]):
        n_states = nc * ni
        y3   += n_states * Y ** 3
        grav += n_states * Y
        if ni == 2: su2_u1 += nc * Y * 0.5
        if nc == 3:
            su3_u1 += ni * Y * 0.5
            su3_3 += ni * (-1 if "^c" in str(name) else +1)
    print(f"  anomalies:  Y^3={y3:.1e}  grav={grav:.0f}  "
          f"SU2^2Y={su2_u1:.0f}  SU3^2Y={su3_u1:.0f}  SU3^3={su3_3:.0f}")

    # 7. y_t consistency: sqrt(2) m_t / v
    v = float(bundle["higgs_parameters"][0])
    m_t = float(bundle["fermion_masses"][2])
    y_t_pred = float(np.sqrt(2) * m_t / v)
    y_t_stored = float(np.real(bundle["yukawa_up"][2, 2]))
    print(f"  y_t stored: {y_t_stored:.4f}   sqrt(2) m_t / v: {y_t_pred:.4f}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    bundle = build_bundle()

    print("=" * 86)
    print("Standard Model tensor bundle")
    print("=" * 86)
    report_schema(bundle)

    npz_path = "sm_data.npz"
    st_path  = "sm_data.safetensors"
    save_npz(bundle, npz_path)
    save_safetensors(bundle, st_path)

    npz_size = os.path.getsize(npz_path)
    st_size  = os.path.getsize(st_path)
    print(f"\nWrote {npz_path}  ({npz_size:>9,} bytes, gzip-compressed)")
    print(f"Wrote {st_path}    ({st_size:>9,} bytes, uncompressed mmap-able)")

    if "--verify" in sys.argv:
        print("\n" + "=" * 86)
        print("Roundtrip verification")
        print("=" * 86)
        verify_bundle(load_npz(npz_path),         "sm_data.npz")
        verify_bundle(load_safetensors(st_path),  "sm_data.safetensors")
