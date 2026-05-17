"""SM as a PyTorch nn.Module, with safetensors export + JSON labels sidecar.

Architecture mirrors the layered dependency we laid out:
    StandardModel
      ├── gauge      (Lie brackets + generators, frozen buffers)
      ├── reps       (Higher SU(2)/SU(3) reps, frozen buffers)
      ├── fields     (Particle vectors + field table, frozen buffers)
      ├── vertices   (Signed incidence matrix, frozen buffer)
      └── params     (19 trainable scalars + BSM extensions)

Export pattern (mirrors how SAEs ship: weights and interpretations split):
    sm.safetensors   -- pure tensor weights (no strings, no objects)
    sm_labels.json   -- axis labels per tensor, layer categorization, schema version

Why this is "more organized" than the flat bundle:
  - state_dict keys form a hierarchy (`gauge.su3_f`, `params.yukawa_up_eigvals`)
    so introspection tools can navigate by module.
  - Parameters and buffers are distinguished: PyTorch knows the 19 SM constants
    are the only trainable things; everything else is frozen architecture.
  - Labels live in JSON, not stuffed into safetensors metadata strings, so they
    can be diffed, edited, re-annotated independently of the binary.

Run:
    python3 sm_torch.py            -> writes sm.safetensors + sm_labels.json
    python3 sm_torch.py --verify   -> writes + reloads, runs sanity checks
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import smsae.sm.groups as G
import smsae.sm.parameters as P
import smsae.sm.reps as R
import smsae.sm.checks as T
from smsae.sm.embeddings import CONSERVED, COORDS, build_sm
from smsae.sm.lagrangian import SM_FIELDS


# ============================================================================
# Helpers
# ============================================================================
def _t(arr, dtype=None) -> torch.Tensor:
    arr = np.asarray(arr)
    if dtype is not None:
        return torch.tensor(arr, dtype=dtype)
    if np.iscomplexobj(arr):
        return torch.tensor(arr, dtype=torch.complex128)
    return torch.tensor(arr, dtype=torch.float64)


def _rep_tag(label: str) -> str:
    """Make a label like '6 (sym^2 fund, (p,0)=2,0)' into a valid attribute name."""
    return (label.split()[0]
            .replace("(", "").replace(")", "")
            .replace(",", "_").replace(".", "_"))


# ============================================================================
# Layered submodules
# ============================================================================
class GaugeStructure(nn.Module):
    """Layer 0: Lie brackets and fundamental-rep generators of SU(2) x SU(3)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("su2_f",          _t(G.SU2_F))
        self.register_buffer("su2_d",          _t(G.SU2_D))
        self.register_buffer("su2_generators", _t(G.SU2_GEN, torch.complex128))
        self.register_buffer("su3_f",          _t(G.SU3_F))
        self.register_buffer("su3_d",          _t(G.SU3_D))
        self.register_buffer("su3_generators", _t(G.SU3_GEN, torch.complex128))


class Representations(nn.Module):
    """Layer 1: SU(2) reps for j in {0, 1/2, 1, 3/2, 2}; SU(3) reps {1, 3, 3bar, 6, 8, 10}."""

    def __init__(self):
        super().__init__()
        for j in (0, 0.5, 1, 1.5, 2):
            T_R = R.spin_rep(j) if j > 0 else np.zeros((3, 1, 1), dtype=complex)
            attr = f"su2_j{str(j).replace('.', '_')}"
            self.register_buffer(attr, _t(T_R, torch.complex128))
        for label, T_R in R.all_su3_reps_up_to(3).items():
            self.register_buffer(f"su3_{_rep_tag(label)}", _t(T_R, torch.complex128))


class FieldContent(nn.Module):
    """Layer 2: Particle dense vectors + field (color, weak, Y, gen, chir, lorentz) table."""

    def __init__(self):
        super().__init__()
        sm = build_sm()
        names = sorted(sm.keys())
        vecs = np.array([sm[n].vec for n in names])
        self.register_buffer("particle_vectors", _t(vecs))

        field_table = np.array([
            [f.n_color, f.n_weak, f.Y, f.n_gen,
             {"L": 1, "R": -1, "-": 0}[f.chirality],
             {"scalar": 0, "spinor": 1, "vector": 2}[f.lorentz]]
            for f in SM_FIELDS
        ], dtype=float)
        self.register_buffer("field_table", _t(field_table))


class VertexAlgebra(nn.Module):
    """Layer 3: Signed vertex incidence matrix (B Q = 0 iff Q is conserved)."""

    def __init__(self):
        super().__init__()
        vertices = T.vertex_catalog()
        appearing = sorted({p for _, inc, out in vertices for p in inc + out})
        idx = {n: i for i, n in enumerate(appearing)}
        B = np.zeros((len(vertices), len(appearing)), dtype=float)
        for v, (_, inc, out) in enumerate(vertices):
            for p in inc:  B[v, idx[p]] += 1
            for p in out:  B[v, idx[p]] -= 1
        self.register_buffer("incidence", _t(B))


class FreeParameters(nn.Module):
    """Layer 4: The 19 minimal-SM trainable scalars (+7 BSM for neutrino sector).

      3  gauge_couplings        (g_s, g, g')
      2  higgs                  (v, lambda)            -- mu^2 derived
      9  yukawa_*_eigvals       (3 each for up, down, lepton)
      4  ckm_wolfenstein        (lambda, A, rho_bar, eta_bar)
      1  theta_qcd
     ---
     19  TOTAL (minimal SM)
     +4  pmns_angles            (theta_12, theta_23, theta_13, delta_CP)
     +3  neutrino_dm2           (dm2_21, dm2_31, dm2_32)
     ---
     26  TOTAL (with Dirac neutrino masses)
    """

    def __init__(self):
        super().__init__()
        self.gauge_couplings    = nn.Parameter(_t([P.G_S, P.G_WEAK, P.G_PRIME]))
        self.higgs              = nn.Parameter(_t([P.HIGGS_VEV, P.HIGGS_LAMBDA]))
        self.yukawa_up_eigvals  = nn.Parameter(_t(np.diag(P.YUKAWA_UP).real))
        self.yukawa_down_eigvals= nn.Parameter(_t(np.diag(P.YUKAWA_DOWN).real))
        self.yukawa_lep_eigvals = nn.Parameter(_t(np.diag(P.YUKAWA_LEP).real))
        self.ckm_wolfenstein    = nn.Parameter(_t([0.22500, 0.826, 0.159, 0.348]))
        self.theta_qcd          = nn.Parameter(_t(0.0))
        # BSM extension for observed neutrino oscillations
        self.pmns_angles        = nn.Parameter(_t([
            np.radians(33.41), np.radians(49.1), np.radians(8.54), np.radians(197),
        ]))
        self.neutrino_dm2       = nn.Parameter(_t([P.DM2_21, P.DM2_31, 0.0]))


class StandardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gauge    = GaugeStructure()
        self.reps     = Representations()
        self.fields   = FieldContent()
        self.vertices = VertexAlgebra()
        self.params   = FreeParameters()


# ============================================================================
# Labels sidecar
# ============================================================================
def build_labels(model: StandardModel) -> dict:
    sm = build_sm()
    particle_names = sorted(sm.keys())
    vertices = T.vertex_catalog()
    appearing = sorted({p for _, inc, out in vertices for p in inc + out})
    rep_index = []
    for j in (0, 0.5, 1, 1.5, 2):
        rep_index.append(("SU(2)", f"reps.su2_j{str(j).replace('.', '_')}", f"spin j={j}"))
    for label, _ in R.all_su3_reps_up_to(3).items():
        rep_index.append(("SU(3)", f"reps.su3_{_rep_tag(label)}", label))

    return {
        "schema_version": 1,
        "description": (
            "Standard Model encoded as a layered PyTorch nn.Module. Weights live in "
            "sm.safetensors; this file annotates each tensor's axes with the physical "
            "meaning of every row/column."
        ),
        "layer_categorization": {
            "gauge.*":    "architecture: Lie algebra structure constants and fundamental generators",
            "reps.*":     "architecture: higher irreducible representations (computed from gauge)",
            "fields.*":   "architecture: particle content and quantum-number table",
            "vertices.*": "architecture: signed vertex-particle incidence matrix",
            "params.*":   "trainable: 19 free SM parameters (+7 BSM for neutrino sector)",
        },
        "axis_labels": {
            "fields.particle_vectors": {
                "rows":    ("particle", particle_names),
                "columns": ("coord",     list(COORDS)),
                "notes":   "Cols 0..6 are conserved charges; cols 7..8 are spin and mass.",
            },
            "fields.field_table": {
                "rows":    ("field",     [f.name for f in SM_FIELDS]),
                "columns": ("attribute", ["n_color", "n_weak", "Y", "n_gen",
                                          "chirality_enc", "lorentz_enc"]),
            },
            "vertices.incidence": {
                "rows":    ("vertex",   [v[0] for v in vertices]),
                "columns": ("particle", appearing),
                "notes":   "B[v,p] = +1 incoming, -1 outgoing.  null(B) = conservation algebra.",
            },
            "params.gauge_couplings":      {"rows": ("coupling", ["g_s", "g", "g_prime"])},
            "params.higgs":                {"rows": ("higgs_param", ["v", "lambda"])},
            "params.yukawa_up_eigvals":    {"rows": ("up_flavor",      ["u", "c", "t"])},
            "params.yukawa_down_eigvals":  {"rows": ("down_flavor",    ["d", "s", "b"])},
            "params.yukawa_lep_eigvals":   {"rows": ("lepton_flavor",  ["e", "mu", "tau"])},
            "params.ckm_wolfenstein":      {"rows": ("ckm_wolfenstein",
                                                     ["lambda", "A", "rho_bar", "eta_bar"])},
            "params.pmns_angles":          {"rows": ("pmns_param",
                                                     ["theta_12", "theta_23", "theta_13", "delta_CP"])},
            "params.neutrino_dm2":         {"rows": ("nu_split",
                                                     ["dm2_21", "dm2_31", "dm2_32"])},
            "params.theta_qcd":            {"rows": ("scalar", ["theta_qcd"])},
        },
        "rep_index": [
            {"group": grp, "tensor_key": key, "label": lab}
            for grp, key, lab in rep_index
        ],
        "parameter_count": {
            "minimal_sm": 19,
            "with_dirac_neutrinos": 26,
            "breakdown": {
                "gauge_couplings": 3, "higgs": 2,
                "yukawa_up": 3, "yukawa_down": 3, "yukawa_lep": 3,
                "ckm": 4, "theta_qcd": 1,
                "pmns": 4, "neutrino_dm2": 3,
            },
        },
    }


# ============================================================================
# Export / import
# ============================================================================
def save_with_labels(model: StandardModel, st_path: str, labels_path: str) -> None:
    """Save state_dict to safetensors (with complex split) and labels to JSON."""
    from safetensors.torch import save_file

    sd = model.state_dict()
    out_sd: dict[str, torch.Tensor] = {}
    complex_keys: list[str] = []
    for k, v in sd.items():
        v = v.detach().contiguous()
        if v.is_complex():
            out_sd[k + "_re"] = v.real.contiguous()
            out_sd[k + "_im"] = v.imag.contiguous()
            complex_keys.append(k)
        else:
            out_sd[k] = v
    save_file(out_sd, st_path, metadata={
        "__complex_keys__": json.dumps(complex_keys),
        "__labels_sidecar__": os.path.basename(labels_path),
        "__schema_version__": "1",
    })

    with open(labels_path, "w") as f:
        json.dump(build_labels(model), f, indent=2)


def load_with_labels(st_path: str, labels_path: str) -> tuple[dict, dict]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    raw = load_file(st_path)
    with safe_open(st_path, framework="pt") as f:
        meta = f.metadata() or {}
    complex_keys = set(json.loads(meta.get("__complex_keys__", "[]")))

    state: dict[str, torch.Tensor] = {}
    consumed = set()
    for k in complex_keys:
        if k + "_re" in raw and k + "_im" in raw:
            state[k] = torch.complex(raw[k + "_re"], raw[k + "_im"])
            consumed.update({k + "_re", k + "_im"})
    for k, v in raw.items():
        if k not in consumed:
            state[k] = v

    with open(labels_path) as f:
        labels = json.load(f)
    return state, labels


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    model = StandardModel()

    print("=" * 86)
    print("StandardModel(nn.Module) -- state_dict hierarchy")
    print("=" * 86)
    by_module: dict[str, list] = {}
    param_names = set(dict(model.named_parameters()).keys())
    for key, tensor in model.state_dict().items():
        by_module.setdefault(key.split(".")[0], []).append((key, tensor))
    n_param = sum(p.numel() for p in model.parameters())
    n_buffer = sum(b.numel() for b in model.buffers())
    for prefix in ["gauge", "reps", "fields", "vertices", "params"]:
        print(f"\n--- {prefix} ---")
        for key, tensor in sorted(by_module[prefix]):
            tag = "[param]" if key in param_names else "[buf]  "
            dt = str(tensor.dtype).replace("torch.", "")
            print(f"  {tag} {key:<48s} {str(tuple(tensor.shape)):<20s} {dt}")
    print(f"\nTotal trainable parameters: {n_param}")
    print(f"Total frozen buffer entries: {n_buffer}")

    save_with_labels(model, "sm.safetensors", "sm_labels.json")
    print(f"\nWrote sm.safetensors  ({os.path.getsize('sm.safetensors'):>9,} bytes)")
    print(f"Wrote sm_labels.json  ({os.path.getsize('sm_labels.json'):>9,} bytes)")

    if "--verify" in sys.argv:
        print("\n" + "=" * 86)
        print("Roundtrip verification")
        print("=" * 86)
        state, labels = load_with_labels("sm.safetensors", "sm_labels.json")
        print(f"  Loaded {len(state)} tensors, {len(labels['axis_labels'])} labeled tensors.")

        # Use labels to reconstruct vertex closure: B Q = 0
        v_cols = labels["axis_labels"]["vertices.incidence"]["columns"][1]
        p_rows = labels["axis_labels"]["fields.particle_vectors"]["rows"][1]
        coord_cols = labels["axis_labels"]["fields.particle_vectors"]["columns"][1]
        name_to_row = {n: i for i, n in enumerate(p_rows)}
        cons_idx = [coord_cols.index(c) for c in
                    ["Q", "B", "L_e", "L_mu", "L_tau", "C3", "C8"]]
        Q = state["fields.particle_vectors"][[name_to_row[n] for n in v_cols]][:, cons_idx]
        B = state["vertices.incidence"]
        residual = float((B @ Q).abs().max())
        rank = int(torch.linalg.matrix_rank(B, tol=1e-9))
        print(f"  ||B Q||_max       = {residual:.2e}  (expect ~0)")
        print(f"  rank(B), nullity  = {rank}, {B.shape[1] - rank}  (expect nullity = 7)")

        # Higher rep check: SU(3) sextet has T(R)=5/2, C2=10/3
        T6 = state["reps.su3_6"]
        tR = float(torch.trace(T6[0] @ T6[0]).real)
        C2 = float(sum(T6[a] @ T6[a] for a in range(8))[0, 0].real)
        print(f"  SU(3) 6:  T(R) = {tR:.4f} (expect 2.5),  C2 = {C2:.4f} (expect 3.333)")

        # Yukawa consistency: y_t = sqrt(2) m_t / v
        v_val = float(state["params.higgs"][0])
        y_t = float(state["params.yukawa_up_eigvals"][2])
        m_t = y_t * v_val / np.sqrt(2)
        print(f"  m_t reconstructed from y_t and v = {m_t:.2f} GeV (PDG 172.76)")

        # Parameter count from labels
        n_total = labels["parameter_count"]["with_dirac_neutrinos"]
        print(f"  Parameter count (from labels):  {n_total}  "
              f"(minimal SM = {labels['parameter_count']['minimal_sm']})")
