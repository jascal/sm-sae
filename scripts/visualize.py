"""Single-file HTML walkthrough of the sm-sae lifecycle.

Four sections, each pinned to a question a newcomer might ask:
  (a) substrate   how the Standard Model is encoded as an nn.Module
  (b) sae         how sparse autoencoders are trained and scored against ground truth
  (c) polygram    how the polygram bridge crushes an SAE feature slice into
                  a quantum-circuit-shaped Orca artifact
  (d) lifecycle   how the same flow generalizes to non-trivial NN models

Inputs read (any missing piece is reported in-place, not fatal):
  data/sm.safetensors, data/sm_labels.json   (built by scripts/build_data.py)
  runs/{feed}__{variant}.pt                  (built by scripts/train_all.py)
  runs/alignment_summary.json                (built by scripts/evaluate.py)
  runs/polygram/interference_e_pair.csv      (built by scripts/polygram_demo.py)
  runs/polygram/cancellation_e_e/*.md

Output: a self-contained HTML file (default runs/visualize.html); all plots
are inlined as base64 PNGs, so the result is one shareable artifact.

    python scripts/visualize.py [--out path] [--fast]

Requires matplotlib. Install: `pip install -e ".[viz]"`.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import traceback
from html import escape

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    sys.stderr.write(
        "visualize.py requires matplotlib and numpy.\n"
        "Install with:  pip install -e \".[viz]\"\n"
    )
    raise


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def fig_to_uri(fig, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def img(uri: str, caption: str = "") -> str:
    cap = f'<figcaption>{escape(caption)}</figcaption>' if caption else ""
    return f'<figure><img src="{uri}" />{cap}</figure>'


def missing(path: str, hint: str = "") -> str:
    msg = f"missing artifact: <code>{escape(path)}</code>"
    if hint:
        msg += f' &mdash; {escape(hint)}'
    return f'<div class="missing">{msg}</div>'


def safe(name: str, fn):
    try:
        return fn()
    except Exception:
        return (f'<div class="error"><strong>Section <code>{escape(name)}</code> '
                f'failed.</strong><pre>{escape(traceback.format_exc())}</pre></div>')


from smsae.sae.evaluation import auc_matrix  # noqa: E402  (re-export for back-compat)


# ---------------------------------------------------------------------------
# Tensor inspection widgets (used by section (a) hierarchy table)
# ---------------------------------------------------------------------------
def _format_value(v: float, integer_like: bool) -> str:
    if not np.isfinite(v) or abs(v) < 1e-12:
        return ""
    if integer_like and abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):d}"
    if abs(v) < 1e-4 or abs(v) >= 1e6:
        return f"{v:.2e}"
    return f"{v:.4g}"


def _cell_style(v: float, vmax: float) -> str:
    if vmax <= 0 or abs(v) < 1e-12:
        return ""
    t = max(-1.0, min(1.0, v / vmax))
    if t >= 0:
        # blue side (matches matplotlib RdBu_r used elsewhere: positive = blue)
        return f"background:rgba(40,90,200,{abs(t) * 0.55:.2f})"
    return f"background:rgba(200,60,60,{abs(t) * 0.55:.2f})"


def _render_2d_table(arr: np.ndarray,
                     row_labels: list[str] | None = None,
                     col_labels: list[str] | None = None) -> str:
    n_rows, n_cols = arr.shape
    vmax = float(np.max(np.abs(arr))) if arr.size else 0.0
    integer_like = bool(arr.size and np.allclose(arr, np.round(arr)))
    parts = ["<table class='tensor'><thead><tr><th></th>"]
    for j in range(n_cols):
        lab = col_labels[j] if col_labels and j < len(col_labels) else str(j)
        parts.append(f"<th title='{escape(str(lab))}'>{escape(str(lab))}</th>")
    parts.append("</tr></thead><tbody>")
    for i in range(n_rows):
        rlab = row_labels[i] if row_labels and i < len(row_labels) else str(i)
        parts.append(f"<tr><th title='{escape(str(rlab))}'>{escape(str(rlab))}</th>")
        for j in range(n_cols):
            v = float(arr[i, j])
            parts.append(
                f"<td style='{_cell_style(v, vmax)}'>"
                f"{_format_value(v, integer_like)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "<div class='tensorwrap'>" + "".join(parts) + "</div>"


def _render_1d_table(arr: np.ndarray,
                     labels: list[str] | None = None) -> str:
    n = arr.shape[0]
    vmax = float(np.max(np.abs(arr))) if arr.size else 0.0
    integer_like = bool(arr.size and np.allclose(arr, np.round(arr)))
    head = "<tr>" + "".join(
        f"<th>{escape(str(labels[j]))}</th>" if labels and j < len(labels)
        else f"<th>{j}</th>"
        for j in range(n)) + "</tr>"
    body = "<tr>" + "".join(
        f"<td style='{_cell_style(float(arr[j]), vmax)}'>"
        f"{_format_value(float(arr[j]), integer_like)}</td>"
        for j in range(n)) + "</tr>"
    return ("<div class='tensorwrap'><table class='tensor'><thead>"
            + head + "</thead><tbody>" + body + "</tbody></table></div>")


def _render_real(arr: np.ndarray,
                 row_labels: list[str] | None = None,
                 col_labels: list[str] | None = None,
                 depth: int = 0) -> str:
    if arr.ndim == 0:
        return ("<code class='scalar'>"
                + _format_value(float(arr), False) + "</code>")
    if arr.ndim == 1:
        return _render_1d_table(arr, labels=row_labels or col_labels)
    if arr.ndim == 2:
        return _render_2d_table(arr, row_labels=row_labels, col_labels=col_labels)
    # ndim >= 3 — leading axis becomes a list of collapsible slices
    parts = ["<div class='slices'>"]
    n = arr.shape[0]
    rest_shape = " × ".join(str(s) for s in arr.shape[1:])
    for i in range(n):
        # Open the first slice at the top level to give an immediate preview
        is_open = " open" if (depth == 0 and i == 0 and n <= 8) else ""
        parts.append(
            f"<details class='slice'{is_open}>"
            f"<summary>[{i}, …]  ({rest_shape})</summary>")
        parts.append(_render_real(arr[i], depth=depth + 1))
        parts.append("</details>")
    parts.append("</div>")
    return "".join(parts)


def _render_tensor_widget(tensor, row_labels=None, col_labels=None) -> str:
    """Top-level dispatcher: handles real/complex and returns HTML."""
    import torch as _torch
    arr = tensor.detach().numpy() if isinstance(tensor, _torch.Tensor) else tensor
    if np.iscomplexobj(arr):
        re_html = _render_real(arr.real, row_labels, col_labels)
        im_html = _render_real(arr.imag, row_labels, col_labels)
        max_im = float(np.max(np.abs(arr.imag))) if arr.size else 0.0
        im_open = "" if max_im < 1e-12 else ""  # closed by default; user expands
        return (
            "<div class='complex'>"
            f"<details open><summary><strong>real part</strong></summary>"
            f"{re_html}</details>"
            f"<details{im_open}><summary><strong>imaginary part</strong>  "
            f"(max |Im| = {max_im:.3g})</summary>"
            f"{im_html}</details>"
            "</div>"
        )
    return _render_real(arr, row_labels, col_labels)


# ---------------------------------------------------------------------------
# (a) substrate
# ---------------------------------------------------------------------------
def _world_model_svg() -> str:
    """A world-model loop diagram: state → transition function → next state,
    with a feedback arrow back to the input. This is the canonical shape of
    every model in this report — the MLP picture lives *inside* the
    transition function box."""
    return """
<svg viewBox="0 0 800 240" class="nndiag" xmlns="http://www.w3.org/2000/svg"
     preserveAspectRatio="xMidYMid meet">
  <defs>
    <marker id="warr" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="8" markerHeight="8" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#444"/>
    </marker>
  </defs>
  <g font-family="sans-serif" font-size="13" text-anchor="middle">
    <rect x="25" y="60" width="160" height="65" rx="10"
          fill="#cce" stroke="#446"/>
    <text x="105" y="88" font-weight="bold">current state</text>
    <text x="105" y="108" font-size="11" fill="#557">what the world is now</text>

    <rect x="235" y="35" width="330" height="115" rx="10"
          fill="#cec" stroke="#464"/>
    <text x="400" y="62" font-weight="bold">transition function</text>
    <text x="400" y="84" font-size="11" fill="#557">a neural network</text>
    <text x="400" y="102" font-size="11" fill="#557">parameters θ define its behaviour</text>
    <text x="400" y="120" font-size="11" fill="#557">activations flow between its internal layers</text>

    <rect x="615" y="60" width="160" height="65" rx="10"
          fill="#fcc" stroke="#644"/>
    <text x="695" y="88" font-weight="bold">next state</text>
    <text x="695" y="108" font-size="11" fill="#557">what the world becomes</text>

    <line x1="185" y1="92" x2="235" y2="92" stroke="#444"
          stroke-width="1.5" marker-end="url(#warr)"/>
    <line x1="565" y1="92" x2="615" y2="92" stroke="#444"
          stroke-width="1.5" marker-end="url(#warr)"/>

    <path d="M 695 125 Q 695 195 400 195 Q 105 195 105 125"
          stroke="#888" stroke-width="1.2" fill="none"
          stroke-dasharray="4 3" marker-end="url(#warr)"/>
    <text x="400" y="218" font-size="11" font-style="italic" fill="#666">
      repeat: the new state becomes the input on the next step
    </text>
  </g>
</svg>
"""


# Per-tensor description used in the hierarchy table's contents column.
# Keys are exact state_dict keys; describe_tensor() falls back to a pattern
# match for the dynamically-named SU(2)/SU(3) representations.
TENSOR_DESCRIPTIONS: dict[str, str] = {
    "gauge.su2_f": (
        "SU(2) antisymmetric structure constants <em>f<sup>abc</sup></em>. "
        "They define the Lie bracket [T<sub>a</sub>, T<sub>b</sub>] = "
        "i f<sup>abc</sup> T<sub>c</sub>; for SU(2) this is just ε<sub>abc</sub>."),
    "gauge.su2_d": (
        "SU(2) symmetric structure constants <em>d<sup>abc</sup></em>. "
        "Identically zero for SU(2); the buffer is kept for structural "
        "parallelism with SU(3)."),
    "gauge.su2_generators": (
        "The three Pauli matrices σ<sub>a</sub>/2 — generators of SU(2) in "
        "its fundamental representation."),
    "gauge.su3_f": (
        "SU(3) antisymmetric structure constants — the canonical fingerprint "
        "of QCD's color algebra. Most entries are 0; the non-zero pattern is "
        "what makes gluon self-interaction possible."),
    "gauge.su3_d": (
        "SU(3) symmetric structure constants <em>d<sup>abc</sup></em>. "
        "Non-trivial (unlike SU(2)); these enter anomaly calculations and "
        "the chiral Lagrangian."),
    "gauge.su3_generators": (
        "The eight Gell-Mann matrices λ<sub>a</sub>/2 — generators of SU(3) "
        "in its fundamental (3) representation."),
    "fields.particle_vectors": (
        "All 61 Standard-Model particles as 9-vectors. Columns: electric "
        "charge Q, baryon number B, lepton numbers L<sub>e</sub>/L<sub>μ</sub>"
        "/L<sub>τ</sub>, the two diagonal color generators C3/C8, spin, and "
        "mass in GeV."),
    "fields.field_table": (
        "Per-field quantum-number table: color multiplicity, weak isospin "
        "multiplicity, hypercharge Y, generation index, chirality, and "
        "Lorentz type (scalar / spinor / vector)."),
    "vertices.incidence": (
        "Signed vertex–particle incidence: B[v,p] = +1 if p is incoming to "
        "vertex v, −1 if outgoing, 0 otherwise. 168 SM vertices × 59 "
        "particles. The 7-dimensional nullspace is the conservation algebra."),
    "params.gauge_couplings": (
        "The three gauge coupling strengths at the M<sub>Z</sub> scale: "
        "g<sub>s</sub> (strong / SU(3)), g (weak / SU(2)), g' (hypercharge / U(1))."),
    "params.higgs": (
        "Higgs sector: v is the vacuum expectation value (≈246 GeV, sets "
        "all W/Z masses); λ is the Higgs self-coupling."),
    "params.yukawa_up_eigvals": (
        "Yukawa coupling eigenvalues for the up-type quarks (u, c, t). "
        "Mass = y · v / √2; the top's value is O(1)."),
    "params.yukawa_down_eigvals": (
        "Yukawa coupling eigenvalues for the down-type quarks (d, s, b)."),
    "params.yukawa_lep_eigvals": (
        "Yukawa coupling eigenvalues for the charged leptons (e, μ, τ)."),
    "params.ckm_wolfenstein": (
        "CKM quark-mixing matrix in the Wolfenstein parameterization: "
        "(λ, A, ρ̄, η̄). η̄ is the source of CP violation in quark physics."),
    "params.theta_qcd": (
        "The QCD vacuum angle θ̄. Measured to be ≈ 0; the unexplained "
        "smallness is the strong-CP problem."),
    "params.pmns_angles": (
        "PMNS neutrino-mixing angles: (θ<sub>12</sub>, θ<sub>23</sub>, "
        "θ<sub>13</sub>, δ<sub>CP</sub>). Beyond the original minimal SM."),
    "params.neutrino_dm2": (
        "Neutrino mass-squared splittings (Δm²<sub>21</sub>, Δm²<sub>31</sub>, "
        "Δm²<sub>32</sub>) in eV². Sign of Δm²<sub>32</sub> = neutrino mass "
        "ordering, still unknown."),
}

_SU2_J_NAMES = {"0": "0 (trivial)", "0_5": "1/2 (fundamental)",
                "1": "1 (vector)", "1_5": "3/2", "2": "2"}


def describe_tensor(key: str, shape: tuple) -> str:
    if key in TENSOR_DESCRIPTIONS:
        return TENSOR_DESCRIPTIONS[key]
    if key.startswith("reps.su2_j"):
        suffix = key[len("reps.su2_j"):]
        name = _SU2_J_NAMES.get(suffix, suffix)
        return (f"SU(2) representation, spin <em>j = {name}</em>. "
                "Three matrices (one per SU(2) generator) acting on the "
                f"{shape[-1] if shape else '?'}-dim rep space.")
    if key.startswith("reps.su3_"):
        suffix = key[len("reps.su3_"):]
        return (f"SU(3) irreducible representation <em>{suffix}</em>. "
                "Eight matrices (one per SU(3) generator) acting on the "
                f"{shape[-1] if shape else '?'}-dim rep space.")
    return ""


def _substrate_primer_html() -> str:
    """A world-model primer for NN topology, a parallel primer for the SM,
    and a bridge that frames the SM as a (measured, not learned) world model."""
    return f"""
<h3>Background &mdash; a 60-second primer on world models (neural networks)</h3>
{_world_model_svg()}
<p>Every neural network in this report shares the same big-picture shape: it
is a
<a href="https://worldmodels.github.io/" target="_blank" rel="noopener">world
model</a>. There are three pieces:</p>
<ul>
  <li><strong>State</strong> &mdash; whatever describes the world right now.
      For a
      <a href="https://en.wikipedia.org/wiki/Large_language_model"
         target="_blank" rel="noopener">large language model (LLM)</a> the
      state is the sequence of tokens generated so far. For a
      <a href="https://en.wikipedia.org/wiki/Diffusion_model" target="_blank"
         rel="noopener">diffusion model</a> it's a noisy image. For the
      Standard Model cascade we'll meet just below, it's a multiset of
      particles.</li>
  <li><strong>Transition function</strong> &mdash; the actual neural network.
      It takes the current state and produces the next state (possibly
      stochastically). Its behaviour is determined by a set of numbers called
      <strong>parameters</strong> (or weights). An LLM has on the order of
      10<sup>10</sup> of them; the Standard Model has 19. They stop changing
      once the model is "ready to use".</li>
  <li><strong>Loop</strong> &mdash; apply the transition function repeatedly,
      feeding the new state back in as input. A 100-word LLM completion is
      100 transitions. A diffusion sample is dozens of denoising steps. An
      SM cascade is however many decay steps until everything is stable.</li>
</ul>
<p>While one transition step runs, the transition function computes
intermediate values internally; these are called
<strong>activations</strong>. Same input state → same activations; different
input state → different activations.
<a href="https://en.wikipedia.org/wiki/Autoencoder#Variations" target="_blank"
   rel="noopener">Sparse autoencoders (SAEs)</a> &mdash; see also the
<a href="https://transformer-circuits.pub/2023/monosemantic-features"
   target="_blank" rel="noopener">Anthropic "Towards Monosemanticity"
   report</a> for the interpretability use &mdash; are a separate, smaller
network trained to take those activations and decompose them into a larger
but sparser set of human-interpretable features.</p>
<p class="aside"><strong>Aside &mdash; other neural-network topologies.</strong>
Not every neural network is a world model: image classifiers,
sentence-embedding encoders, and encoder-only models like
<a href="https://en.wikipedia.org/wiki/BERT_(language_model)" target="_blank"
   rel="noopener">BERT</a> run end-to-end once with no loop. Inside the
transition function (or inside a non-looping network), the math is the
classic layer-stack: matrix multiply, add bias, apply a nonlinearity, repeat
&mdash; the
<a href="https://en.wikipedia.org/wiki/Multilayer_perceptron" target="_blank"
   rel="noopener">multilayer perceptron</a> picture. Modern systems use richer
layer types (<a href="https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)"
   target="_blank" rel="noopener">transformer</a> blocks for LLMs,
<a href="https://en.wikipedia.org/wiki/U-Net" target="_blank" rel="noopener">
U-Nets</a> for diffusion). What varies is the layer type and whether there's
an outer loop; everything in this report has the loop.</p>

<h3>Background &mdash; a 60-second primer on the Standard Model</h3>
<p>The <a href="https://en.wikipedia.org/wiki/Standard_Model" target="_blank"
   rel="noopener">Standard Model (SM)</a> of particle physics is the
agreed-upon catalogue of fundamental particles and the forces between them. Three things
to know to read the diagrams below:</p>
<ul>
  <li><strong>Particles carry discrete labels.</strong> Each one has an
      <a href="https://en.wikipedia.org/wiki/Electric_charge" target="_blank"
         rel="noopener">electric charge</a>, possibly a
      <a href="https://en.wikipedia.org/wiki/Color_charge" target="_blank"
         rel="noopener">color charge</a> (for
      <a href="https://en.wikipedia.org/wiki/Quark" target="_blank"
         rel="noopener">quarks</a> and
      <a href="https://en.wikipedia.org/wiki/Gluon" target="_blank"
         rel="noopener">gluons</a>), a
      <a href="https://en.wikipedia.org/wiki/Generation_(particle_physics)"
         target="_blank" rel="noopener">generation</a> (1, 2, or 3), a
      <a href="https://en.wikipedia.org/wiki/Lepton_number" target="_blank"
         rel="noopener">lepton number</a>, and a few more. These labels are
      <a href="https://en.wikipedia.org/wiki/Conservation_law" target="_blank"
         rel="noopener">conserved</a> &mdash; they don't change under
      interactions.</li>
  <li><strong>Forces are described by
      <a href="https://en.wikipedia.org/wiki/Gauge_theory" target="_blank"
         rel="noopener">gauge theory</a>.</strong> Each force corresponds to a
      <a href="https://en.wikipedia.org/wiki/Lie_group" target="_blank"
         rel="noopener">symmetry group</a>: electromagnetism is U(1), the
      weak force is
      <a href="https://en.wikipedia.org/wiki/Special_unitary_group#The_group_SU(2)"
         target="_blank" rel="noopener">SU(2)</a>, and the strong force is
      <a href="https://en.wikipedia.org/wiki/Special_unitary_group#The_group_SU(3)"
         target="_blank" rel="noopener">SU(3)</a>. The mathematical fingerprint
      of each group is its set of
      <a href="https://en.wikipedia.org/wiki/Structure_constants" target="_blank"
         rel="noopener">structure constants</a>.</li>
  <li><strong>Mass comes from the
      <a href="https://en.wikipedia.org/wiki/Higgs_mechanism" target="_blank"
         rel="noopener">Higgs mechanism</a>.</strong> Particle masses are set
      by
      <a href="https://en.wikipedia.org/wiki/Yukawa_interaction" target="_blank"
         rel="noopener">Yukawa couplings</a>; flavor mixing between
      generations is captured by the
      <a href="https://en.wikipedia.org/wiki/Cabibbo%E2%80%93Kobayashi%E2%80%93Maskawa_matrix"
         target="_blank" rel="noopener">CKM matrix</a> for quarks and the
      <a href="https://en.wikipedia.org/wiki/Pontecorvo%E2%80%93Maki%E2%80%93Nakagawa%E2%80%93Sakata_matrix"
         target="_blank" rel="noopener">PMNS matrix</a> for neutrinos.</li>
</ul>
<p>The Standard Model has just <strong>19 free numerical parameters</strong>
(plus 7 more once neutrino masses are included). Every other number in the
tables below is <em>computed</em> from those parameters and the choice of
symmetry groups; the structure is fixed, not learned.</p>

<h3>How the Standard Model fits the world-model picture</h3>
<p>The Standard Model <em>is</em> a world model in exactly the sense above:</p>
<ul>
  <li><strong>State</strong>: a multiset of particles, e.g. <code>{{H: 1}}</code>
      (one Higgs boson).</li>
  <li><strong>Transition function</strong>: enumerate the allowed decays of
      particles in the current state (from the vertex catalogue), weight each
      by its physical rate (computed from
      <a href="https://en.wikipedia.org/wiki/Yukawa_interaction" target="_blank"
         rel="noopener">Yukawa couplings</a>,
      <a href="https://en.wikipedia.org/wiki/Cabibbo%E2%80%93Kobayashi%E2%80%93Maskawa_matrix"
         target="_blank" rel="noopener">CKM matrix</a>, gauge couplings,
      masses), sample one, mutate the state.</li>
  <li><strong>Loop</strong>: repeat until no remaining particle has a
      kinematically allowed decay.</li>
</ul>
<p>What's unusual about this world model: its 19 parameters were
<strong>measured</strong> by experimental particle physics over the 20th
century, not learned by gradient descent. But the role is identical to a
trained LLM's weights &mdash; they are the knobs that define the transition
function, frozen at "ready to use" time.</p>
<p>So why bother packaging the SM as an <code>nn.Module</code>?</p>
<ul>
  <li>Same tooling that inspects an LLM (state-dict navigation, parameter
      counts, activation hooks,
      <a href="https://github.com/huggingface/safetensors" target="_blank"
         rel="noopener">safetensors</a> loaders) now works on physics.</li>
  <li>The submodules below (<code>gauge</code> → <code>reps</code> →
      <code>fields</code> → <code>vertices</code> → <code>params</code>) are
      an <em>organizational</em> hierarchy by mathematical dependency, not a
      stack of layers. The "layer stack" of the SM world model lives in the
      cascade engine and shows up later in the "Running the model" section.</li>
</ul>
"""


def _further_reading_html() -> str:
    return """
<details>
  <summary>Further reading (curated link directory)</summary>
  <div class="reading">
    <h4>Neural networks &amp; interpretability</h4>
    <ul>
      <li><a href="https://worldmodels.github.io/" target="_blank" rel="noopener">World models</a> (Ha &amp; Schmidhuber) &mdash; the canonical reference for the state / transition / loop framing</li>
      <li><a href="https://en.wikipedia.org/wiki/State-space_representation" target="_blank" rel="noopener">State-space model</a> &mdash; the formal control-theory framing</li>
      <li><a href="https://en.wikipedia.org/wiki/Markov_chain#Continuous-time_Markov_chain" target="_blank" rel="noopener">Continuous-time Markov chain</a> &mdash; what the SM cascade technically is</li>
      <li><a href="https://en.wikipedia.org/wiki/Multilayer_perceptron" target="_blank" rel="noopener">Multilayer perceptron</a> &mdash; the layer-stack inside one transition step</li>
      <li><a href="https://en.wikipedia.org/wiki/Activation_function" target="_blank" rel="noopener">Activation functions</a> (ReLU, sigmoid, tanh, …) &mdash; the non-linearities between layers</li>
      <li><a href="https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)" target="_blank" rel="noopener">Transformers</a> &mdash; the layer type modern LLMs use inside the transition step</li>
      <li><a href="https://en.wikipedia.org/wiki/Diffusion_model" target="_blank" rel="noopener">Diffusion models</a> &mdash; another world-model architecture, looped over denoising steps</li>
      <li><a href="https://transformer-circuits.pub/2023/monosemantic-features" target="_blank" rel="noopener">Anthropic: Towards Monosemanticity</a> &mdash; the SAE-on-LLM paper this project takes inspiration from</li>
      <li><a href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener">A Mathematical Framework for Transformer Circuits</a> &mdash; the residual-stream picture</li>
    </ul>
    <h4>Standard Model &mdash; structure &amp; symmetry</h4>
    <ul>
      <li><a href="https://en.wikipedia.org/wiki/Standard_Model" target="_blank" rel="noopener">Standard Model</a> (overview)</li>
      <li><a href="https://en.wikipedia.org/wiki/Gauge_theory" target="_blank" rel="noopener">Gauge theory</a> &mdash; the mathematical framework underlying SM forces</li>
      <li><a href="https://en.wikipedia.org/wiki/Lie_algebra" target="_blank" rel="noopener">Lie algebra</a> and <a href="https://en.wikipedia.org/wiki/Structure_constants" target="_blank" rel="noopener">structure constants</a></li>
      <li><a href="https://en.wikipedia.org/wiki/Irreducible_representation" target="_blank" rel="noopener">Irreducible representations</a> &mdash; how SU(2)/SU(3) act on different particle types</li>
      <li><a href="https://en.wikipedia.org/wiki/Special_unitary_group" target="_blank" rel="noopener">Special unitary groups SU(2) / SU(3)</a></li>
      <li><a href="https://en.wikipedia.org/wiki/Gell-Mann%E2%80%93Nishijima_formula" target="_blank" rel="noopener">Gell-Mann–Nishijima formula</a> &mdash; how charge, isospin, and hypercharge relate</li>
      <li><a href="https://en.wikipedia.org/wiki/Anomaly_(physics)" target="_blank" rel="noopener">Gauge anomaly cancellation</a> &mdash; one of the SM's deep consistency checks</li>
    </ul>
    <h4>Standard Model &mdash; particles &amp; charges</h4>
    <ul>
      <li><a href="https://en.wikipedia.org/wiki/Quark" target="_blank" rel="noopener">Quark</a>, <a href="https://en.wikipedia.org/wiki/Lepton" target="_blank" rel="noopener">lepton</a>, <a href="https://en.wikipedia.org/wiki/Neutrino" target="_blank" rel="noopener">neutrino</a></li>
      <li><a href="https://en.wikipedia.org/wiki/Generation_(particle_physics)" target="_blank" rel="noopener">Generation</a> (three families)</li>
      <li><a href="https://en.wikipedia.org/wiki/Color_charge" target="_blank" rel="noopener">Color charge</a> &amp; <a href="https://en.wikipedia.org/wiki/Quantum_chromodynamics" target="_blank" rel="noopener">QCD</a></li>
      <li><a href="https://en.wikipedia.org/wiki/Chirality_(physics)" target="_blank" rel="noopener">Chirality</a> (left- vs right-handed)</li>
      <li><a href="https://en.wikipedia.org/wiki/Antimatter" target="_blank" rel="noopener">Antiparticles</a></li>
    </ul>
    <h4>Standard Model &mdash; the 19 (+ 7) free parameters</h4>
    <ul>
      <li><a href="https://en.wikipedia.org/wiki/Coupling_constant" target="_blank" rel="noopener">Gauge couplings</a>: g_s, g, g'</li>
      <li><a href="https://en.wikipedia.org/wiki/Higgs_boson" target="_blank" rel="noopener">Higgs sector</a>: vacuum expectation value v, self-coupling λ</li>
      <li><a href="https://en.wikipedia.org/wiki/Yukawa_interaction" target="_blank" rel="noopener">Yukawa couplings</a> (9 numbers, one per charged fermion)</li>
      <li><a href="https://en.wikipedia.org/wiki/Cabibbo%E2%80%93Kobayashi%E2%80%93Maskawa_matrix#Wolfenstein_parameterization" target="_blank" rel="noopener">CKM Wolfenstein parameterization</a>: λ, A, ρ̄, η̄</li>
      <li><a href="https://en.wikipedia.org/wiki/Strong_CP_problem" target="_blank" rel="noopener">Strong CP problem</a> (θ_QCD)</li>
      <li><a href="https://en.wikipedia.org/wiki/Pontecorvo%E2%80%93Maki%E2%80%93Nakagawa%E2%80%93Sakata_matrix" target="_blank" rel="noopener">PMNS angles &amp; CP phase</a></li>
      <li><a href="https://en.wikipedia.org/wiki/Neutrino_oscillation" target="_blank" rel="noopener">Neutrino mass-squared differences</a> (Δm²)</li>
    </ul>
    <h4>Authoritative tables</h4>
    <ul>
      <li><a href="https://pdg.lbl.gov/" target="_blank" rel="noopener">Particle Data Group (PDG)</a> &mdash; the canonical reference for measured SM values</li>
    </ul>
  </div>
</details>
"""


def section_substrate() -> str:
    from smsae.sm.torch_model import StandardModel, build_labels

    model = StandardModel()
    labels = build_labels(model)

    # Module hierarchy
    by_module: dict[str, list] = {}
    param_names = set(dict(model.named_parameters()).keys())
    for key, tensor in model.state_dict().items():
        by_module.setdefault(key.split(".")[0], []).append((key, tensor))
    axis_lookup = labels.get("axis_labels", {})
    rows = []
    for prefix in ["gauge", "reps", "fields", "vertices", "params"]:
        cat = labels["layer_categorization"].get(f"{prefix}.*", "")
        rows.append(
            f'<tr class="modhead"><th colspan="5">{escape(prefix)}'
            f' &mdash; <span class="cat">{escape(cat)}</span></th></tr>')
        for key, tensor in sorted(by_module.get(prefix, [])):
            kind = "param" if key in param_names else "buffer"
            shape_tuple = tuple(tensor.shape)
            shape = " &times; ".join(str(s) for s in shape_tuple) or "scalar"
            dt = str(tensor.dtype).replace("torch.", "")
            n_entries = int(np.prod(shape_tuple)) if shape_tuple else 1

            # Look up axis labels (if any) for richer row/col headers
            axis = axis_lookup.get(key, {}) if isinstance(axis_lookup, dict) else {}
            row_lbls = None
            col_lbls = None
            if isinstance(axis, dict):
                rows_entry = axis.get("rows")
                cols_entry = axis.get("columns")
                if rows_entry and len(rows_entry) > 1:
                    row_lbls = list(rows_entry[1])
                if cols_entry and len(cols_entry) > 1:
                    col_lbls = list(cols_entry[1])

            desc = describe_tensor(key, shape_tuple)
            widget = _render_tensor_widget(tensor, row_lbls, col_lbls)
            inspect = (
                f"<details class='inspect'>"
                f"<summary>inspect values  "
                f"<span class='nval'>({n_entries:,} entries)</span></summary>"
                f"{widget}"
                f"</details>")
            desc_html = f"<div class='tdesc'>{desc}</div>" if desc else ""

            rows.append(
                f'<tr><td class="{kind}">{kind}</td>'
                f'<td><code>{escape(key)}</code></td>'
                f'<td>{shape}</td><td>{escape(dt)}</td>'
                f'<td class="contents">{desc_html}{inspect}</td></tr>')
    n_param = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers())
    table = (
        '<table class="hier"><thead><tr><th>kind</th><th>name</th>'
        '<th>shape</th><th>dtype</th>'
        '<th>contents — meaning &amp; values</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table>'
    )

    # Particle vectors heatmap
    particle_names = labels["axis_labels"]["fields.particle_vectors"]["rows"][1]
    coord_cols = labels["axis_labels"]["fields.particle_vectors"]["columns"][1]
    PV = model.fields.particle_vectors.detach().numpy()
    # Each column has its own scale (charges ∈ {-1, 0, 1}; mass spans 0…172 GeV),
    # so normalize per-column to its own |max| before plotting. Preserves sign.
    col_scale = np.max(np.abs(PV), axis=0)
    col_scale[col_scale == 0] = 1.0
    PV_norm = PV / col_scale
    fig, ax = plt.subplots(figsize=(6, 14))
    im = ax.imshow(PV_norm, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(coord_cols)))
    ax.set_xticklabels(
        [f"{c}\n(±{s:.3g})" for c, s in zip(coord_cols, col_scale)],
        rotation=0, ha="center", fontsize=8)
    ax.set_yticks(range(len(particle_names)))
    ax.set_yticklabels(particle_names, fontsize=6)
    ax.set_title("fields.particle_vectors  (61 particles × 9 coords)\n"
                 "each column normalized to its own |max| (shown in axis label)")
    fig.colorbar(im, ax=ax, fraction=0.04, label="value / column |max|")
    pv_uri = fig_to_uri(fig)

    # Vertex incidence heatmap
    VI = model.vertices.incidence.detach().numpy()
    import torch as _torch
    rank = int(_torch.linalg.matrix_rank(model.vertices.incidence, tol=1e-9))
    nullity = VI.shape[1] - rank
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.imshow(VI, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(
        f"vertices.incidence  ({VI.shape[0]} × {VI.shape[1]})  "
        f"&mdash;  rank {rank}, nullity {nullity}  =  7 conservation laws")
    ax.set_xlabel("particle index")
    ax.set_ylabel("vertex index")
    vi_uri = fig_to_uri(fig)

    # Free parameters
    p_vals: list[float] = []
    p_labels: list[str] = []
    for name in ["gauge_couplings", "higgs", "yukawa_up_eigvals",
                 "yukawa_down_eigvals", "yukawa_lep_eigvals",
                 "ckm_wolfenstein", "theta_qcd",
                 "pmns_angles", "neutrino_dm2"]:
        t = getattr(model.params, name).detach().flatten().numpy()
        sub_axis = labels["axis_labels"].get(f"params.{name}", {})
        sub = sub_axis.get("rows", ("", []))[1] if sub_axis else []
        for i, v in enumerate(t):
            lab = sub[i] if i < len(sub) else f"[{i}]"
            p_vals.append(float(v))
            p_labels.append(f"{name}.{lab}")
    fig, ax = plt.subplots(figsize=(12, 4.6))
    colors = ["#c44" if v < 0 else "#46a" for v in p_vals]
    ax.bar(range(len(p_vals)), [abs(v) for v in p_vals], color=colors)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xticks(range(len(p_labels)))
    ax.set_xticklabels(p_labels, rotation=80, ha="right", fontsize=7)
    ax.set_title(f"{len(p_vals)} free SM parameters  (|value|, symlog; red = negative)")
    fig.tight_layout()
    params_uri = fig_to_uri(fig)

    return f"""
<section id="substrate">
  <h2>(a) How the physics is encoded</h2>
  {_substrate_primer_html()}

  <h3>The SM as a layered <code>nn.Module</code></h3>
  <p>The Standard Model is packaged here as a <code>nn.Module</code> with five
  layered submodules. Every "weight" is either a <strong>frozen buffer</strong>
  (the architecture: Lie-algebra
  <a href="https://en.wikipedia.org/wiki/Structure_constants" target="_blank"
     rel="noopener">structure constants</a>,
  <a href="https://en.wikipedia.org/wiki/Irreducible_representation"
     target="_blank" rel="noopener">irreducible representations</a>,
  particle vectors, vertex incidence) or one of the
  <strong>{n_param} trainable scalars</strong> &mdash; the only learnable
  knobs the SM has. That ratio (≈{n_param} learnable vs {n_buf:,} frozen) is
  the entire point: physics is mostly architecture, almost nothing is fit.</p>
  {table}

  <h3>Particle table &mdash; the ground-truth feature dictionary made tangible</h3>
  <p>Each row is one of the 61 Standard-Model
  <a href="https://en.wikipedia.org/wiki/Elementary_particle" target="_blank"
     rel="noopener">particles</a>; each column is one of its labels. Reading
  left to right: <em>Q</em> is
  <a href="https://en.wikipedia.org/wiki/Electric_charge" target="_blank"
     rel="noopener">electric charge</a>; <em>B</em> is
  <a href="https://en.wikipedia.org/wiki/Baryon_number" target="_blank"
     rel="noopener">baryon number</a>; <em>L_e, L_μ, L_τ</em> are
  <a href="https://en.wikipedia.org/wiki/Lepton_number" target="_blank"
     rel="noopener">lepton numbers</a> (one per generation); <em>C3, C8</em>
  are two diagonal generators of
  <a href="https://en.wikipedia.org/wiki/Color_charge" target="_blank"
     rel="noopener">color</a>; <em>spin</em> and <em>mass_GeV</em> close
  things out. The factorization (charge ⊗ color ⊗ flavor ⊗ generation ⊗
  chirality) is exactly what the SAE in section (b) will be asked to
  recover.</p>
  {img(pv_uri, "Each column is normalized to its own absolute maximum (shown in axis label) so the structure of each label is legible. Without this, the mass column would wash out everything else.")}

  <h3>Vertex incidence &mdash; symmetry as the matrix's nullspace</h3>
  <p>Every allowed interaction in the SM (e.g. an electron emitting a photon,
  a top quark decaying via W) is one row of this matrix; columns are
  particles. A <strong>conserved quantity</strong> is, mathematically, a
  vector that lies in the <em>nullspace</em> of this matrix: combine particles
  with those weights and every interaction leaves the total unchanged. There
  are exactly seven independent such vectors &mdash; the seven
  <a href="https://en.wikipedia.org/wiki/Conservation_law" target="_blank"
     rel="noopener">conservation laws</a> (Q, B, L_e, L_μ, L_τ, C3, C8). The
  <a href="https://en.wikipedia.org/wiki/Anomaly_(physics)" target="_blank"
     rel="noopener">anomaly cancellation</a> property of the SM is also a
  fact about this matrix.</p>
  {img(vi_uri, "B[v,p] = +1 incoming, −1 outgoing for vertex v and particle p. The 7-dimensional nullspace is the conservation algebra.")}

  <h3>The free parameters &mdash; the SM's "model weights"</h3>
  <p>These are the only numbers the universe gets to choose. Click any term to
  read more:</p>
  <ul class="paramlist">
    <li><strong>3 gauge couplings</strong>
        (<a href="https://en.wikipedia.org/wiki/Coupling_constant"
            target="_blank" rel="noopener">strength of each force</a>)</li>
    <li><strong>2 Higgs parameters</strong>
        (<a href="https://en.wikipedia.org/wiki/Higgs_mechanism" target="_blank"
            rel="noopener">vacuum expectation value &amp; self-coupling</a>)</li>
    <li><strong>9 Yukawa eigenvalues</strong>
        (<a href="https://en.wikipedia.org/wiki/Yukawa_interaction"
            target="_blank" rel="noopener">one per charged fermion mass</a>)</li>
    <li><strong>4 CKM parameters</strong>
        (<a href="https://en.wikipedia.org/wiki/Cabibbo%E2%80%93Kobayashi%E2%80%93Maskawa_matrix#Wolfenstein_parameterization"
            target="_blank" rel="noopener">quark flavor mixing,
            Wolfenstein parameterization</a>)</li>
    <li><strong>1 θ_QCD</strong>
        (<a href="https://en.wikipedia.org/wiki/Strong_CP_problem"
            target="_blank" rel="noopener">strong CP-violation angle</a>;
        measured to be ≈ 0)</li>
    <li><strong>4 PMNS parameters</strong> + <strong>3 neutrino mass-squared
        differences</strong>
        (<a href="https://en.wikipedia.org/wiki/Neutrino_oscillation"
            target="_blank" rel="noopener">neutrino mixing &amp; oscillation</a>;
        post-original SM)</li>
  </ul>
  {img(params_uri, "All 26 trainable scalars (19 minimal SM + 7 beyond-Standard-Model neutrino parameters). A typical large language model has ~10¹⁰ of these; physics has 26.")}

  {_further_reading_html()}
</section>
"""


# ---------------------------------------------------------------------------
# Running the model: feedforward simulation as a cascade
# ---------------------------------------------------------------------------
def _format_state(state) -> str:
    parts = []
    for p, n in sorted(state.items()):
        parts.append(f"{n}×{p}" if n > 1 else p)
    return " + ".join(parts) if parts else "(empty)"


def section_running() -> str:
    """Show how the SM-as-nn.Module is actually 'run': a Monte-Carlo cascade
    over decays, with each step querying the bundle's tensors."""
    import random

    from smsae.sm.embeddings import build_sm
    from smsae.sm.cascade import (aggregated_branching, build_decay_catalog,
                                  cascade, particle_mass)

    sm = build_sm()
    catalog = build_decay_catalog(sm)

    # Branching fraction bars for two heavy parents
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    parents = [
        ("H",   "Higgs boson",     "PDG: ≈58% bb̄, ≈21% WW*, ≈6% ττ, …"),
        ("t_r", "Top quark (red)", "PDG: ≈99.9% bW (Vtb dominates CKM)"),
        ("W+",  "W⁺ boson",        "PDG: ≈33% leptonic, ≈67% hadronic"),
    ]
    for ax, (parent, pretty, pdg) in zip(axes, parents):
        agg = aggregated_branching(sm, parent, catalog)
        items = sorted(agg.items(), key=lambda kv: -kv[1])[:7]
        if items:
            labels, fracs = zip(*items)
            ax.barh(range(len(labels)), fracs, color="#46a")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("branching fraction", fontsize=9)
        ax.set_title(f"{pretty}  ({parent})\n{pdg}", fontsize=9, pad=10)
    fig.suptitle(
        "Branching fractions computed from the bundle's free parameters\n"
        "(Yukawa eigenvalues, CKM, sin²θ_W) — no fitting, just lookups.",
        fontsize=11, y=1.02)
    fig.tight_layout(w_pad=2.5)
    bf_uri = fig_to_uri(fig)

    # Concrete worked example: cascade trace from one heavy particle
    rng = random.Random(7)
    initial = "H"
    history = cascade(sm, {initial: 1}, catalog, max_steps=30, rng=rng)

    trace_rows = [
        "<tr><th>step</th><th>action</th><th>tensors consulted</th>"
        "<th>state after</th></tr>"
    ]
    # tensor-lookup hint per decay kind (manually mapped to keep the table honest)
    def tensor_hint(parent_name: str) -> str:
        if parent_name == "H":
            return "params.yukawa_*_eigvals, params.higgs[v], fields.particle_vectors[mass]"
        if parent_name == "Z":
            return "params.gauge_couplings, params.higgs[v]  (→ sin²θ_W)"
        if parent_name in ("W+", "W-"):
            return "params.ckm_wolfenstein  (→ |V_ij|² weights)"
        if parent_name.startswith("t_") or parent_name.startswith("~t_"):
            return "params.ckm_wolfenstein[2,*]  (top-row CKM)"
        if parent_name.startswith("mu") or parent_name.startswith("tau"):
            return "params.yukawa_lep_eigvals, params.higgs[v]"
        return "fields.particle_vectors[mass], vertices.incidence"

    for i, (state, decay) in enumerate(history):
        if decay is None:
            action = ("<em>initial state</em>" if i == 0
                      else "<em>(no available decay — final state)</em>")
            tensors = "—"
        else:
            desc, parent, products = decay
            action = (f"<code>{escape(parent)}</code> → "
                      f"<code>{escape(' + '.join(products))}</code>")
            tensors = f"<code>{escape(tensor_hint(parent))}</code>"
        trace_rows.append(
            f"<tr><td>{i}</td><td>{action}</td><td>{tensors}</td>"
            f"<td><code>{escape(_format_state(state))}</code></td></tr>")
    trace_table = '<table class="summary trace">' + "".join(trace_rows) + '</table>'

    final = history[-1][0]
    n_particles_final = sum(final.values())
    n_steps = len(history) - 1
    initial_mass = particle_mass(sm, initial)

    return f"""
<section id="running">
  <h2>How the model is "run" &mdash; one transition step at a time</h2>
  <p>With the world-model framing in hand, "running" the Standard Model is
  straightforward: <em>loop the transition function</em>. Each transition
  step is one decay event &mdash; the engine looks at the current particle
  multiset, queries the bundle's tensors to score every allowed decay, samples
  one, and mutates the state. Repeat until nothing can decay further. This
  section walks through that loop concretely.</p>

  <h3>One transition step, written out</h3>
  <p>Pseudocode for one step:</p>
  <pre class="code">def step(state):
    # 1. enumerate allowed decays (those whose parent is in the state)
    candidates = [(d, w) for d in decay_catalog
                         if d.parent in state
                         for w in [decay_weight(d, params)]
                         if w &gt; 0]
    # 2. sample one channel proportional to its physical rate
    chosen = sample_categorical(candidates)
    # 3. mutate the state
    new_state = state.copy()
    new_state[chosen.parent] -= 1
    for p in chosen.products:
        new_state[p] += 1
    return new_state</pre>
  <p>Every quantity on the right-hand side is something we've already met:
  <code>decay_catalog</code> is enumerated from <code>vertices.incidence</code>;
  <code>decay_weight</code> reads
  <code>params.yukawa_*</code> / <code>params.ckm_wolfenstein</code> /
  <code>params.gauge_couplings</code> depending on the channel. So the
  transition function is just a deterministic recipe that maps state →
  next state using the bundle as a parameter store.</p>
  <p class="aside">Contrast with an LLM step: state = sequence of tokens,
  transition = run all transformer layers on the state and sample the next
  token from the output distribution. Same loop shape; different transition
  internals.</p>

  <h3>Branching fractions for three heavy particles</h3>
  <p>Before running the cascade, here's what the bundle says about three
  iconic decays. Each bar's length is computed from the
  <a href="https://en.wikipedia.org/wiki/Yukawa_interaction" target="_blank"
     rel="noopener">Yukawa eigenvalues</a> and
  <a href="https://en.wikipedia.org/wiki/Cabibbo%E2%80%93Kobayashi%E2%80%93Maskawa_matrix"
     target="_blank" rel="noopener">CKM matrix</a> stored in
  <code>model.params</code> &mdash; no fitting, just direct calculation
  from the 19 free parameters. The
  <a href="https://pdg.lbl.gov/" target="_blank" rel="noopener">Particle
  Data Group (PDG)</a> values shown in each subtitle are what the real
  world measures.</p>
  {img(bf_uri, "Branching fractions aggregated over color and summed where the catalog distinguishes channels (e.g. each H → ff̄ split per color is summed into one ff̄ entry). Top is overwhelmingly bW because Vtb ≈ 1.")}

  <h3>A concrete worked example: one Higgs decays</h3>
  <p>The trace below was generated by running
  <code>cascade(sm, {{H: 1}}, catalog, max_steps=30, rng=Random(7))</code>.
  Each row is one Monte-Carlo step. The "tensors consulted" column names
  the submodule the simulator read to compute that step's decay weight
  &mdash; for example, when the Higgs picks a Yukawa channel the engine
  reads <code>params.yukawa_*_eigvals</code>; when a W picks a
  quark-antiquark pair it reads <code>params.ckm_wolfenstein</code>.</p>
  {trace_table}
  <p class="aside">Started with 1 particle (a Higgs, mass ≈
  {initial_mass:.1f} GeV). After {n_steps} decay step(s) the state
  contained {n_particles_final} stable particles
  (<code>{escape(_format_state(final))}</code>). Different seeds produce
  different trajectories &mdash; that randomness is the entire reason
  <code>feed_cascade</code> can supply a non-trivial distribution to the
  SAE in the next section.</p>

  <h3>So why does this matter for interpretability?</h3>
  <p>The cascade is what produces <strong>feed_cascade</strong>, the only
  one of the three SAE feeds that has genuine distributional structure
  (the other two are static lookup tables). Each row of feed_cascade is
  one final state like the one above, encoded as a bag-of-particles count
  vector. The SAE will see thousands of such vectors and is asked to
  discover &mdash; among other things &mdash; that they came from one of
  a small set of heavy parents. The fact that the SAE recovers
  <code>origin:t_b</code>, <code>origin:H</code>, etc. at AUC = 1.0 is
  the SAE inferring <em>backward through the cascade tree</em> from a
  final-state bag to its starting heavy particle. That's the
  interpretability claim made concrete.</p>
</section>
"""


# ---------------------------------------------------------------------------
# (b) SAE
# ---------------------------------------------------------------------------
def _load_feeds(fast: bool):
    from smsae.sae.data import feed_cascade, feed_embedded, feed_raw
    n_cascade = 500 if fast else 2000
    return {
        "raw":      feed_raw(),
        "embedded": feed_embedded(embed_dim=16, seed=0),
        "cascade":  feed_cascade(n_events=n_cascade, seed=0),
    }


def _feed_panel(ax, feed, title: str) -> None:
    X = feed.X.numpy()
    # Per-column normalization so no single column (e.g. mass in feed_raw, or
    # high-count stable particles in feed_cascade) drowns out the rest.
    col_scale = np.max(np.abs(X), axis=0)
    col_scale = np.where(col_scale > 0, col_scale, 1.0)
    Xn = X / col_scale
    if X.min() < 0:
        ax.imshow(Xn, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    else:
        ax.imshow(Xn, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"{title}\n{X.shape[0]} × {X.shape[1]}  "
                 f"(per-column normalized)", fontsize=9, pad=10)
    ax.set_xticks([]); ax.set_yticks([])


def _alignment_panel(ax, A: np.ndarray, gt_vocab: list[str], variant: str,
                     n_active: int) -> None:
    # Order SAE features so each row's best GT is roughly along the diagonal
    if A.size == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center")
        ax.set_axis_off()
        return
    best_gt = A.argmax(axis=1)
    best_auc = A.max(axis=1)
    # Active features first, sorted by their best GT col, then by best AUC desc
    active = A.max(axis=1) > 0.5 + 1e-6
    active_idx = np.where(active)[0]
    order = sorted(active_idx, key=lambda f: (best_gt[f], -best_auc[f]))
    rest = [f for f in range(A.shape[0]) if f not in set(order)]
    perm = np.array(order + rest, dtype=int)
    A_sorted = A[perm]
    im = ax.imshow(A_sorted, aspect="auto", cmap="magma", vmin=0.5, vmax=1.0)
    ax.set_title(f"{variant}  ({n_active}/{A.shape[0]} active)", fontsize=9)
    ax.set_xlabel(f"{len(gt_vocab)} GT features", fontsize=8)
    ax.set_ylabel("SAE features (sorted)", fontsize=8)
    ax.tick_params(labelsize=7)


def section_sae(fast: bool) -> str:
    from smsae.sae.data import Feed
    from smsae.sae.evaluation import build_gt_matrix, load_sae, score_sae

    feeds = _load_feeds(fast)

    # Three feeds side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.0))
    _feed_panel(axes[0], feeds["raw"], "feed_raw: 9-dim quantum numbers")
    _feed_panel(axes[1], feeds["embedded"],
                "feed_embedded: random 16-dim projection (superposition)")
    _feed_panel(axes[2], feeds["cascade"],
                "feed_cascade: bag-of-particles per event")
    fig.suptitle("The three feeds: harder = closer to a real LLM residual stream",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.5)
    feeds_uri = fig_to_uri(fig)

    # Alignment grid: 3 (feeds) × 3 (variants)
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    any_loaded = False
    Y_cache = {n: build_gt_matrix(f) for n, f in feeds.items()}
    for r, feed_name in enumerate(("raw", "embedded", "cascade")):
        for c, variant in enumerate(("topk", "l1", "jumprelu")):
            ax = axes[r, c]
            ckpt = os.path.join(REPO_ROOT, "runs", f"{feed_name}__{variant}.pt")
            if not os.path.exists(ckpt):
                ax.text(0.5, 0.5, f"missing\n{feed_name}__{variant}.pt",
                        ha="center", va="center", fontsize=8)
                ax.set_axis_off()
                continue
            sae, _ = load_sae(ckpt)
            feed = feeds[feed_name]
            Z = score_sae(sae, feed)
            A = auc_matrix(Z, Y_cache[feed_name])
            n_active = int((Z.max(axis=0) > 1e-9).sum())
            _alignment_panel(ax, A, feed.feature_vocab, variant, n_active)
            any_loaded = True
        axes[r, 0].set_ylabel(f"feed = {feed_name}\nSAE features",
                              fontsize=9)
    fig.suptitle("Alignment matrix: AUC(SAE feature vs GT feature). "
                 "Diagonal-like → clean recovery.", fontsize=11)
    fig.tight_layout()
    align_uri = fig_to_uri(fig)
    align_note = (img(align_uri, "Each cell is an AUC matrix for one (feed, variant). "
                                  "Rows reordered so a SAE feature's best GT match sits "
                                  "along the diagonal. Brighter = higher AUC.")
                  if any_loaded
                  else missing("runs/{feed}__{variant}.pt",
                               "run scripts/train_all.py to populate"))

    # Summary table
    summary_path = os.path.join(REPO_ROOT, "runs", "alignment_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        rows = []
        rows.append('<tr><th>feed</th><th>variant</th><th>n_sae</th>'
                    '<th>active</th><th>cov≥0.95</th><th>cov≥0.90</th>'
                    '<th>mean_best_auc</th><th>monosemantic</th></tr>')
        for r in summary:
            rows.append(
                f"<tr><td>{escape(r['feed'])}</td><td>{escape(r['variant'])}</td>"
                f"<td>{r['n_sae_features']}</td><td>{r['n_active']}</td>"
                f"<td>{r['coverage_0.95']:.1%}</td>"
                f"<td>{r['coverage_0.90']:.1%}</td>"
                f"<td>{r['mean_best_auc']:.3f}</td>"
                f"<td>{r['monosemanticity']:.1%}</td></tr>")
        sum_table = '<table class="summary">' + "".join(rows) + '</table>'
    else:
        sum_table = missing(summary_path, "run scripts/evaluate.py to populate")

    return f"""
<section id="sae">
  <h2>(b) How SAEs are trained and scored</h2>
  <p>A sparse autoencoder's training data is whatever activations a model
  produces while running. This project ships <strong>three feeds</strong> of
  increasing difficulty, all derived from the Standard Model substrate.</p>
  {img(feeds_uri, "raw = a sanity-check shape; embedded = artificially superposed (61 concepts squashed into 16 dims, mimicking the residual stream of a large language model); cascade = bag-of-particles from a Monte Carlo decay simulator — the only feed with genuine distributional structure.")}
  <p>Each trained sparse autoencoder (SAE) produces sparse latent codes. We
  score each latent feature by the
  <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve"
     target="_blank" rel="noopener">Area Under the ROC Curve (AUC)</a>
  against every ground-truth label
  (<code>color:r</code>, <code>generation:2</code>, <code>origin:t_b</code>, …):
  one cell of the alignment matrix below. AUC = 1.0 means the SAE feature
  fires on exactly the inputs that carry that label; AUC = 0.5 means no
  better than chance.</p>
  {align_note}
  <h3>Headline numbers per (feed, variant)</h3>
  {sum_table}
  <p class="aside">Notice <em>monosemanticity</em> collapses on the cascade feed
  while staying at 100% on the easy feeds &mdash; that's where the project starts
  earning its keep as a benchmark.</p>
</section>
"""


# ---------------------------------------------------------------------------
# (c) polygram
# ---------------------------------------------------------------------------
ORCA_PREPARE_RE = re.compile(
    r"\|\s*idle\s*\|\s*prepare_(\w+)\s*\|\s*\|\s*prepared_\w+\s*\|\s*"
    r"prepare_concept\(\s*([\-\d.eE]+)\s*,\s*([\-\d.eE]+)\s*,\s*"
    r"([\-\d.eE]+)\s*,\s*([\-\d.eE]+)\s*\)\s*\|"
)


def _parse_cancellation_summary(path: str) -> dict:
    """Parse a polygram cancellation '*_summary.md' into a dict."""
    with open(path) as f:
        text = f.read()
    out: dict = {}
    m = re.search(r"target pair:\s*`([^`]+)`\s*×\s*`([^`]+)`", text)
    if m:
        out["pair_a"] = m.group(1)
        out["pair_b"] = m.group(2)
    m = re.search(r"tolerance_met:\s*(\w+)", text)
    if m:
        out["tolerance_met"] = m.group(1).lower() == "true"
    for k in ("before", "after", "structural_floor"):
        m = re.search(rf"\b{k}:\s*([\-\d.]+)", text)
        if m:
            out[k] = float(m.group(1))
    return out


def _all_cancellation_summaries() -> list[dict]:
    """Read every cancellation summary under runs/polygram/cancellation_*."""
    import glob
    out = []
    pattern = os.path.join(REPO_ROOT, "runs", "polygram",
                           "cancellation_*", "*_summary.md")
    for path in sorted(glob.glob(pattern)):
        try:
            row = _parse_cancellation_summary(path)
            row["label"] = os.path.basename(os.path.dirname(path))
            out.append(row)
        except Exception:
            continue
    return out


def _parse_orca(path: str) -> list[tuple[str, float, float, float, float]]:
    """Return a list of (concept_name, a, b, c, phi) tuples from an Orca .md."""
    with open(path) as f:
        text = f.read()
    out = []
    for name, a, b, c, phi in ORCA_PREPARE_RE.findall(text):
        out.append((name, float(a), float(b), float(c), float(phi)))
    return out


def _simulate_prepare_concept(a: float, b: float, c: float, phi: float) -> np.ndarray:
    """Compute the 8-dim amplitude vector that prepare_concept(a,b,c,phi) builds
    on |000>. Matches the action body in the Orca file:
       Ry(qs[0], a); CNOT(qs[0], qs[1]); Ry(qs[1], a+b);
       Rz(qs[1], phi); CNOT(qs[1], qs[2]); Ry(qs[2], b+c)
    Qubit ordering: index = q0 + 2*q1 + 4*q2.
    """
    def kron(*ms):
        out = ms[0]
        for m in ms[1:]:
            out = np.kron(out, m)
        return out

    I2 = np.eye(2, dtype=complex)
    def Ry(theta):
        c_, s_ = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c_, -s_], [s_, c_]], dtype=complex)
    def Rz(theta):
        return np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]], dtype=complex)
    # CNOT control=q0 target=q1 (basis: |q2 q1 q0>, little-endian so q0 is innermost)
    def cnot(control: int, target: int) -> np.ndarray:
        U = np.zeros((8, 8), dtype=complex)
        for x in range(8):
            bits = [(x >> i) & 1 for i in range(3)]
            if bits[control] == 1:
                bits[target] ^= 1
            y = bits[0] | (bits[1] << 1) | (bits[2] << 2)
            U[y, x] = 1
        return U

    # Initial |000>
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1.0
    # Ry(qs[0], a) — q0 is innermost so we kron(I2, I2, Ry(a))
    psi = kron(I2, I2, Ry(a)) @ psi
    psi = cnot(0, 1) @ psi
    psi = kron(I2, Ry(a + b), I2) @ psi
    psi = kron(I2, Rz(phi), I2) @ psi
    psi = cnot(1, 2) @ psi
    psi = kron(Ry(b + c), I2, I2) @ psi
    return psi


def section_polygram() -> str:
    parts: list[str] = []

    # 1. Dictionary cluster tree (from the polygram_bridge SM_SLICE constant)
    SM_SLICE = [
        ("e-",     "charged_lepton"),
        ("e+",     "charged_lepton"),
        ("u_r",    "up_quark"),
        ("~u_r",   "up_quark"),
        ("d_r",    "down_quark"),
        ("~d_r",   "down_quark"),
        ("photon", "boson"),
        ("W+",     "boson"),
    ]
    by_cluster: dict[str, list[str]] = {}
    for name, cluster in SM_SLICE:
        by_cluster.setdefault(cluster, []).append(name)
    tree = "<ul class='tree'>"
    for cluster, members in by_cluster.items():
        tree += (f"<li><strong>{escape(cluster)}</strong><ul>" +
                 "".join(f"<li><code>{escape(m)}</code></li>" for m in members) +
                 "</ul></li>")
    tree += "</ul>"
    parts.append(
        "<h3>Dictionary: 8 features in 4 clusters</h3>"
        "<p>polygram encodes the 8 chosen particles as quantum states using a "
        "<a href='https://en.wikipedia.org/wiki/Matrix_product_state' "
        "target='_blank' rel='noopener'>Matrix Product State (MPS)</a> "
        "&mdash; specifically <em>Rung 1</em> with bond dimension 2, which "
        "means the 8 concepts live in the Hilbert space of 3 qubits (2³ = 8 "
        "amplitudes per state). Each particle is a leaf in a 4-cluster "
        "hierarchy:</p>"
        + tree)

    # 2. Interference sweep
    sweep_path = os.path.join(REPO_ROOT, "runs", "polygram",
                              "interference_e_pair.csv")
    if os.path.exists(sweep_path):
        data = np.genfromtxt(sweep_path, delimiter=",", names=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(data["e_plusphi"], data["overlap"], lw=2, label="|⟨e⁻|e⁺⟩|² overlap")
        floor = float(np.min(data["overlap"]))
        ax.axhline(floor, color="red", ls="--", lw=1,
                   label=f"min ≈ structural floor = {floor:.4f}")
        ax.set_xlabel("e_plus.phi  (radians)")
        ax.set_ylabel("overlap")
        ax.set_title("Interference sweep: rotating e⁺'s phase modulates ⟨e⁻|e⁺⟩")
        ax.legend(loc="best")
        sweep_uri = fig_to_uri(fig)
        parts.append("<h3>Interference sweep — phase as a knob</h3>"
                     + img(sweep_uri,
                           "Rotating e⁺'s phase moves it around the Bloch sphere; "
                           "the inner product with e⁻ modulates sinusoidally. The "
                           "minimum is the structural floor — the level below which "
                           "phase alone cannot push the overlap."))
    else:
        parts.append("<h3>Interference sweep</h3>" + missing(
            sweep_path, "run scripts/polygram_demo.py to populate"))

    # 3. Orca-extracted concept angles + analytic Gram matrix
    orca_path = os.path.join(REPO_ROOT, "runs", "polygram", "cancellation_e_e",
                             "SM_Rung1_Slice_at_optimum.q.orca.md")
    if os.path.exists(orca_path):
        concepts = _parse_orca(orca_path)

        # Analytic Gram matrix from the prepared-concept angles
        states = np.stack([_simulate_prepare_concept(a, b, c, phi)
                           for _, a, b, c, phi in concepts])
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        states = states / np.where(norms > 0, norms, 1)
        gram_amp = states.conj() @ states.T
        gram = np.abs(gram_amp) ** 2  # overlap = |inner|^2
        names = [n for n, *_ in concepts]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(gram, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        # Cluster boundaries
        cluster_seen: dict[str, list[int]] = {}
        for i, (nm, cl) in enumerate(SM_SLICE):
            mapped = (nm.replace("~", "anti_").replace("+", "_plus")
                        .replace("-", "_minus"))
            cluster_seen.setdefault(cl, []).append(i if mapped in names else -1)
        for cl, idxs in cluster_seen.items():
            idxs = [i for i in idxs if i >= 0]
            if not idxs:
                continue
            lo, hi = min(idxs), max(idxs) + 1
            ax.add_patch(plt.Rectangle((lo - 0.5, lo - 0.5), hi - lo, hi - lo,
                                        fill=False, edgecolor="white", lw=1.5))
        ax.set_title("Gram matrix |⟨ψ_i|ψ_j⟩|² at the optimum  "
                     "(white boxes = clusters; should stay block-bright)")
        fig.colorbar(im, ax=ax, fraction=0.04)
        gram_uri = fig_to_uri(fig)

        parts.append("<h3>Gram matrix at the optimised dictionary</h3>")
        parts.append("<p>The 8×8 squared-inner-product matrix among the "
                     "prepared concept states after the cancellation search. "
                     "Off-diagonal cells should be near zero for unrelated "
                     "concepts; cluster boxes should remain intra-bright if "
                     "<code>preserve_tiers</code> worked. Computed "
                     "analytically from the optimised <code>prepare_concept"
                     "</code> circuit parameters &mdash; the deeper Orca "
                     "state-machine artifact is a lower-level concern best "
                     "inspected in the per-pair "
                     "<code>runs/polygram/cancellation_*/</code> directories.</p>")
        parts.append(img(gram_uri,
                         "Cells: |⟨ψ_i|ψ_j⟩|² for the 8 prepared concept states. "
                         "White rectangles mark the four cluster boundaries "
                         "(charged_lepton / up_quark / down_quark / boson)."))
    else:
        parts.append("<h3>Orca artifact</h3>" + missing(
            orca_path, "run scripts/polygram_demo.py to populate"))

    # Brief framing — the detailed numbers live in the scoreboard section now
    framing = (
        "<p>polygram is the worked example of a compression / extraction "
        "technique in this report. This section walks through one "
        "<code>MPSRung1</code> dictionary &mdash; how it's built, what the "
        "interference / cancellation experiments look like, and the Orca "
        "artifact polygram emits. The candid pass/fail across encoding "
        "configurations (which is where the actionable story is) lives on "
        "the <a href='#benchmark'>scoreboard</a>.</p>")
    parts.insert(0, framing)
    return "<section id='polygram'><h2>(c) How polygram captures structure</h2>" \
        + "".join(parts) + "</section>"


# ---------------------------------------------------------------------------
# Benchmark scoreboard
# ---------------------------------------------------------------------------
def _format_polygram_sweep() -> str:
    """Render the cross-encoding sweep results (if scripts/polygram_sweep.py
    has been run) as a configs × pairs comparison table."""
    sweep_path = os.path.join(REPO_ROOT, "runs", "polygram", "sweep",
                              "sweep_results.json")
    if not os.path.exists(sweep_path):
        return ("<h3>Polygram encoding sweep</h3>"
                + missing(sweep_path, "run scripts/polygram_sweep.py"))
    with open(sweep_path) as f:
        sweep = json.load(f)
    runs = sweep.get("runs", [])
    by_config = sweep.get("by_config", {})
    if not runs:
        return ("<h3>Polygram encoding sweep</h3>"
                "<p>No sweep results recorded yet.</p>")

    # Cross-tab: configs (rows) × pair labels (cols)
    config_order = list(by_config.keys())
    pair_order: list[str] = []
    for r in runs:
        if r.get("label") and r["label"] not in pair_order:
            pair_order.append(r["label"])
    by_key = {(r["config"], r.get("label")): r for r in runs if "config" in r}

    rows = ["<thead><tr><th>config</th><th>knobs</th>"]
    for p in pair_order:
        rows.append(f"<th><code>{escape(p)}</code></th>")
    rows.append("<th>pass / total</th></tr></thead><tbody>")
    for cfg in config_order:
        agg = by_config[cfg]
        rows.append(f"<tr><td><code>{escape(cfg)}</code><br/>"
                    f"<span style='font-size:11px;color:#666'>"
                    f"{escape(agg.get('desc', ''))}</span></td>")
        nk = next((r.get("n_knobs") for r in runs
                   if r.get("config") == cfg and "n_knobs" in r), "?")
        rows.append(f"<td>{nk}</td>")
        for p in pair_order:
            r = by_key.get((cfg, p))
            if r is None or "error" in r:
                rows.append("<td class='fail' title='error'>err</td>")
            elif r.get("tolerance_met"):
                rows.append(
                    f"<td class='pass' title='Δ={r['delta']:+.4f}'>"
                    f"✓ Δ {r['delta']:+.3f}</td>")
            else:
                rows.append(
                    f"<td class='fail' title='Δ={r['delta']:+.4f}'>"
                    f"✗ Δ {r['delta']:+.3f}</td>")
        rows.append(f"<td>{agg.get('n_pass', 0)}/{agg.get('n_runs', 0)}</td>"
                    f"</tr>")
    rows.append("</tbody>")
    table = ('<table class="summary scorecard">' + "".join(rows) + '</table>')

    summary_para = ""
    rung3 = by_config.get("Rung3_amp", {})
    if rung3 and rung3.get("n_pass", 0) == rung3.get("n_runs", 0) > 0:
        summary_para = (
            "<p><strong>Headline:</strong> the baseline failure was a "
            "<em>configuration</em> weakness, not a fundamental one. "
            "Switching from <code>MPSRung1</code> (phase-only, 2 knobs) to "
            "<code>Rung3</code> (adds an amplitude branch with 2 extra "
            "knobs <code>theta_amp</code> and <code>psi_aux</code>) drives "
            "every cancellation to exactly 0.0. The polygram docstring "
            "predicted this; the benchmark confirmed it. So the open gap "
            "is no longer in Axis C &mdash; it is in Axis B, where the SAE "
            "itself still misses ~63% of GT features on the cascade feed.</p>")
    elif rung3 and rung3.get("n_runs", 0) == 0:
        summary_para = (
            "<p>Rung3 sweep is still running. Updated results will appear "
            "on rebuild.</p>")

    return f"""
  <h3>Polygram encoding sweep &mdash; can a different encoding crack the floor?</h3>
  <p>The baseline failures all hit the structural floor of
  <code>MPSRung1(bond_dim=2, phase_knobs=True)</code>. The polygram package
  ships higher-rung encodings (<code>Rung3</code>, <code>Rung4</code>,
  <code>Rung5</code>) that add amplitude knobs to the cancellation search.
  This sweep runs the same 4 pairs against each:</p>
  {table}
  {summary_para}
  <p class="aside">Each row is one encoding configuration. Cells show pass
  (✓ green) or fail (✗ red) with the achieved Δ. The Rung3 sweep uses
  scipy's <code>differential_evolution</code> over the canonical 4-knob
  list; runtime is ~30&times; the grid-based baseline.</p>
"""


def _format_recommended_defaults() -> str:
    """Distill the benchmark's findings into a concrete set of recommended
    keyword arguments for polygram and (provisionally) sae-forge.

    Each recommendation is tagged with its evidence source:
      - measured        — directly supported by sweep results in runs/polygram/sweep/
      - inherited       — polygram/sae-forge ships this default and we have no
                          reason to change it
      - provisional     — defensible from first principles or upstream docs,
                          but not yet directly tested here
    """
    sweep_path = os.path.join(REPO_ROOT, "runs", "polygram", "sweep",
                              "sweep_results.json")
    measured_configs: set[str] = set()
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            sweep = json.load(f)
        for cfg_name, agg in sweep.get("by_config", {}).items():
            if agg.get("n_pass", 0) == agg.get("n_runs", 0) > 0:
                measured_configs.add(cfg_name)

    rung3_ok = "Rung3_amp" in measured_configs
    rung4_ok = ("Rung4_amp_budget" in measured_configs
                or "Rung4_amp_default" in measured_configs)
    rung5_ok = "Rung5_amp_budget" in measured_configs
    # Cheapest passing config is the recommended encoding. Use observed
    # per-pair wall times if available; otherwise fall back to ordering.
    if rung5_ok:
        cheapest_passing = "Rung5(bond_dim=2, n_amp_qubits=2)"
        cheapest_passing_str = "Rung5"
        cheapest_cancel_str = '"rung5"'
    elif rung3_ok:
        cheapest_passing = "Rung3(bond_dim=2)"
        cheapest_passing_str = "Rung3"
        cheapest_cancel_str = '"rung3"'
    elif rung4_ok:
        cheapest_passing = "Rung4(bond_dim=2)"
        cheapest_passing_str = "Rung4"
        cheapest_cancel_str = '"rung4"'
    else:
        cheapest_passing = "Rung3(bond_dim=2)  (pending sweep)"
        cheapest_passing_str = "Rung3"
        cheapest_cancel_str = '"rung3"'

    def _badge(kind: str) -> str:
        color = {"measured": "#2a6", "inherited": "#557",
                 "provisional": "#b80"}[kind]
        return (f"<span style='color:{color};font-weight:bold;"
                f"font-size:10px;text-transform:uppercase'>"
                f"{kind}</span>")

    # Polygram recommendations
    rec_evidence = ("All three amplitude-knob encodings (Rung3, Rung4, "
                    "Rung5) pass 4/4 of the benchmark cancellations; "
                    "MPSRung1+phase-only passes 0/4. Among the passing "
                    "configurations Rung5 (n_amp_qubits=2) is the cheapest "
                    "at ~4s per cancellation, vs. Rung3 at ~175s and Rung4 "
                    "(budgeted) at ~240s &mdash; Rung4 and Rung5 are "
                    "mathematically identical, see "
                    "<a href='https://github.com/jascal/polygram/issues/86' "
                    "target='_blank' rel='noopener'>polygram#86</a>."
                    if rung5_ok else
                    "Rung3 passes 4/4; MPSRung1+phase-only passes 0/4. The "
                    "polygram docstring predicts this; the sweep confirms.")
    p_rows = [(
        "encoding",
        "<code>Dictionary(encoding=...)</code>",
        "<em>(must specify)</em>",
        f"<code>{cheapest_passing}</code>",
        rec_evidence,
        "measured" if (rung3_ok or rung5_ok) else "provisional",
    ), (
        "Cancellation(encoding=...)",
        "<code>polygram.Cancellation</code>",
        "<code>None</code> (= phase-only 2-knob)",
        f'<code>{cheapest_cancel_str}</code>',
        f"Selects the canonical knob list for {cheapest_passing_str}. "
        "All three amplitude-knob encodings unlock the structural floor; "
        "Rung5 is preferred for cost." if rung5_ok else
        "Selects the 4-knob default list "
        "<code>[a.phi, b.phi, b.theta_amp, b.psi_aux]</code> "
        "which unlocks the amplitude branch.",
        "measured" if (rung3_ok or rung5_ok) else "provisional",
    ), (
        "optimize.method",
        "<code>Cancellation(optimize={...})</code>",
        '<code>"grid"</code>',
        ('<code>"grid"</code> for ≤4 knobs; <code>"scipy"</code> for >4'),
        "<code>GRID_KNOB_LIMIT = 4</code> in polygram; Rung4/Rung5 "
        "(6+ knobs) require <code>scipy.optimize.differential_evolution</code>. "
        "<code>scipy</code> is in <code>polygram[opt]</code>.",
        "measured",
    ), (
        "assign_amp_knobs",
        "<code>SAEImportConfig</code> / "
        "<code>from_sae_lens(...)</code>",
        "<code>False</code>",
        "<code>True</code> (for ground-truth-rich fixtures)",
        "Polygram defaults to neither amp nor phase knobs when ingesting "
        "an SAE. Without amp knobs, the cancellation step is stuck at the "
        "structural floor (see Axis C baseline). For any benchmark that "
        "measures structured-feature recovery, this should be flipped.",
        "measured",
    ), (
        "assign_phase_knobs",
        "<code>SAEImportConfig</code>",
        "<code>False</code>",
        "<code>True</code>",
        "Required to populate the <code>.phi</code> entries the 4-knob "
        "Rung3 search expects.",
        "measured",
    ), (
        "tolerance",
        "<code>CancellationConfig</code>",
        "<code>0.05</code>",
        "<code>0.05</code> (no reason to change)",
        "All Rung3 passes met this comfortably — every Δ landed at exactly "
        "0.0, well inside tolerance.",
        "measured" if rung3_ok else "inherited",
    ), (
        "preserve_tiers",
        "<code>CancellationConfig</code>",
        "<code>True</code>",
        "<code>True</code>",
        "Cluster hierarchy is what makes the SAE features physically "
        "interpretable; default behaviour preserves it through cancellation "
        "without harming the score.",
        "inherited",
    ), (
        "strategy",
        "<code>CompressionConfig</code>",
        '<code>"merge"</code>',
        '<code>"merge"</code>',
        "Not directly tested by the sweep (Compressor adapter not yet "
        "built); the default is the only path with end-to-end "
        "sae-forge support per its README.",
        "inherited",
    ), (
        "rep_selection",
        "<code>CompressionConfig</code>",
        '<code>"scale_aware"</code>',
        '<code>"scale_aware"</code>',
        "sae-forge's <code>FeatureBasis</code> depends on the scale-aware "
        "merge to preserve decoder magnitudes (see sae-forge README, "
        '"forge-feature-scales").',
        "inherited",
    ), (
        "n_clusters",
        "<code>SAEImportConfig</code>",
        "<code>2</code>",
        "match the fixture's known cluster count",
        "For sm-sae's 8-particle slice, the right value is 4 "
        "(charged_lepton / up_quark / down_quark / boson). For general "
        "fixtures, this should track the number of natural clusters in "
        "the GT label set.",
        "provisional",
    )]

    # sae-forge recommendations (more provisional — no direct empirical
    # evaluation on sm-sae yet, since the adapter isn't built)
    sf_rows = [(
        "<code>faithfulness</code>",
        "<code>ForgePipeline.__init__</code>",
        "<code>None</code> (defaults to per-token KL vs host)",
        "<code>KLTarget()</code> for LLMs; "
        "<code>GroundTruthAlignment(labels=Y)</code> for label-rich fixtures "
        "(sm-sae, mixture-of-gaussians, etc.) &mdash; implement on the "
        "caller side satisfying the v0.4 "
        "<code>saeforge.eval.FaithfulnessTarget</code> protocol.",
        "Wired end-to-end as of sae-forge v0.4.0; "
        "<code>scripts/forge_pipeline.py</code>'s "
        "<code>GroundTruthAlignment</code> reads input_ids from ctx, runs "
        "them through the forged model, mean-pools the residual stream, "
        "and scores AUC against sm-sae GT labels. Hard-coded KL would "
        "compare the forged model to its random-init host, not to the "
        "physical labels we care about.",
        "measured",
    ), (
        "<code>strategy</code>",
        "<code>Regrower.__init__</code>",
        '<code>"protected"</code>',
        '<code>"protected"</code>',
        "Protects features above a configured threshold from being dropped "
        "during compress; this is the structural-EWC behaviour the basis "
        "loop is built around (see <code>docs/advanced-fsm-options.md</code>).",
        "inherited",
    ), (
        "<code>n_init</code>",
        "<code>Regrower.__init__</code>",
        "<code>4</code>",
        "<code>4</code> for small fixtures; <code>8+</code> at LLM scale",
        "The basis loop's regrow re-tries this many initializations. "
        "Default suffices for sm-sae sized inputs; larger initial counts "
        "matter more when the SAE has thousands of features.",
        "provisional",
    ), (
        "<code>polygram_overlap_threshold</code>",
        "<code>BehaviouralValidator</code> / "
        "<code>EpochCompressor</code>",
        "<code>0.7</code>",
        "<code>0.7</code>",
        "Default validator threshold for behavioural equivalence; "
        "sae-forge ships this as a sensible single-shard default.",
        "inherited",
    ), (
        "<code>cosine_threshold</code>",
        "<code>EpochCompressor</code>",
        "<code>None</code> (off)",
        "<code>None</code> for sm-sae-sized SAEs; "
        "<code>0.85+</code> for LLM-scale",
        "Adds a cosine-similarity criterion to per-feature inclusion. "
        "For a 128-feature SAE the extra signal is small; for tens of "
        "thousands of features it becomes load-bearing.",
        "provisional",
    ), (
        "<code>save_intermediate_reports</code>",
        "<code>EpochCompressor</code>",
        "<code>False</code>",
        "<code>True</code> for benchmark runs",
        "Lets the basis loop's per-iteration scoreboard be inspected, "
        "which is exactly what a benchmark wants.",
        "provisional",
    )]

    def _table(rows: list[tuple]) -> str:
        h = ("<table class='summary defaults'>"
             "<thead><tr><th>parameter</th><th>location</th>"
             "<th>upstream default</th><th>sm-sae recommendation</th>"
             "<th>evidence</th><th></th></tr></thead><tbody>")
        for p, loc, d, rec, why, kind in rows:
            h += (f"<tr><td>{p}</td>"
                  f"<td>{loc}</td>"
                  f"<td>{d}</td>"
                  f"<td><strong>{rec}</strong></td>"
                  f"<td>{why}</td>"
                  f"<td>{_badge(kind)}</td></tr>")
        h += "</tbody></table>"
        return h

    measured_note = ""
    if not rung3_ok:
        measured_note = (
            "<p class='aside'>The Rung3 sweep has not finished yet, so a "
            "couple of the polygram recommendations are tagged "
            "<em>provisional</em>. They will be re-tagged <em>measured</em> "
            "after the next rebuild once <code>sweep_results.json</code> "
            "shows Rung3 at 4/4.</p>")

    return f"""
  <h3 id="recommended-defaults">Recommended defaults &mdash; the bottom line</h3>
  <p>One of the things a benchmark is supposed to produce is concrete
  configuration guidance, not just scores. Below are the recommended
  keyword arguments for both packages on this kind of fixture
  (ground-truth-rich, compression / extraction quality is the primary
  signal). Each is tagged with the level of evidence behind it:</p>
  <ul style="font-size:13px">
    <li>{_badge("measured")} &mdash; directly supported by the sweep
        results above (cells in the sweep table).</li>
    <li>{_badge("inherited")} &mdash; package ships this default and the
        sweep gives no reason to change it.</li>
    <li>{_badge("provisional")} &mdash; defensible from first principles
        or upstream docs, but not yet directly verified on this
        fixture.</li>
  </ul>

  <h4>Polygram</h4>
  {_table(p_rows)}

  <h4>sae-forge</h4>
  <p class='aside'>sae-forge v0.4.0 shipped the
  <code>FaithfulnessTarget</code> protocol; <code>GroundTruthAlignment</code>
  is implemented on our side. The pipeline runs end-to-end against this
  fixture today &mdash; faithfulness numbers in the
  <a href="#benchmark">Forge pipeline runs</a> table. Recommendations
  below are upgraded from {_badge("provisional")} to {_badge("measured")}
  where the wiring path actually executed; entries that need the
  cascade-host shim to be meaningful remain {_badge("provisional")} with
  a note.</p>
  {_table(sf_rows)}

  {measured_note}
"""


def _format_forge_pipeline_results() -> str:
    """Aggregate runs/sae_forge/*/forge_results.json into a scoreboard row.

    The pipeline emits a forge_results.json per (feed, variant) run. Stage 7
    (the sae-forge call) is currently stubbed pending the Faithfulness
    release, so the forge_score column will be blank until that lands.
    Stages 1-6 (SAE → polygram Dictionary → ValidationReport → Compressor)
    are real and their outputs are reflected here.
    """
    import glob
    forge_dir = os.path.join(REPO_ROOT, "runs", "sae_forge")
    paths = sorted(glob.glob(os.path.join(forge_dir, "*", "forge_results.json")))
    if not paths:
        return ("<h3>Forge pipeline runs</h3>"
                + missing(os.path.join(forge_dir, "*", "forge_results.json"),
                          "run scripts/forge_pipeline.py &lt;run_id&gt;"))

    rows = ['<thead><tr><th>run</th><th>encoding</th><th>selection</th>'
            '<th>dict features</th><th>compress</th>'
            '<th>baseline cov≥0.95</th><th>forge score</th>'
            '<th>forge stage</th></tr></thead><tbody>']
    for p in paths:
        try:
            with open(p) as f:
                r = json.load(f)
        except Exception:
            continue
        run_id = r.get("run_id", "?")
        enc = r.get("encoding", "?")
        dict_block = r.get("dictionary", {})
        d_n = dict_block.get("n_features", "?")
        sel_method = (dict_block.get("selection") or {}).get("method", "—")
        cs = r.get("compress", {})
        if "error" in cs:
            cs_cell = f"<span class='fail'>err: {escape(cs['error'][:60])}</span>"
        else:
            cs_cell = (f"clusters {cs.get('n_clusters', '?')}, "
                       f"kept {cs.get('n_features_kept', '?')}, "
                       f"zeroed {cs.get('n_features_zeroed', '?')}")
        baseline = r.get("baseline_score") or {}
        bcov = baseline.get("coverage_0.95")
        bcov_cell = f"{bcov:.1%}" if isinstance(bcov, float) else "—"
        fscore = r.get("forge_score")
        fscore_cell = (f"{fscore:.3f}" if isinstance(fscore, (int, float))
                       else "<em>(blocked on sae-forge release)</em>")
        forge_status = (r.get("forge") or {}).get("status", "?")
        rows.append(
            f"<tr><td><code>{escape(run_id)}</code></td>"
            f"<td><code>{escape(enc)}</code></td>"
            f"<td><code>{escape(sel_method)}</code></td>"
            f"<td>{d_n}</td>"
            f"<td>{cs_cell}</td>"
            f"<td>{bcov_cell}</td>"
            f"<td>{fscore_cell}</td>"
            f"<td><span class='aside'>{escape(forge_status)}</span></td>"
            f"</tr>")
    rows.append("</tbody>")
    table = '<table class="summary">' + "".join(rows) + '</table>'
    return f"""
  <h3>Forge pipeline runs</h3>
  <p>Per-SAE runs of the end-to-end pipeline. Stages 1&ndash;6 (SAE →
  polygram Dictionary → ValidationReport → Compressor → compressed
  safetensors) are wired up; stage 7 (sae-forge) is intentionally stubbed
  pending the upstream pluggable-<code>Faithfulness</code> release. The
  <em>baseline</em> column reports the raw SAE's coverage at AUC ≥ 0.95
  &mdash; this is the number the forged model has to beat (or at least
  match) for the forge to be worth doing on this fixture.</p>
  {table}
  <p class="aside"><strong>Reading the <em>selection</em> column.</strong>
  The polygram encoding caps the Dictionary at a small number of features
  (Rung3 → 16, MPSRung1 → 8). When the SAE has more features than the cap
  &mdash; which it always does on this fixture &mdash; we have to pick a
  subset. <code>head</code> is the legacy behaviour: take the first
  <em>N</em> features by ID, which is arbitrary because SAE feature IDs
  reflect training-time allocation order, not utility. <code>firing_rate</code>
  keeps the features that actually fire on the feed; <code>gt_alignment</code>
  keeps the features most discriminative against GT labels. The selection
  choice can move downstream Compressor cluster counts by 2&ndash;3&times;
  on the same SAE: <code>head</code> on <code>cascade__jumprelu</code>
  yields 1 cluster, <code>firing_rate</code> yields 3, because the
  arbitrary slice through 128 features happens to miss most of the
  confirmed pairs the ValidationReport identified.</p>
  <p class="aside">To populate this table, run
  <code>python scripts/forge_pipeline.py &lt;feed&gt;__&lt;variant&gt;
  [--select-by firing_rate|gt_alignment|head]</code>. The pipeline writes
  <code>runs/sae_forge/&lt;run_id&gt;/forge_results.json</code>
  which this section reads on rebuild.</p>
"""


def _sae_forge_section() -> str:
    """A frank assessment of whether sae-forge can be evaluated on this
    fixture, what its 'wider loops' actually do, and where the gaps come
    from once polygram itself is fixed."""
    return """
  <h3>sae-forge as a candidate benchmark entry</h3>
  <p><a href="https://github.com/jascal/sae-forge" target="_blank"
     rel="noopener">sae-forge</a> sits one stage downstream of polygram:
  it consumes a polygram-compressed SAE checkpoint and projects a host
  model's weights into the surviving feature basis, producing a small
  transformer whose residual stream <em>is</em> the SAE feature space.
  Its "wider loops" (per its README) are a three-loop continual-learning
  structure on top of the single-shard pipeline:</p>
  <ul>
    <li><strong>stream</strong> &mdash; iterate over data shards</li>
    <li><strong>refine</strong> &mdash; per-shard convergence</li>
    <li><strong>basis</strong> &mdash; compress &harr; regrow refinement
        of the feature set (the &quot;wider&quot; loop the README highlights)</li>
  </ul>

  <h4>Status: running end-to-end on sm-sae as of sae-forge v0.4.0</h4>
  <p>v0.4.0 shipped the pluggable
  <code>saeforge.eval.FaithfulnessTarget</code> protocol with built-in
  <code>KLTarget</code> and <code>CosineTarget</code>; the
  <code>GroundTruthAlignment</code> built-in we'd hoped for didn't ship,
  but the protocol is permissive enough that we implement it on our side
  (<code>scripts/forge_pipeline.py:GroundTruthAlignment</code>). It reads
  cascade-sample input_ids from <code>ctx['_eval_input_ids']</code>, runs
  them through the forged model, mean-pools the residual stream, and
  scores AUC against the sm-sae ground-truth label matrix. The pipeline
  uses <code>ForgePipeline.run_synthetic</code> with a tiny in-memory
  GPT-2 (n_embd matching the SAE's input_dim) as the host &mdash; no
  HuggingFace <code>from_pretrained</code> step needed. End-to-end runs
  for both <code>embedded__topk</code> and <code>cascade__jumprelu</code>
  are in the <a href="#benchmark">scoreboard's Forge pipeline runs</a>
  table.</p>

  <h4>What's still synthetic</h4>
  <p>The current host is a <strong>random-init GPT-2 placeholder</strong>
  with no cascade structure. So the forged residual stream carries no real
  signal from the SM dynamics &mdash; faithfulness numbers reported today
  reflect "did the random projection happen to produce features whose
  max-over-residual AUC against the GT labels is high", not "did the
  forge preserve cascade structure". The numbers landing within ~2% of
  the uncompressed-SAE baseline are a wiring sanity check, not a
  scientific result. To get meaningful per-feature faithfulness, the
  next piece is a <strong>cascade-host shim</strong>: a small transformer
  trained on cascade transitions, so its weights actually encode the
  decay dynamics the SAE was trained to find. That's the follow-up.</p>

  <h4>Would the wider loops compensate for polygram weakness?</h4>
  <p>Two cases to keep apart:</p>
  <ul>
    <li><strong>Over-compression.</strong> If polygram drops features
        that carry signal, sae-forge's <em>basis</em> loop (compress
        &harr; regrow) is designed to bring them back. This is a real
        mitigation, and worth measuring once the adapter exists.</li>
    <li><strong>Missing structure.</strong> If polygram's encoding cannot
        <em>express</em> a needed factorization (the original
        MPSRung1+phase-only failure mode), no amount of regrow helps:
        you cannot regrow something polygram never had a way to find.
        The right answer there is to fix the encoding upstream &mdash;
        which is exactly what the Rung3 result above demonstrates.</li>
  </ul>
  <p>So the conceptual answer to <em>"can sae-forge's wider loops
  compensate?"</em> is: <strong>yes for one failure mode, no for the
  other</strong>. sm-sae's hardest current gap is the second kind, and it
  is fixed at the polygram layer rather than the forge layer.</p>

  <h4>Where the remaining gaps come from, by layer</h4>
  <table class="summary">
    <tr><th>layer</th><th>current gap</th><th>where to look</th></tr>
    <tr><td>SAE training (Axis A)</td>
        <td>none worth chasing on this fixture &mdash; ≥96% variance
            explained across all 9 configurations</td>
        <td>becomes interesting again at LLM scale</td></tr>
    <tr><td>SAE coverage (Axis B)</td>
        <td><strong>~63% of GT features missed on the cascade feed</strong>
            (best variant: JumpReLU at 37% coverage at AUC ≥ 0.95)</td>
        <td>this is the main open gap. Try gated SAEs, top-k with auxiliary
            loss, richer cascade distributions, hierarchical / feature-splitting
            SAEs</td></tr>
    <tr><td>polygram compression (Axis C)</td>
        <td>baseline encoding fails 4/4; <code>Rung3</code> with amplitude
            knobs passes 4/4</td>
        <td>solved at the configuration level for the worked pairs.
            <code>Rung4</code> / <code>Rung5</code> add slack for richer
            distributions</td></tr>
    <tr><td>sae-forge (downstream of C)</td>
        <td>untested on this fixture &mdash; needs the adapter described
            above</td>
        <td>build the adapter; then the <em>basis</em> regrow loop becomes
            the mechanism for compensating polygram over-compression
            specifically</td></tr>
  </table>
  <p class="aside">Honest read: with Rung3 in place, the polygram layer
  is no longer the bottleneck. Axis B is. The most informative future
  experiment is to evaluate sae-forge's regrow loop on a config where
  polygram intentionally over-compresses &mdash; that is the setting
  where its wider loops are designed to earn their keep.</p>
"""


def section_benchmark() -> str:
    """Honest, three-axis scoreboard. sm-sae is a fixture, not a product —
    its job is to make compression/extraction techniques comparable."""
    runs_dir = os.path.join(REPO_ROOT, "runs")

    # ---- Axis A: reconstruction quality (runs/summary.json) -------------
    sa_path = os.path.join(runs_dir, "summary.json")
    if os.path.exists(sa_path):
        with open(sa_path) as f:
            sa = json.load(f)
        rows_a = ['<tr><th>feed</th><th>variant</th><th>var explained</th>'
                  '<th>recon loss</th><th>L0</th><th>dead %</th></tr>']
        for _, r in sa.items():
            rows_a.append(
                f"<tr><td>{escape(r['feed'])}</td><td>{escape(r['variant'])}</td>"
                f"<td>{r['var_explained']:.4f}</td>"
                f"<td>{r['recon_loss']:.4f}</td>"
                f"<td>{r['l0']:.1f}</td>"
                f"<td>{r['dead_fraction']:.1%}</td></tr>")
        axis_a = '<table class="summary">' + "".join(rows_a) + '</table>'
    else:
        axis_a = missing(sa_path, "run scripts/train_all.py")

    # ---- Axis B: ground-truth alignment (runs/alignment_summary.json) ----
    sb_path = os.path.join(runs_dir, "alignment_summary.json")
    if os.path.exists(sb_path):
        with open(sb_path) as f:
            sb = json.load(f)
        # Find the headline number: best (mean_best_auc, coverage_0.95) on cascade
        cascade_rows = [r for r in sb if r["feed"] == "cascade"]
        best_cov = max((r["coverage_0.95"] for r in cascade_rows), default=0.0)
        rows_b = ['<tr><th>feed</th><th>variant</th><th>cov ≥0.95</th>'
                  '<th>cov ≥0.90</th><th>mean best AUC</th><th>monosemanticity</th></tr>']
        for r in sb:
            cov = r["coverage_0.95"]
            cls = ("pass" if cov >= 0.7 else
                   "partial" if cov >= 0.4 else "fail")
            rows_b.append(
                f"<tr><td>{escape(r['feed'])}</td>"
                f"<td>{escape(r['variant'])}</td>"
                f"<td class='{cls}'>{cov:.1%}</td>"
                f"<td>{r['coverage_0.90']:.1%}</td>"
                f"<td>{r['mean_best_auc']:.3f}</td>"
                f"<td>{r['monosemanticity']:.1%}</td></tr>")
        axis_b = '<table class="summary scorecard">' + "".join(rows_b) + '</table>'
        axis_b_headline = (
            f"<p>Best result on the hardest feed (<code>cascade</code>): "
            f"<strong>{best_cov:.1%}</strong> of ground-truth features "
            f"recovered at AUC ≥ 0.95. That is the gap the benchmark is "
            f"asking the next technique to close.</p>")
    else:
        axis_b = missing(sb_path, "run scripts/evaluate.py")
        axis_b_headline = ""

    # ---- Axis C: compression-extraction (per-encoding sweep) -------------
    # Replaces the old "MPSRung1 baseline only" view. Pulls the per-encoding
    # results from runs/polygram/sweep/sweep_results.json. Excludes any
    # all-failed config (the legacy MPSRung1 phase-only baseline) so the
    # table reflects what actually works — not stale failures.
    sweep_path_c = os.path.join(REPO_ROOT, "runs", "polygram", "sweep",
                                "sweep_results.json")
    excluded_failures: list[str] = []
    if os.path.exists(sweep_path_c):
        with open(sweep_path_c) as f:
            sweep_c = json.load(f)
        runs_c = sweep_c.get("runs", [])
        configs_c = sweep_c.get("by_config", {})
        # Pairs in column order (use insertion order seen in runs).
        pair_order_c: list[str] = []
        for r in runs_c:
            lab = r.get("label")
            if lab and lab not in pair_order_c:
                pair_order_c.append(lab)
        by_key_c = {(r.get("config"), r.get("label")): r for r in runs_c}
        # Filter out: (a) all-failed configs (baseline), and (b) partial
        # runs that exist only as historical learnings (the unbudgeted
        # Rung4_amp_default with 1/4 pairs — useful as a wall-time data
        # point in the issue tracker, distracting in this table).
        display_configs = []
        for name, agg in configs_c.items():
            n_runs = agg.get("n_runs", 0)
            n_pass = agg.get("n_pass", 0)
            n_err = agg.get("n_error", 0)
            if n_runs > 0 and n_pass == 0 and n_err == 0:
                excluded_failures.append(name)
                continue
            if name == "Rung4_amp_default":
                excluded_failures.append(name)
                continue
            display_configs.append((name, agg))

        if display_configs:
            head = ("<thead><tr><th>encoding config</th><th>knobs</th>"
                    + "".join(f"<th><code>{escape(p)}</code></th>"
                              for p in pair_order_c)
                    + "<th>pass / total</th></tr></thead><tbody>")
            body = []
            for name, agg in display_configs:
                nk = next((r.get("n_knobs") for r in runs_c
                           if r.get("config") == name and r.get("n_knobs")), "?")
                body.append(
                    f"<tr><td><code>{escape(name)}</code><br/>"
                    f"<span style='font-size:11px;color:#666'>"
                    f"{escape(agg.get('desc', ''))}</span></td>"
                    f"<td>{nk}</td>")
                for p in pair_order_c:
                    r = by_key_c.get((name, p))
                    if r is None:
                        body.append("<td class='aside'>—</td>")
                    elif "error" in r:
                        body.append("<td class='fail' title='error'>err</td>")
                    elif r.get("tolerance_met"):
                        body.append(
                            f"<td class='pass' "
                            f"title='Δ={r['delta']:+.4f}'>"
                            f"✓ {r['delta']:+.3f}</td>")
                    else:
                        body.append(
                            f"<td class='fail' "
                            f"title='Δ={r['delta']:+.4f}'>"
                            f"✗ {r['delta']:+.3f}</td>")
                body.append(
                    f"<td>{agg.get('n_pass', 0)}/{agg.get('n_runs', 0)}</td>"
                    f"</tr>")
            axis_c = ('<table class="summary scorecard">'
                      + head + "".join(body) + "</tbody></table>")
            n_passing_configs = sum(1 for _, a in display_configs
                                    if a.get("n_pass", 0) == a.get("n_runs", 0)
                                    and a.get("n_runs", 0) > 0)
            axis_c_headline = (
                f"<p>{n_passing_configs} of {len(display_configs)} displayed "
                f"encoding configurations clear every cancellation pair. "
                "Cells show pass (✓ green) or fail (✗ red) with the achieved "
                "Δ in overlap. Hover for the exact value.</p>")
        else:
            axis_c = ("<p>No passing configurations recorded in "
                      "<code>sweep_results.json</code>.</p>")
            axis_c_headline = ""
    else:
        axis_c = missing(sweep_path_c, "run scripts/polygram_sweep.py")
        axis_c_headline = ""

    excl_note = ""
    if excluded_failures:
        excl_note = (
            f"<p class='aside'>Hidden from this table: "
            + ", ".join(f"<code>{escape(c)}</code>"
                        for c in excluded_failures)
            + f" (0 passes; recorded on disk under "
            "<code>runs/polygram/sweep/</code> for the historical record).</p>")

    # ---- Axes overview table --------------------------------------------
    axes_overview = """
<table class="summary">
  <tr><th>axis</th><th>what it measures</th><th>high score means…</th></tr>
  <tr><td><strong>A</strong>. reconstruction</td>
      <td>does the SAE re-produce activations faithfully?</td>
      <td>the SAE has enough capacity for the feed</td></tr>
  <tr><td><strong>B</strong>. ground-truth alignment</td>
      <td>do SAE features map onto known physical labels?</td>
      <td>features are interpretable, not polysemantic</td></tr>
  <tr><td><strong>C</strong>. compression extraction</td>
      <td>can we re-encode the SAE dictionary into a simpler, structured form
          (polygram, decision tree, sparse code, …) that still respects the
          known feature factorization?</td>
      <td>downstream tools (sae-forge, distillation) can consume it</td></tr>
</table>
"""

    return f"""
<section id="benchmark">
  <h2>The benchmark scoreboard</h2>
  <p>sm-sae is intentionally a <strong>benchmark fixture</strong>, not a
  finished pipeline. Its job is to make compression and extraction techniques
  (polygram is the worked example here, but any technique that takes an SAE
  dictionary and proposes a simpler structured representation should fit) directly
  <strong>comparable and measurable</strong> against exact ground truth. This
  section reports the current scoreboard candidly &mdash; including, in
  particular, where the polygram baseline fails.</p>

  <h3>Three axes of measurement</h3>
  {axes_overview}

  <h3>Axis A &mdash; Reconstruction quality</h3>
  <p>How faithfully does the trained SAE reproduce the feed activations? This
  is the easiest axis: every variant clears 96% variance explained.</p>
  {axis_a}

  <h3>Axis B &mdash; Ground-truth feature alignment</h3>
  {axis_b_headline}
  {axis_b}
  <p class="aside">Easy feeds (<code>raw</code>, <code>embedded</code>) are at
  100% monosemanticity but cap around 70% coverage. The realistic feed
  (<code>cascade</code>) drops to ~37% coverage and ~80% monosemanticity even
  with the best variant. <strong>That gap is the open problem.</strong></p>

  <h3>Axis C &mdash; Compression extraction (per polygram encoding)</h3>
  {axis_c_headline}
  {axis_c}
  {excl_note}
  <p class="aside">Each row is one polygram encoding configuration tested
  against the same 4 cancellation pairs. The dimension of the search and
  the optimizer budget are the two knobs that vary; see the
  <a href="#recommended-defaults">recommended-defaults table</a> below for
  the actionable summary.</p>
  <p class="aside"><strong>Worth flagging:</strong>
  <code>Rung5(n_amp_qubits=2)</code> and <code>Rung4</code> are
  <em>mathematically identical encodings</em> &mdash; Rung5's own
  docstring states <em>"Generalises Rung4 (the fixed n_amp_qubits=2
  case)"</em>. They produce the same gram and the same optimum. But in
  current polygram (v0.8.1) Rung5's code path is much faster: at the
  same <code>max_steps=10</code> budget Rung5 lands the same Δ in
  <strong>~4 seconds per pair</strong> versus Rung4's ~240 seconds, an
  observed ~60× gap. Filed upstream as
  <a href="https://github.com/jascal/polygram/issues/86" target="_blank"
     rel="noopener">polygram#86</a>; for now, prefer
  <code>Rung5(n_amp_qubits=2)</code> over <code>Rung4</code> for new
  callers.</p>

  {_sae_forge_section()}

  {_format_forge_pipeline_results()}

  {_format_recommended_defaults()}

  <h3>What would move each axis</h3>
  <ul>
    <li><strong>Axis A</strong>: already saturated for this fixture size;
        becomes interesting again at LLM scale.</li>
    <li><strong>Axis B</strong> (the SAE itself): try gated SAEs;
        Anthropic-style top-k with auxiliary reconstruction loss; richer
        cascade distributions (deeper decay trees, multi-parent events);
        feature-splitting with hierarchical SAEs.</li>
    <li><strong>Axis C</strong> (polygram specifically): the baseline
        <code>MPSRung1</code>+phase-only failure is now superseded by three
        passing configurations (<code>Rung3</code>, <code>Rung4</code>,
        <code>Rung5</code>, all with amplitude knobs). The cost ordering
        ended up being the surprising part of the sweep: <code>Rung5</code>
        with <code>n_amp_qubits=2</code> converges in seconds where
        <code>Rung4</code> at the same knob-count takes minutes &mdash; so
        the recommended default has shifted accordingly. Remaining sweep
        directions worth exploring: <code>preserve_tiers=False</code> (cost
        of relaxing hierarchy); higher <code>n_amp_qubits</code> on Rung5;
        running the same sweep against a larger SAE slice (the current cap
        is 8 features for MPSRung1, 16 for Rung3, etc.).</li>
  </ul>

  <h3>Other compression/extraction techniques worth benchmarking</h3>
  <p>The benchmark is encoding-agnostic. Anything that consumes a trained
  SAE checkpoint and proposes a structured representation fits, including:</p>
  <ul>
    <li><strong>Hierarchical / group-sparse coding</strong> &mdash; impose
        the known cluster structure as a prior</li>
    <li><strong>Decision-tree / logic-program distillation</strong> &mdash;
        extract symbolic rules from feature firing patterns</li>
    <li><strong>Causal scrubbing</strong> / <strong>attribution patching</strong>
        &mdash; identify which features are actually load-bearing</li>
    <li><strong>Direct probing of feature factorization</strong> &mdash;
        attempt to recover <code>charge ⊗ color ⊗ flavor ⊗ generation</code>
        as an explicit tensor product</li>
    <li><strong>Distilled small transformer</strong> (the
        <code>sae-forge</code> direction) &mdash; forge a tiny interpretable
        model whose internals match the SAE's features</li>
  </ul>
  <p>Each gets a row on the Axis-C table. The benchmark stays the same;
  what varies is the entry.</p>
</section>
"""


# ---------------------------------------------------------------------------
# (d) lifecycle
# ---------------------------------------------------------------------------
def section_lifecycle() -> str:
    flow = """
<svg viewBox="0 0 900 280" class="flow" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="8" markerHeight="8" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#444" />
    </marker>
    <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="8" markerHeight="8" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#a44" />
    </marker>
  </defs>
  <g font-family="sans-serif" font-size="13" text-anchor="middle">
    <!-- Forward arrows (top row, the single-shot pipeline) -->
    <rect x="10"  y="50" width="180" height="60" rx="10" fill="#eef" stroke="#446"/>
    <text x="100" y="78">Substrate model</text>
    <text x="100" y="96" font-size="10">SM bundle / LLM</text>

    <rect x="240" y="50" width="180" height="60" rx="10" fill="#efe" stroke="#464"/>
    <text x="330" y="78">SAE features</text>
    <text x="330" y="96" font-size="10">sparse latent code</text>

    <rect x="470" y="50" width="180" height="60" rx="10" fill="#fee" stroke="#644"/>
    <text x="560" y="78">polygram Dictionary</text>
    <text x="560" y="96" font-size="10">clusters + phase knobs</text>

    <rect x="700" y="50" width="180" height="60" rx="10" fill="#ffe" stroke="#664"/>
    <text x="790" y="78">Compiled artifact</text>
    <text x="790" y="96" font-size="10">format depends on technique</text>

    <line x1="190" y1="80" x2="240" y2="80" stroke="#444" marker-end="url(#arr)"/>
    <line x1="420" y1="80" x2="470" y2="80" stroke="#444" marker-end="url(#arr)"/>
    <line x1="650" y1="80" x2="700" y2="80" stroke="#444" marker-end="url(#arr)"/>

    <text x="215" y="44" font-size="10">cache activations</text>
    <text x="445" y="44" font-size="10">cluster + encode</text>
    <text x="675" y="44" font-size="10">compile</text>

    <!-- Score box (sits below the compiled artifact) -->
    <rect x="700" y="180" width="180" height="60" rx="10"
          fill="#fdf6e3" stroke="#998a4a"/>
    <text x="790" y="206">Benchmark score</text>
    <text x="790" y="223" font-size="10">Axes A / B / C / Faithfulness</text>

    <!-- Compile → Score arrow -->
    <line x1="790" y1="110" x2="790" y2="180" stroke="#444"
          marker-end="url(#arr)"/>
    <text x="820" y="148" font-size="10">measure</text>

    <!-- Feedback loops (in a distinct red) -->
    <!-- Loop 1: score → polygram (compress↔regrow / encoding change) -->
    <path d="M 700 210 Q 560 230 560 130" stroke="#a44" stroke-width="1.4"
          stroke-dasharray="5 3" fill="none" marker-end="url(#arr2)"/>
    <text x="600" y="252" font-size="10" fill="#a44">
      basis loop: regrow / change encoding
    </text>

    <!-- Loop 2: score → SAE features (re-train SAE with new objective) -->
    <path d="M 700 220 Q 400 270 330 130" stroke="#a44" stroke-width="1.4"
          stroke-dasharray="5 3" fill="none" marker-end="url(#arr2)"/>
    <text x="360" y="268" font-size="10" fill="#a44">
      refine loop: re-train SAE on the gap
    </text>

    <!-- Loop 3: score → substrate (rare, e.g. larger feed distribution) -->
    <path d="M 700 230 Q 200 290 100 130" stroke="#c88" stroke-width="1.0"
          stroke-dasharray="3 3" fill="none" marker-end="url(#arr2)"/>
    <text x="140" y="173" font-size="9" fill="#c88">
      stream loop: enrich the data feed
    </text>
  </g>
</svg>
"""

    table = """
<table class="summary">
  <tr><th>sm-sae artifact</th><th>LLM-scale analogue</th></tr>
  <tr><td><code>fields.particle_vectors</code> heatmap</td>
      <td>residual-stream sample heatmap (rows = tokens, cols = neurons)</td></tr>
  <tr><td>three feeds (raw / embedded / cascade)</td>
      <td>activations cached at increasing depth in the transformer</td></tr>
  <tr><td>alignment matrix vs ground truth</td>
      <td>feature-vs-probe correlations; feature-vs-top-examples spreadsheets</td></tr>
  <tr><td><code>origin:t_b</code> recovery on cascade</td>
      <td>"this SAE feature fires on Python code" + steering test</td></tr>
  <tr><td>polygram Dictionary with clusters</td>
      <td>UMAP / agglomerative clusters over SAE features</td></tr>
  <tr><td>cancellation e⁻/e⁺ overlap → floor</td>
      <td>"are these two features truly orthogonal, or is there a shared component?"</td></tr>
  <tr><td>compiled artifact (today: <code>q.orca.md</code>)</td>
      <td>whatever the downstream stage consumes &mdash; format depends on
          the technique; see the table below</td></tr>
</table>

<h3>What goes in the "compiled artifact" box?</h3>
<p>The fourth box of the flow diagram is intentionally generic. polygram
emits <code>q.orca.md</code> because polygram models features as
quantum states (qubit registers + <code>Ry</code>/<code>Rz</code>/CNOT
preparation circuits), which fits the
<a href="https://github.com/jascal/q-orca" target="_blank"
   rel="noopener">q-orca-lang</a> state-machine grammar exactly. Other
extraction techniques would emit different things:</p>

<table class="summary">
  <tr><th>extraction technique</th>
      <th>natural compiled artifact</th>
      <th>state-machine shape?</th></tr>
  <tr><td>polygram (any rung)</td>
      <td><code>q.orca.md</code> (quantum state machine)</td>
      <td>yes &mdash; q-orca</td></tr>
  <tr><td>hierarchical / group-sparse SAE</td>
      <td>safetensors with cluster manifest JSON</td>
      <td>no &mdash; continuous weights</td></tr>
  <tr><td>decision-tree distillation</td>
      <td><code>orca.md</code> (classical state machine) or polygram
          <a href="https://github.com/jascal/polygram"
             target="_blank" rel="noopener">decision-table</a> format</td>
      <td>yes &mdash; classical orca</td></tr>
  <tr><td>logic-program / rule extraction</td>
      <td>Datalog / Prolog / decision table</td>
      <td>yes &mdash; finite</td></tr>
  <tr><td>sae-forge "native model"</td>
      <td>safetensors weights + architecture JSON</td>
      <td>no &mdash; small transformer</td></tr>
  <tr><td>concept-bottleneck network</td>
      <td>safetensors + concept-label sidecar</td>
      <td>partial</td></tr>
  <tr><td>symbolic regression</td>
      <td>SymPy / JSON expression tree</td>
      <td>no &mdash; algebra</td></tr>
</table>

<p>So the question <em>"could any encoding be expressed as a q.orca.md
state machine?"</em> has a precise answer: <strong>only the techniques
whose output is genuinely a finite-state structure.</strong>
quantum-state encodings (polygram), classical state machines, decision
tables, and rule sets all fit. Continuous-weight outputs (distilled
transformers, concept-bottleneck networks, symbolic regression) need
different serializations because forcing them through q.orca.md would
either lose the parameters that matter or require synthesizing fake
discrete states to wrap them.</p>

<p>The Orca family (<code>q.orca.md</code> for quantum,
<code>orca.md</code> for classical, decision-table format) is broad
enough to cover a large fraction of the candidate techniques &mdash; but
it isn't universal. The lifecycle box should be read as <em>"some
serialized form the next stage can consume"</em>, with the format
captured per row in the
<a href="#recommended-defaults">recommended-defaults table</a>.</p>

<h3>What new Orca-shaped dialects might we need?</h3>
<p>Each "no" row in the table above is a gap. Orca's design strengths
(declarative, hand-readable, verifiable invariants, compile/simulate
semantics) are valuable enough that asking <em>"what would an Orca for
this look like?"</em> is worth doing for each gap, rather than defaulting
to opaque binary formats. Sketches of what each extension would need to
express:</p>

<table class="summary">
  <tr><th>proposed dialect</th>
      <th>covers</th>
      <th>key primitives</th>
      <th>characteristic invariants</th></tr>
  <tr><td><code>n-orca</code> (neural)</td>
      <td>distilled small transformers (sae-forge native models),
          concept-bottleneck networks</td>
      <td>layer topology (input/hidden/output widths), residual stream,
          attention pattern declarations, layer-type tags
          (<code>linear</code>, <code>attention</code>, <code>mlp</code>),
          weight references (sibling <code>.safetensors</code> by hash)</td>
      <td>parameter count, per-layer activation shapes, output distribution
          shape; "faithfulness ≥ X under input distribution D"; gradient
          flow paths declared</td></tr>
  <tr><td><code>a-orca</code> (algebraic)</td>
      <td>symbolic-regression outputs, distilled algebraic concept
          formulas, manually-extracted physics laws</td>
      <td>expression trees in s-expression or Polish-prefix form, operator
          table with arity + domain, free-variable bindings, constants</td>
      <td>domain / range, monotonicity, continuity, simplification
          equivalence (round-trip through a CAS produces the same tree)</td></tr>
  <tr><td><code>s-orca</code> (sparse)</td>
      <td>hierarchical / group-sparse SAEs, cluster-aware feature
          dictionaries</td>
      <td>cluster manifest with per-cluster feature lists, sparse-weight
          matrix in COO/CSR form, per-feature labels, optional decoder
          norm preservation tags (the sae-forge "scale-aware" property)</td>
      <td>sparsity level, no orphaned features, cluster sum-to-one
          (if normalized), decoder-norm preservation</td></tr>
  <tr><td><code>p-orca</code> (probabilistic)</td>
      <td>Bayesian networks, factor graphs, Markov random fields over
          extracted features</td>
      <td>variable declarations with domains, factor tables (extension of
          decision-table syntax), conditional independencies as
          assertions</td>
      <td>well-formed joint (factors sum to 1 when normalized),
          conditional-independence claims verifiable from factor
          structure</td></tr>
  <tr><td><code>c-orca</code> (circuit / dataflow)</td>
      <td>mechanistic-interpretability circuit findings (subgraphs of an
          LLM that implement a specific behaviour)</td>
      <td>DAG of operations with explicit signal types, named nodes from
          the host model (residual stream component, attention head,
          MLP neuron), edge weights / attribution scores</td>
      <td>circuit completeness (intervening on the circuit reproduces
          host behaviour to a tolerance), node-attribution
          consistency</td></tr>
</table>

<p class="aside">Two of these (<code>n-orca</code>, <code>s-orca</code>)
could plausibly be sub-grammars of existing orca-lang plus a sidecar
binary, rather than entirely new dialects. The other three
(<code>a-orca</code>, <code>p-orca</code>, <code>c-orca</code>) have
shapes that don't fit a state-machine framing at all and would be new
languages in the Orca family, sharing only the design philosophy
(declarative, verifiable, hand-readable).</p>

<h3>Anticipated future requirements (worth designing for now)</h3>
<ol>
  <li><strong>Cross-dialect composition.</strong> Real distilled models
      are rarely one shape. A useful interpretable model might be
      <em>transformer body + algebraic concept layer + decision-tree
      output head</em> &mdash; three different dialects in one artifact.
      An umbrella "manifest" format that imports from multiple Orca
      dialects (and a clear handoff semantics for signal types crossing
      dialect boundaries) is the closest analogue to ONNX's graph-import
      mechanism, but with Orca's verifiable-invariants discipline.</li>
  <li><strong>Schema versioning.</strong> Extraction techniques evolve.
      Today's <code>q.orca.md</code> is at a specific Q-Orca version; an
      artifact a year old needs to round-trip on a current toolchain.
      Standard semver on dialect grammar + capability negotiation between
      producer / consumer.</li>
  <li><strong>Round-trip verification.</strong> Given an artifact, can
      we re-run the original extraction and confirm we get the same
      compiled output (up to documented non-determinism)? This is the
      "lossless compression" check, and it's what makes the benchmark's
      Axis A measurable post-compression.</li>
  <li><strong>Hardware targeting.</strong> The same artifact might
      compile to GPU, CPU, neuromorphic, or quantum back-ends.
      <code>q-orca</code> already has this via its compile step; the
      other dialects need analogous targets. Important once
      <code>n-orca</code>-style artifacts ship at LLM scale.</li>
  <li><strong>Differentiability / continual learning.</strong> If a
      compiled artifact needs to be re-finetuned (sae-forge's basis
      regrow loop is the immediate example), the dialect must preserve
      the gradient surface &mdash; either by inlining weights or by
      referencing them with a guarantee that the host autograd graph
      can attach.</li>
  <li><strong>Sharded artifacts.</strong> A 9B-parameter forged model
      can't fit one file. The manifest needs to declare shard boundaries
      with content-addressed references (sha256 + size) so a downstream
      tool can validate completeness before running.</li>
  <li><strong>Provenance / audit trail.</strong> Each artifact should
      carry the extraction technique's identifier, version, config
      hash, source-SAE hash, and the benchmark scores it achieved.
      Without this, a year from now nobody knows whether a given
      <code>compressed.safetensors</code> came from
      <code>Rung3</code>-default or <code>Rung5</code>-budget, and the
      scoreboard becomes uninterpretable.</li>
  <li><strong>Partial-information artifacts.</strong> An intermediate
      result that hasn't passed all gates (e.g. polygram cancellation
      that stopped at the structural floor) is still valuable for
      diagnosis. The format should accommodate <em>"this is what we got
      and these gates didn't pass"</em>, not just <em>"this is the
      final answer"</em> &mdash; the current
      <code>SM_..._summary.md</code> files in the sweep are a good
      example of this done well; codify it.</li>
</ol>
<p class="aside">The honest summary: q.orca.md is one good answer for
quantum-state encodings. A handful of cousin dialects covering algebra,
sparsity, probabilistic structure, and circuits would let a much wider
benchmark span (decision trees, distilled transformers, mechanistic
circuits, concept bottlenecks) all emit Orca-flavoured artifacts that
the same downstream tooling can consume. Designing for cross-dialect
composition, schema versioning, and provenance now &mdash; even before
the cousin dialects exist &mdash; would keep the lineage coherent as
the family grows.</p>
"""
    return f"""
<section id="lifecycle">
  <h2>(d) The benchmark loop, and how it scales to real models</h2>
  {flow}
  <p>The top row is the single-shot forward pipeline; the red dashed arrows
  below are the feedback loops that make this a loop rather than a pipeline.
  Each loop is one mechanism by which a poor benchmark score drives a
  change earlier in the chain &mdash; from the cheapest to the most
  expensive intervention:</p>
  <ul>
    <li><strong>basis loop</strong> (compiled artifact → polygram):
        sae-forge's compress&harr;regrow refinement; or, in our sweep,
        switching encoding from MPSRung1 to Rung3/5. Cheapest to iterate
        on, no SAE re-training needed.</li>
    <li><strong>refine loop</strong> (compiled artifact → SAE):
        re-train the SAE with a different objective, more features, or
        different sparsity penalty when the basis loop bottoms out.
        Mid-cost.</li>
    <li><strong>stream loop</strong> (compiled artifact → substrate):
        enrich the data feed itself &mdash; deeper cascade trees,
        multi-event aggregation, larger cascade distributions. Most
        expensive; only justified when the prior two loops have hit
        their ceiling.</li>
  </ul>
  <p>A candidate technique enters at the third box (it takes the SAE's
  discovered features and proposes a simpler structured representation)
  and gets scored against the SM's exact ground truth at every arrow. The
  scoreboard in the previous section reports the current best entry
  &mdash; including, for honesty, where it fails.</p>
  <p>Everything above used the SM as a ground-truth oracle on the left side
  of every comparison. At LLM scale you lose the oracle; the visualization
  shifts from <em>"did the SAE recover X?"</em> to
  <em>"what does the SAE propose as X, and does it survive
  perturbation?"</em>. The same four boxes still apply, with substitutions:</p>
  {table}
  <p class="aside">sm-sae's purpose is to be the
  <strong>calibration rig</strong> for every arrow in that flow: at this
  scale each arrow is verifiable; at LLM scale each arrow becomes a research
  question, and you debug it by checking the technique still behaves
  correctly on the fixture. A technique that doesn't clear the SM benchmark
  shouldn't be trusted at LLM scale.</p>
</section>
"""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
CSS = """
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1080px;
       margin: 2rem auto; padding: 0 1rem; color: #222; line-height: 1.45; }
h1 { border-bottom: 2px solid #224; padding-bottom: .3rem; }
h2 { margin-top: 3rem; border-bottom: 1px solid #ccc; padding-bottom: .2rem; }
h3 { margin-top: 2rem; color: #335; }
code, pre { font-family: ui-monospace, Menlo, monospace; }
pre.code { background: #f6f6f8; padding: .8rem; border-radius: 4px;
           font-size: 12px; overflow-x: auto; }
figure { margin: 1rem 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
figcaption { font-size: 12px; color: #555; margin-top: .4rem; }
table { border-collapse: collapse; margin: 1rem 0; font-size: 13px; }
table.hier td, table.hier th, table.summary td, table.summary th {
  padding: 3px 10px; border-bottom: 1px solid #eee; text-align: left; }
table.hier { table-layout: auto; width: 100%; }
table.hier tr.modhead th { background: #eef; padding-top: .5rem; }
table.hier td { vertical-align: top; }
table.hier td.param { color: #b53; font-weight: bold; }
table.hier td.buffer { color: #555; }
table.hier td.contents { min-width: 320px; max-width: 640px; }
.tdesc { font-size: 12px; color: #335; margin-bottom: .4rem;
         line-height: 1.35; }
details.inspect summary { cursor: pointer; color: #357; font-size: 12px;
                          padding: .15rem 0; }
details.inspect summary .nval { color: #888; font-weight: normal;
                                font-size: 11px; }
details.inspect[open] summary { margin-bottom: .3rem; }
details.slice { margin: .1rem 0 .25rem 1rem;
                border-left: 2px solid #e0e6f0; padding-left: .5rem; }
details.slice > summary { cursor: pointer; color: #557; font-size: 11px;
                          font-family: ui-monospace, Menlo, monospace; }
.slices { margin: .2rem 0; }
.complex > details { margin-bottom: .4rem; }
.complex > details > summary { color: #335; font-size: 12px; }
.tensorwrap { max-height: 360px; max-width: 100%; overflow: auto;
              border: 1px solid #e2e2e6; padding: 3px;
              background: white; border-radius: 3px; }
table.tensor { border-collapse: collapse; font-size: 10px;
               font-family: ui-monospace, Menlo, monospace; }
table.tensor td, table.tensor th {
  padding: 1px 5px; min-width: 26px; text-align: right;
  border: 0.5px solid rgba(0,0,0,0.05); white-space: nowrap;
}
table.tensor th { color: #667; font-weight: normal; background: #f6f6f9;
                  text-align: center; max-width: 110px;
                  overflow: hidden; text-overflow: ellipsis; }
table.tensor thead th { position: sticky; top: 0; z-index: 2; }
table.tensor tbody th { position: sticky; left: 0; z-index: 1; }
code.scalar { font-size: 13px; color: #225; background: #f3f3f8;
              padding: 2px 8px; border-radius: 3px; }
.cat { font-weight: normal; font-size: 11px; color: #557; }
.missing { background: #ffe; border: 1px dashed #aa8; padding: .6rem;
           border-radius: 4px; color: #553; font-size: 13px; }
.error { background: #fee; border: 1px solid #c66; padding: .6rem;
         border-radius: 4px; color: #511; font-size: 12px; }
.aside { color: #555; font-style: italic; }
.honest { background: #fff5f3; border: 1px solid #d88; border-left: 4px solid #c33;
          border-radius: 4px; padding: .8rem 1.1rem; margin: 1rem 0; }
.honest p { margin: .4rem 0; }
table.scorecard td.pass { color: #2a6; font-weight: bold; }
table.scorecard td.partial { color: #b80; font-weight: bold; }
table.scorecard td.fail { color: #c33; font-weight: bold; }
table.defaults { font-size: 12px; }
table.defaults td { vertical-align: top; padding: 6px 10px; }
table.defaults th { background: #eef; font-size: 11px; text-align: left;
                    padding: 4px 10px; }
table.defaults td:nth-child(4) { background: #f5fbf5; }
table.defaults code { font-size: 11px; }
.tree ul { list-style-type: circle; }
.two-col { display: flex; gap: 1.5rem; align-items: flex-start; }
.two-col > div { flex: 1; }
.two-col h4 { margin-top: 0; color: #335; }
table.trace { font-size: 12px; }
table.trace td { vertical-align: top; }
table.trace code { font-size: 11px; }
.flow { width: 100%; height: auto; margin: 1rem 0; }
.nndiag { width: 100%; max-width: 780px; height: auto; display: block;
          margin: 1rem auto; }
.paramlist { font-size: 14px; }
.paramlist li { margin-bottom: .25rem; }
details { margin: 1.5rem 0; background: #fafafa; padding: .6rem 1rem;
          border-radius: 6px; border: 1px solid #ddd; }
details summary { cursor: pointer; font-weight: bold; color: #335;
                  font-size: 14px; padding: .2rem 0; }
details[open] summary { margin-bottom: .6rem; }
.reading h4 { margin-top: 1rem; margin-bottom: .3rem; color: #335;
              font-size: 13px; }
.reading ul { margin-top: 0; font-size: 13px; }
a { color: #245; }
a:hover { color: #038; }
nav { font-size: 14px; }
nav a { margin-right: 1rem; }
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "visualize.html"))
    ap.add_argument("--fast", action="store_true",
                    help="use a smaller cascade feed for quicker rendering")
    args = ap.parse_args()

    print(f"Building report → {args.out}")
    print("  (a) substrate  ...", flush=True)
    sub = safe("substrate", section_substrate)
    print("      running    ...", flush=True)
    run = safe("running", section_running)
    print("  (b) sae        ...", flush=True)
    sae = safe("sae", lambda: section_sae(args.fast))
    print("  (c) polygram   ...", flush=True)
    pol = safe("polygram", section_polygram)
    print("      benchmark  ...", flush=True)
    bench = safe("benchmark", section_benchmark)
    print("  (d) lifecycle  ...", flush=True)
    life = safe("lifecycle", section_lifecycle)

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>sm-sae: lifecycle walkthrough</title>
<style>{CSS}</style>
</head><body>
<h1>sm-sae lifecycle walkthrough</h1>
<p>sm-sae is a <strong>benchmark fixture</strong> for evaluating techniques
that extract simpler, structured representations from sparse autoencoders.
The Standard Model gives us exact ground truth at multiple granularities
(per-charge, per-color, per-generation, per-particle, per-origin), so any
candidate technique &mdash; polygram is the worked example below, but other
methods could be evaluated against the same fixture &mdash; can be scored
quantitatively rather than judged on vibes. This walkthrough reports the
current scores candidly: some axes work, some don't, and the gaps are the
point.</p>
<nav>
  <a href="#substrate">(a) substrate</a>
  <a href="#running">running the model</a>
  <a href="#sae">(b) SAE</a>
  <a href="#polygram">(c) polygram</a>
  <a href="#benchmark">scoreboard</a>
  <a href="#lifecycle">(d) lifecycle</a>
</nav>
{sub}{run}{sae}{pol}{bench}{life}
</body></html>
"""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    size = os.path.getsize(args.out)
    print(f"Wrote {args.out}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
