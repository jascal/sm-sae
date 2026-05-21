"""Train a tiny GPT-2 to predict next-state cascade transitions.

The trained model is used as the synthetic host by sae-forge's
ForgePipeline.run_synthetic. With a trained host, forge faithfulness
numbers reflect cascade-structure preservation; with a random-init
host (the fallback), they reflect random projection noise.

Output layout:
    runs/cascade_host/<n_embd>/
        host/                  HuggingFace save_pretrained output
        config.json            train metadata (n_trajectories, loss, seed, git)

Usage:
    python scripts/train_cascade_host.py --n-embd 16
    python scripts/train_cascade_host.py --n-embd 16 --n-trajectories 100 --epochs 1  # smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _build_dataset(n_trajectories: int, seed: int, max_seq: int,
                    *, with_aux: bool = False, with_gt_aux: bool = False):
    """Materialize the (input_ids, target_ids[, aux_labels]) tensors.

    ``with_aux=False`` (default) returns a 2-tuple matching the
    legacy contract. ``with_aux=True`` returns a 3-tuple whose third
    element is a ``(N, 5)`` float32 tensor of v1 5-label aux labels.
    ``with_gt_aux=True`` returns a 3-tuple whose third element is a
    ``(N, n_gt_features)`` tensor of v2 110-feature GT aux labels.
    The two aux modes are mutually exclusive.
    """
    from smsae.sae.data import cascade_transitions
    rows = list(cascade_transitions(
        n_trajectories=n_trajectories, seed=seed, max_seq=max_seq,
        with_aux=with_aux, with_gt_aux=with_gt_aux,
    ))
    if not rows:
        raise RuntimeError(
            f"cascade_transitions produced no pairs for "
            f"n_trajectories={n_trajectories}; check the rollout config.")
    input_ids = torch.from_numpy(np.stack([r[0] for r in rows], axis=0))
    targets = torch.from_numpy(np.stack([r[1] for r in rows], axis=0))
    if not (with_aux or with_gt_aux):
        return input_ids, targets
    aux_labels = torch.from_numpy(np.stack([r[2] for r in rows], axis=0))
    return input_ids, targets, aux_labels


def _focal_bce_loss(logits: "torch.Tensor", targets: "torch.Tensor",
                    gamma: float = 2.0) -> "torch.Tensor":
    """Focal binary-cross-entropy loss, matching
    ``saeforge.training.heads.focal_bce_loss``'s contract: standard
    BCE_with_logits per element, weighted by ``(1 - p_t)^gamma`` where
    ``p_t = sigmoid(logits)`` for positives and ``1 - sigmoid(logits)``
    for negatives. ``gamma=0`` reduces exactly to plain BCE.

    Implemented locally rather than imported from sae-forge because
    sm-sae's torch-light contract (cascade-host trainer runs in <30s on
    CPU) doesn't justify the full sae-forge import chain just for one
    loss function; the implementation is < 10 lines.

    Reduction is mean over all elements (matches the legacy pooled-BCE
    setup that ships in v1).
    """
    bce_per_element = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
    )
    if gamma == 0:
        return bce_per_element.mean()
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    return ((1.0 - p_t) ** gamma * bce_per_element).mean()


def _cosine_lr(step: int, total: int, base_lr: float,
               warmup_frac: float = 0.1) -> float:
    """Cosine schedule with linear warmup over `warmup_frac` of steps."""
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


_VALID_AUX_SUPERVISION_MODES = ("off", "pooled", "per_channel", "dual")


def train(n_embd: int, n_trajectories: int = 2000, epochs: int = 5,
          batch_size: int = 32, lr: float = 5e-3, seed: int = 0,
          n_layer: int = 2, n_head: int | None = None,
          max_seq: int = 32, out: str | None = None,
          log_every: int = 100,
          aux_supervision: str = "off",
          aux_lambda: float = 1.0,
          focal_gamma: float = 2.0) -> dict:
    """Run the training loop and persist the trained host. Returns metadata.

    Args:
        aux_supervision: one of ``"off"`` (default; byte-identical to
            pre-change behaviour), ``"pooled"`` (mean-pool the final
            hidden state across the sequence axis, feed into a single
            ``Linear -> BCE_with_logits`` head against the 5-label v1
            vocabulary), ``"per_channel"`` (v2 from
            richer-cascade-host-supervision-v2; pooled head against the
            full 110-feature GT vocabulary with focal-BCE loss).
            ``"dual"`` still raises ``NotImplementedError``.
        aux_lambda: coefficient on the BCE term. Ignored when
            ``aux_supervision == "off"``.
        focal_gamma: focal-loss focusing parameter. Ignored when
            ``aux_supervision != "per_channel"``. Default ``2.0`` matches
            the econ-sae Phase 6.2 setting that lifted regime-tier mAUC
            from 0.595 to 0.991.
    """
    from smsae.host import tiny_gpt2

    if aux_supervision not in _VALID_AUX_SUPERVISION_MODES:
        raise ValueError(
            f"--aux-supervision must be one of "
            f"{list(_VALID_AUX_SUPERVISION_MODES)}; got {aux_supervision!r}")
    if aux_supervision == "dual":
        raise NotImplementedError(
            f"--aux-supervision=dual is not implemented yet. v1 ships "
            f"pooled (5 labels) and v2 ships per_channel (110 labels). "
            f"Dual-head (pooled + per_channel simultaneously) is the "
            f"documented follow-up in "
            f"openspec/changes/archive/2026-05-20-aux-supervise-cascade-host/proposal.md.")
    aux_on = aux_supervision in ("pooled", "per_channel")
    use_gt_vocab = aux_supervision == "per_channel"

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = out or os.path.join(REPO_ROOT, "runs", "cascade_host", str(n_embd))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[train] n_embd={n_embd}  n_layer={n_layer}  n_trajectories={n_trajectories}")
    if aux_on:
        loss_kind = "focal-BCE" if use_gt_vocab else "BCE"
        print(f"[train] aux_supervision={aux_supervision}  "
              f"aux_lambda={aux_lambda}  loss={loss_kind}"
              + (f"  focal_gamma={focal_gamma}" if use_gt_vocab else ""))
    print(f"[train] building dataset...")
    t0 = time.time()
    if aux_on:
        input_ids, targets, aux_labels = _build_dataset(
            n_trajectories=n_trajectories, seed=seed, max_seq=max_seq,
            with_aux=not use_gt_vocab,
            with_gt_aux=use_gt_vocab,
        )
        n_aux_labels = aux_labels.shape[1]
    else:
        input_ids, targets = _build_dataset(
            n_trajectories=n_trajectories, seed=seed, max_seq=max_seq,
        )
        aux_labels = None
        n_aux_labels = 0
    n_samples = input_ids.shape[0]
    print(f"[train]   {n_samples} transition pairs "
          f"({time.time() - t0:.1f}s)")

    print(f"[train] building model...")
    model = tiny_gpt2(n_embd=n_embd, n_layer=n_layer, n_head=n_head,
                      n_positions=max_seq,
                      aux_heads=n_aux_labels if aux_on else 0)
    n_params = sum(p.numel() for p in model.parameters())
    aux_param_note = ""
    if aux_on:
        aux_param_note = f"  (aux_head: {n_embd}*{n_aux_labels} + {n_aux_labels} params)"
    print(f"[train]   {n_params} params  n_head={model.config.n_head}  "
          f"n_inner={model.config.n_inner}{aux_param_note}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, math.ceil(n_samples / batch_size) * epochs)

    model.train()
    step = 0
    last_loss = float("nan")
    last_ce = float("nan")
    last_aux = float("nan")
    for epoch in range(epochs):
        # Reshuffle each epoch
        perm = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed + epoch))
        for start in range(0, n_samples, batch_size):
            batch_idx = perm[start:start + batch_size]
            x = input_ids[batch_idx]
            y = targets[batch_idx]
            if aux_on:
                a = aux_labels[batch_idx]
                model_out = model(input_ids=x, output_hidden_states=True)
                logits = model_out.logits  # (B, T, V)
                hidden = model_out.hidden_states[-1]  # (B, T, n_embd)
                pooled = hidden.mean(dim=1)  # (B, n_embd)
                aux_logits = model.aux_head(pooled)  # (B, n_aux_labels)
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    ignore_index=-100,
                )
                if use_gt_vocab:
                    aux_loss = _focal_bce_loss(aux_logits, a, gamma=focal_gamma)
                else:
                    aux_loss = F.binary_cross_entropy_with_logits(aux_logits, a)
                loss = ce_loss + aux_lambda * aux_loss
                last_ce = float(ce_loss.detach())
                last_aux = float(aux_loss.detach())
            else:
                model_out = model(input_ids=x)
                logits = model_out.logits  # (B, T, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    ignore_index=-100,
                )
                last_ce = float(loss.detach())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            cur_lr = _cosine_lr(step, total_steps, lr)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            opt.step()
            last_loss = float(loss.detach())
            if step % log_every == 0:
                msg = (f"[train]   epoch={epoch} step={step}/{total_steps}  "
                       f"loss={last_loss:.4f}  lr={cur_lr:.2e}")
                if aux_on:
                    msg += f"  ce={last_ce:.4f}  aux={last_aux:.4f}"
                print(msg)
            step += 1

    host_dir = os.path.join(out_dir, "host")
    print(f"[train] saving → {host_dir}")
    model.save_pretrained(host_dir)

    # Lazy import to avoid importing aux_labels in the off path.
    aux_label_list: list[str] = []
    if aux_on:
        if use_gt_vocab:
            from smsae.host.aux_labels import gt_aux_label_names
            aux_label_list = gt_aux_label_names()
        else:
            from smsae.host.aux_labels import aux_label_names
            aux_label_list = aux_label_names()

    meta = {
        "n_embd":              n_embd,
        "n_layer":             n_layer,
        "n_head":              model.config.n_head,
        "n_inner":             model.config.n_inner,
        "vocab_size":          model.config.vocab_size,
        "n_positions":         model.config.n_positions,
        "n_params":            n_params,
        "n_trajectories":     n_trajectories,
        "n_train_pairs":       n_samples,
        "n_train_steps":       step,
        "epochs":              epochs,
        "batch_size":          batch_size,
        "lr":                  lr,
        "seed":                seed,
        "train_loss_final":   last_loss,
        # Aux-supervision metadata (always written; default sentinels
        # when off so consumers can read a single uniform schema).
        "aux_supervision":     aux_supervision,
        "aux_labels":          aux_label_list,
        "aux_lambda":          aux_lambda if aux_on else 0.0,
        "aux_loss_final":      last_aux if aux_on else None,
        "focal_gamma":         focal_gamma if use_gt_vocab else None,
        "git_commit":          _git_commit(),
        "elapsed_seconds":     time.time() - t0,
    }
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] wrote {cfg_path}  (final loss = {last_loss:.4f}"
          + (f", aux loss = {last_aux:.4f}" if aux_on else "") + ")")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-embd", type=int, required=True,
                    help="hidden width; must match the target SAE's input_dim")
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=None)
    ap.add_argument("--n-trajectories", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=32)
    ap.add_argument("--out", default=None,
                    help="output dir; default runs/cascade_host/<n_embd>/")
    ap.add_argument(
        "--aux-supervision",
        choices=_VALID_AUX_SUPERVISION_MODES,
        default="off",
        help=("Optional auxiliary supervision head over 5 per-state binary "
              "labels (charge/baryon conservation + lineage/existence). "
              "Default 'off' preserves byte-identical pre-change behaviour. "
              "'pooled' adds a single mean-pool -> Linear -> BCE head; "
              "'per_channel' and 'dual' raise NotImplementedError until the "
              "follow-up changes ship. Closes gate 7.3 if successful: "
              "trained-vs-random faithfulness delta on cascade__jumprelu "
              ">= 0.05."),
    )
    ap.add_argument(
        "--aux-lambda",
        type=float,
        default=1.0,
        help=("Coefficient on the BCE-on-aux term in the combined loss. "
              "Ignored when --aux-supervision off. Default 1.0 matches "
              "the econ-sae Phase 5.1 setting."),
    )
    ap.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help=("Focal-loss focusing parameter for --aux-supervision per_channel. "
              "Ignored otherwise. Default 2.0 matches econ-sae Phase 6.2; "
              "gamma=0 reduces exactly to plain BCE."),
    )
    args = ap.parse_args()

    train(
        n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
        n_trajectories=args.n_trajectories, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        max_seq=args.max_seq, out=args.out,
        aux_supervision=args.aux_supervision,
        aux_lambda=args.aux_lambda,
        focal_gamma=args.focal_gamma,
    )


if __name__ == "__main__":
    main()
