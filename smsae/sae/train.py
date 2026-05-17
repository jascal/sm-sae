"""Training loop for sm-sae.

Standard SAE-training rituals:
  - Adam optimizer, cosine LR with warmup
  - Decoder columns kept unit-norm (constrain after each step)
  - Periodic dead-neuron resampling
  - Logging of recon loss, L0, dead-feature rate, total loss

Used by smsae_run_all.py (Phase A.3) for the full sweep, but the
`train()` function is generic and can be called for any (SAE, Feed) pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from smsae.sae.data import Feed
from smsae.sae.models import _BaseSAE


@dataclass
class TrainConfig:
    epochs: int = 2000
    batch_size: int = 256
    lr: float = 1e-3
    warmup_steps: int = 100
    resample_every: int = 500
    resample_threshold: int = 200
    constrain_decoder: bool = True   # keep ||W_dec[:,k]|| = 1 after each step
    log_every: int = 200
    device: str = "cpu"


@dataclass
class TrainHistory:
    step: list[int] = field(default_factory=list)
    recon_loss: list[float] = field(default_factory=list)
    sparsity_loss: list[float] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    l0: list[float] = field(default_factory=list)
    dead_fraction: list[float] = field(default_factory=list)
    resamples: list[tuple[int, int]] = field(default_factory=list)   # (step, n_resampled)


def _lr_at(step: int, cfg: TrainConfig, total_steps: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    # Cosine decay after warmup
    import math
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def train(sae: _BaseSAE, feed: Feed, cfg: TrainConfig = TrainConfig(),
          verbose: bool = True) -> TrainHistory:
    """Train a single SAE on a single feed. Returns history."""
    device = torch.device(cfg.device)
    sae = sae.to(device)
    X = feed.X.to(device)
    N, D = X.shape
    assert D == sae.input_dim, f"feed D={D} != SAE input_dim={sae.input_dim}"

    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    hist = TrainHistory()

    steps_per_epoch = max(1, (N + cfg.batch_size - 1) // cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    step = 0

    for ep in range(cfg.epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, cfg.batch_size):
            batch_idx = perm[i:i + cfg.batch_size]
            x = X[batch_idx]
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, cfg, total_steps)

            out = sae(x)
            loss = out.recon_loss + out.sparsity_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Constrain decoder columns to unit norm
            if cfg.constrain_decoder:
                with torch.no_grad():
                    norms = sae.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-9)
                    sae.W_dec.div_(norms)

            # Dead-neuron tracking
            sae.register_activation(out.z)

            # Resample
            if (step + 1) % cfg.resample_every == 0:
                n_resampled = sae.resample_dead(x, threshold=cfg.resample_threshold)
                if n_resampled > 0:
                    hist.resamples.append((step, n_resampled))

            # Log
            if step % cfg.log_every == 0 or step == total_steps - 1:
                dead = float((sae.steps_dead >= cfg.resample_threshold).float().mean())
                hist.step.append(step)
                hist.recon_loss.append(float(out.recon_loss))
                hist.sparsity_loss.append(float(out.sparsity_loss))
                hist.total_loss.append(float(loss))
                hist.l0.append(out.l0)
                hist.dead_fraction.append(dead)
                if verbose:
                    print(f"  step {step:>6d}  recon={float(out.recon_loss):.4f}  "
                          f"sparsity={float(out.sparsity_loss):.4f}  "
                          f"L0={out.l0:.2f}  dead={dead:.2%}")

            step += 1

    return hist


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Smoke test on the raw feed with a small TopK SAE
    from smsae.sae.data import feed_raw
    from smsae.sae.models import make_sae

    torch.manual_seed(0)
    feed = feed_raw()
    sae = make_sae("topk", input_dim=feed.D, n_features=32, k=4)
    cfg = TrainConfig(epochs=200, batch_size=32, lr=3e-3, log_every=200)
    hist = train(sae, feed, cfg)
    print(f"\nFinal: recon={hist.recon_loss[-1]:.4f}  L0={hist.l0[-1]:.2f}  "
          f"dead={hist.dead_fraction[-1]:.2%}  resamples={len(hist.resamples)}")
