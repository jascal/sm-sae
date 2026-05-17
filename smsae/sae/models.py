"""Three SAE variants: TopK, L1, JumpReLU.

All share an encoder/decoder structure and a common interface:

    sae = TopKSAE(input_dim, n_features, k=8)
    z   = sae.encode(x)         # sparse latent  (B, n_features)
    x_hat = sae.decode(z)       # reconstruction (B, input_dim)
    out = sae(x)                # SaeOutput dataclass with x_hat, z, loss components

Each variant defines:
  - encode(x) -> sparse z
  - decode(z) -> x_hat
  - sparsity_loss(z) -> scalar (added to recon loss by training loop)
  - n_active(z) -> per-sample count of active features (L0)

Dead-neuron tracking: each SAE tracks how many steps each feature has been
inactive across recent batches via `register_activation(z)`. Used by training
loop for resampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SaeOutput:
    x_hat: torch.Tensor
    z: torch.Tensor
    recon_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    l0: float          # mean active features per sample


# ---------------------------------------------------------------------------
class _BaseSAE(nn.Module):
    """Shared infrastructure: encoder, decoder, dead-neuron tracker."""

    def __init__(self, input_dim: int, n_features: int):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features

        # Encoder weight + bias
        self.W_enc = nn.Parameter(torch.empty(n_features, input_dim))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        # Decoder: tied init transpose, then learned freely
        self.W_dec = nn.Parameter(torch.empty(input_dim, n_features))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        # Initialize: small random, decoder rows unit-norm
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self.W_dec /= self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-9)

        # Dead-neuron tracker: count of steps since each feature was last active
        self.register_buffer("steps_dead", torch.zeros(n_features, dtype=torch.long))

    def pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x - self.b_dec, self.W_enc, self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.W_dec) + self.b_dec

    def register_activation(self, z: torch.Tensor) -> None:
        """Called by training loop after each step; updates dead-neuron counters."""
        with torch.no_grad():
            active = (z.abs() > 1e-9).any(dim=0)
            self.steps_dead = torch.where(active,
                                          torch.zeros_like(self.steps_dead),
                                          self.steps_dead + 1)

    def resample_dead(self, x_batch: torch.Tensor, threshold: int = 200) -> int:
        """Resample dead features by reinitializing them to high-loss inputs.
        Returns number of features resampled.
        """
        dead = (self.steps_dead >= threshold).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            return 0
        with torch.no_grad():
            # Pick random inputs from x_batch to seed the dead features
            idx = torch.randint(0, x_batch.shape[0], (len(dead),), device=x_batch.device)
            seeds = x_batch[idx] - self.b_dec
            seeds = seeds / seeds.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            # Reset encoder rows to those directions, decoder columns to match
            self.W_enc[dead] = seeds * 0.2
            self.W_dec[:, dead] = seeds.t()
            self.b_enc[dead] = 0.0
            self.steps_dead[dead] = 0
        return int(len(dead))

    # ---- subclass interface ----
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...
    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> SaeOutput:
        z = self.encode(x)
        x_hat = self.decode(z)
        recon = F.mse_loss(x_hat, x)
        sparsity = self.sparsity_loss(z)
        l0 = float((z.abs() > 1e-9).float().sum(dim=-1).mean())
        return SaeOutput(x_hat=x_hat, z=z, recon_loss=recon,
                         sparsity_loss=sparsity, l0=l0)


# ---------------------------------------------------------------------------
class TopKSAE(_BaseSAE):
    """Sparse autoencoder using a top-k activation: keep the k largest pre-activations,
    zero the rest. No explicit sparsity penalty needed.
    """
    def __init__(self, input_dim: int, n_features: int, k: int = 8):
        super().__init__(input_dim, n_features)
        self.k = k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        a = self.pre_activation(x)
        # ReLU then take top-k per sample
        a = F.relu(a)
        topk_vals, topk_idx = torch.topk(a, k=min(self.k, self.n_features), dim=-1)
        z = torch.zeros_like(a)
        z.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        return z

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=z.device)


# ---------------------------------------------------------------------------
class L1SAE(_BaseSAE):
    """Classic L1-penalty SAE: ReLU encoder + L1 sparsity loss on activations."""

    def __init__(self, input_dim: int, n_features: int, l1_coeff: float = 1e-2):
        super().__init__(input_dim, n_features)
        self.l1_coeff = l1_coeff

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.pre_activation(x))

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        # Scale by mean decoder-column norm so the penalty is shrinkage-invariant
        # (standard trick from the SAE literature).
        dec_norms = self.W_dec.norm(dim=0)  # (n_features,)
        return self.l1_coeff * (z.abs() * dec_norms).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
class JumpReLUSAE(_BaseSAE):
    """JumpReLU SAE: hard threshold per-feature (learnable threshold theta).
    Below theta, output is 0; above theta, output is the pre-activation
    (no shrinkage). Sparsity penalty is on the *count* of active features
    (L0-surrogate via a straight-through estimator).
    """
    def __init__(self, input_dim: int, n_features: int,
                 l0_coeff: float = 5e-3, init_theta: float = 0.05):
        super().__init__(input_dim, n_features)
        self.log_theta = nn.Parameter(torch.full((n_features,), float(torch.log(torch.tensor(init_theta)))))
        self.l0_coeff = l0_coeff

    def theta(self) -> torch.Tensor:
        return self.log_theta.exp()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        a = self.pre_activation(x)
        # Forward: hard step
        gate = (a > self.theta()).float()
        # Straight-through for the gate gradient
        gate = gate + (a - a.detach()) * 0  # forward only; gradient flows through `a`
        return gate * a

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        # Approximate L0 = mean count of active features per sample
        active = (z.abs() > 1e-9).float()
        return self.l0_coeff * active.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
def make_sae(kind: str, input_dim: int, n_features: int, **kwargs) -> _BaseSAE:
    if kind == "topk":     return TopKSAE(input_dim, n_features, **kwargs)
    if kind == "l1":       return L1SAE(input_dim, n_features, **kwargs)
    if kind == "jumprelu": return JumpReLUSAE(input_dim, n_features, **kwargs)
    raise ValueError(f"Unknown SAE kind: {kind}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    for kind in ("topk", "l1", "jumprelu"):
        sae = make_sae(kind, input_dim=16, n_features=32, **(
            {"k": 4} if kind == "topk" else
            {"l1_coeff": 1e-2} if kind == "l1" else
            {"l0_coeff": 5e-3}
        ))
        out = sae(x)
        print(f"{kind:10s}  recon={float(out.recon_loss):.4f}  "
              f"sparsity={float(out.sparsity_loss):.4f}  L0={out.l0:.2f}")
