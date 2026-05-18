"""Tiny GPT-2 builder for the cascade-host shim.

`tiny_gpt2(n_embd)` returns a fresh `GPT2LMHeadModel` sized for the
sm-sae fixture's dimensions (typical: `n_embd=16` for embedded__topk,
`n_embd=61` for cascade__jumprelu). The shape matches what sae-forge's
`GPT2Adapter` expects.
"""

from __future__ import annotations


def tiny_gpt2(n_embd: int, n_layer: int = 2, n_head: int | None = None,
              vocab_size: int = 62, n_positions: int = 64,
              n_inner: int | None = None):
    """Build a tiny GPT2LMHeadModel sized for the cascade-host shim.

    Args:
        n_embd: hidden width. Must match the target SAE's `input_dim` so
            the SubspaceProjector can project the host's weights into
            the feature basis.
        n_layer: number of transformer blocks.
        n_head: number of attention heads. If None, defaults to
            `max(1, n_embd // 8)`, then snapped down until
            `n_embd % n_head == 0`.
        vocab_size: token vocabulary (default 62 = 61 SM particles + PAD).
        n_positions: max sequence length.
        n_inner: FFN inner dim (default `n_embd * 4`, the standard GPT-2 ratio).

    Returns:
        A fresh `transformers.GPT2LMHeadModel` with random init.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    if n_head is None:
        n_head = max(1, n_embd // 8)
    # Snap n_head down to a divisor of n_embd
    while n_head > 1 and n_embd % n_head != 0:
        n_head -= 1
    if n_embd % n_head != 0:
        raise ValueError(
            f"could not find n_head that divides n_embd={n_embd} "
            f"(requested n_head={n_head}).")
    if n_inner is None:
        n_inner = n_embd * 4

    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        n_positions=n_positions,
    )
    return GPT2LMHeadModel(cfg)
