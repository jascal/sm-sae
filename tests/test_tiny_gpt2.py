"""Smoke tests for smsae.host.tiny_gpt2."""

from __future__ import annotations

import torch


def test_tiny_gpt2_n_embd_16_param_count():
    """n_embd=16 model is non-degenerate. The openspec's 30k forecast
    didn't account for the tied LM head + small (62-token) vocab; the
    floor here is the actually-measured size."""
    from smsae.host import tiny_gpt2
    model = tiny_gpt2(n_embd=16)
    n_params = sum(p.numel() for p in model.parameters())
    assert 5_000 <= n_params <= 200_000, (
        f"n_embd=16 model has {n_params} params, expected in [5k, 200k]")


def test_tiny_gpt2_n_embd_61_param_count():
    """n_embd=61 (cascade SAE width). Spec target was ~110k; measured ~99k."""
    from smsae.host import tiny_gpt2
    model = tiny_gpt2(n_embd=61)
    n_params = sum(p.numel() for p in model.parameters())
    assert 50_000 <= n_params <= 200_000, (
        f"n_embd=61 model has {n_params} params, expected in [50k, 200k]")


def test_tiny_gpt2_forward_shape():
    """Forward pass on a (4, 32) batch yields (4, 32, vocab_size) logits."""
    from smsae.host import tiny_gpt2
    model = tiny_gpt2(n_embd=16)
    model.eval()
    input_ids = torch.zeros((4, 32), dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits
    assert logits.shape == (4, 32, 62)
    assert torch.isfinite(logits).all()


def test_tiny_gpt2_n_head_divisor_snap():
    """n_head is snapped down to a divisor of n_embd."""
    from smsae.host import tiny_gpt2
    # n_embd=17 is prime; only divisor in range is 1
    model = tiny_gpt2(n_embd=17, n_head=4)
    assert model.config.n_head == 1
