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


def test_tiny_gpt2_aux_heads_default_zero_no_attribute():
    """Default `aux_heads=0` keeps the model byte-identical to the
    pre-change artefact: no `aux_head` attribute, no extra parameters."""
    from smsae.host import tiny_gpt2
    m_default = tiny_gpt2(n_embd=16)
    m_explicit_zero = tiny_gpt2(n_embd=16, aux_heads=0)
    assert not hasattr(m_default, "aux_head")
    assert not hasattr(m_explicit_zero, "aux_head")
    # Param counts match — no aux_head, no extra weights
    p_default = sum(p.numel() for p in m_default.parameters())
    p_zero = sum(p.numel() for p in m_explicit_zero.parameters())
    assert p_default == p_zero


def test_tiny_gpt2_aux_heads_positive_attaches_linear():
    """`aux_heads=5` attaches a `model.aux_head` `nn.Linear(n_embd, 5)`."""
    from smsae.host import tiny_gpt2
    import torch.nn as nn

    m = tiny_gpt2(n_embd=16, aux_heads=5)
    assert hasattr(m, "aux_head")
    assert isinstance(m.aux_head, nn.Linear)
    assert m.aux_head.in_features == 16
    assert m.aux_head.out_features == 5
    # Param delta: 16 * 5 + 5 = 85 (weight + bias)
    p_base = sum(p.numel() for p in tiny_gpt2(n_embd=16).parameters())
    p_aux = sum(p.numel() for p in m.parameters())
    assert p_aux - p_base == 16 * 5 + 5


def test_tiny_gpt2_aux_heads_does_not_change_forward_output():
    """The standard `model(input_ids)` forward path is unchanged by the
    aux head — same logit shape, the head is consumed separately."""
    from smsae.host import tiny_gpt2

    m_with = tiny_gpt2(n_embd=16, aux_heads=5)
    m_with.eval()
    input_ids = torch.zeros((2, 8), dtype=torch.long)
    with torch.no_grad():
        out = m_with(input_ids=input_ids)
    assert out.logits.shape == (2, 8, 62)
