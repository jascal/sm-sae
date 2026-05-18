"""Cascade-host shim: tiny transformers trained to predict next-state
cascade transitions, used as the synthetic host model for
sae-forge's ForgePipeline.run_synthetic.

The host is not a substitute for the cascade simulator; it gives
sae-forge a transformer-shaped artefact to project into the polygram
feature basis. With a *trained* host, the resulting forge faithfulness
numbers reflect cascade-structure preservation; with a *random-init*
host (the fallback), they reflect random projection noise.
"""

from __future__ import annotations

from smsae.host.tiny_gpt2 import tiny_gpt2

__all__ = ["tiny_gpt2"]
