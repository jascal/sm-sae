"""Build the SM tensor data files under data/.

Generates:
    data/sm_data.npz          - flat numpy bundle
    data/sm_data.safetensors  - safetensors equivalent (complex split into _re/_im)
    data/sm.safetensors       - PyTorch nn.Module state_dict export
    data/sm_labels.json       - JSON labels sidecar for sm.safetensors

Usage:
    python scripts/build_data.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

DATA_DIR = os.path.join(REPO_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def main():
    print("=" * 78)
    print("Building flat numpy + safetensors bundle (data/sm_data.*)")
    print("=" * 78)
    from smsae.sm.export import build_bundle, save_npz, save_safetensors
    bundle = build_bundle()
    npz_path = os.path.join(DATA_DIR, "sm_data.npz")
    st_path  = os.path.join(DATA_DIR, "sm_data.safetensors")
    save_npz(bundle, npz_path)
    save_safetensors(bundle, st_path)
    print(f"  wrote {npz_path}  ({os.path.getsize(npz_path):,} bytes)")
    print(f"  wrote {st_path}    ({os.path.getsize(st_path):,} bytes)")

    print("\n" + "=" * 78)
    print("Building PyTorch nn.Module export (data/sm.safetensors + sm_labels.json)")
    print("=" * 78)
    from smsae.sm.torch_model import StandardModel, save_with_labels
    model = StandardModel()
    st_path = os.path.join(DATA_DIR, "sm.safetensors")
    lab_path = os.path.join(DATA_DIR, "sm_labels.json")
    save_with_labels(model, st_path, lab_path)
    print(f"  wrote {st_path}     ({os.path.getsize(st_path):,} bytes)")
    print(f"  wrote {lab_path}    ({os.path.getsize(lab_path):,} bytes)")


if __name__ == "__main__":
    main()
