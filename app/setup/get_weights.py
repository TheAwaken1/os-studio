"""Download OS Studio weights for a fresh machine.

Usage:
    python setup/get_weights.py <models_dir>

Creates <models_dir>/{YuE2/YuE2-3B, YuE2/YuE2-Vae, SheetSage2,
MERT2/MERT-v2-FullSong, clip, seed_vc} and downloads what's missing.
Reuses the HF cache; safe to re-run.
"""
from __future__ import annotations

import os
import sys


def main(models_dir: str) -> None:
    from huggingface_hub import snapshot_download

    targets = [
        # (repo_id, local_dir, allow_patterns)
        ("m-a-p/YuE2-3B", os.path.join(models_dir, "YuE2", "YuE2-3B"), None),
        ("m-a-p/YuE2-Vae", os.path.join(models_dir, "YuE2", "YuE2-Vae"), None),
    ]
    # SheetSage2 / MERT2 layout used by ComfyUI-YuE2 docs
    for name in ("SheetSage2",):
        targets.append((f"m-a-p/{name}", os.path.join(models_dir, name), None))

    for repo, local, allow in targets:
        if os.path.exists(os.path.join(local, "config.json")):
            print(f"[skip] {repo} -> {local}")
            continue
        print(f"[get ] {repo} -> {local}")
        snapshot_download(repo_id=repo, local_dir=local,
                          allow_patterns=allow)

    print("""
Remaining weights are fetched automatically on first use:
  - seed-vc (DiT, whisper-small, bigvgan, rmvpe, campplus, hdemucs,
    openvoice, se_db) -> <models_dir>/seed_vc
  - Qwen3-VL-4B-Instruct (lyrics) -> HF cache
If you have the Pinokio MiniMax-H3 install, the fastest path is to copy or
point config.json "models_dir" at:
  E:\\AI\\api\\minimax-h3-pinokio.git\\app\\models
""")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
