"""Bootstrap the standalone engine: vendored comfy + node stacks + shims.

Import this BEFORE anything from engine/. It wires sys.path so that:
  - `comfy`, `fl_yue2`, `yue2_repo`, `song_lrc`, `song_voice`, `node_helpers`
    resolve to ./engine
  - `folder_paths` resolves to ./shims (config-driven, no ComfyUI server)

The real comfy.model_management / ModelPatcher are kept (they provide VRAM
offload for the 16 GB card); only folder_paths is faked.
"""
from __future__ import annotations

import os
import sys

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(APP_ROOT, "engine")
SHIMS = os.path.join(APP_ROOT, "shims")


def bootstrap() -> None:
    if SHIMS not in sys.path:
        sys.path.insert(0, SHIMS)
    if ENGINE not in sys.path:
        sys.path.insert(0, ENGINE)
    # shims win for folder_paths; engine for the rest
    import folder_paths  # noqa: F401  (shims/folder_paths.py)
    assert os.path.dirname(folder_paths.__file__) == SHIMS, folder_paths.__file__


def set_progress_hook(hook) -> None:
    """hook(value, total, preview_image, node_id=None) per comfy ProgressBar."""
    import comfy.utils
    comfy.utils.set_progress_bar_global_hook(hook)


def set_cancel_check(fn) -> None:
    """fn() raises to abort; wired into comfy.model_management interrupt."""
    import comfy.model_management as mm

    def _throw():
        if fn is not None:
            fn()

    mm.throw_exception_if_processing_interrupted = _throw
