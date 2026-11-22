"""Drop-in replacement for ComfyUI's folder_paths, backed by app config.

The vendored stacks (fl_yue2, yue2_repo, song_lrc, song_voice) call exactly
these: models_dir, get_output_directory(), get_temp_directory(),
get_folder_paths(name), add_model_folder_path(name, path), and read
folder_names_and_paths. Everything else in the original module is unused.
"""
from __future__ import annotations

import json
import os

_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_APP_ROOT, "config.json")

def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}

_cfg = _load_config()

base_path = _cfg.get("app_dir", _APP_ROOT)
models_dir = os.path.abspath(_cfg.get("models_dir", os.path.join(base_path, "models")))
output_directory = os.path.abspath(_cfg.get("output_dir", os.path.join(base_path, "output")))
temp_directory = os.path.abspath(_cfg.get("temp_dir", os.path.join(base_path, "output", "temp")))
input_directory = os.path.abspath(_cfg.get("input_dir", os.path.join(base_path, "input")))
# Trained voices belong to this app and user, never to a shared model tree.
voices_directory = os.path.abspath(_cfg.get("voices_dir", os.path.join(base_path, "voices")))

for _d in (models_dir, output_directory, temp_directory, input_directory, voices_directory):
    os.makedirs(_d, exist_ok=True)

supported_pt_extensions = {'.ckpt', '.pt', '.pt2', '.bin', '.pth', '.safetensors', '.pkl', '.sft'}

folder_names_and_paths: dict[str, tuple[list[str], set[str]]] = {}

_SEED = {
    "checkpoints": ["checkpoints"],
    "text_encoders": ["text_encoders", "clip"],
    "vae": ["vae"],
    "diffusion_models": ["unet", "diffusion_models"],
    "loras": ["loras"],
    "yue2": ["yue2", "YuE2"],
}
for _name, _subs in _SEED.items():
    folder_names_and_paths[_name] = (
        [os.path.join(models_dir, s) for s in _subs],
        supported_pt_extensions,
    )


def add_model_folder_path(folder_name: str, full_folder_path: str, is_default: bool = False) -> None:
    full_folder_path = os.path.abspath(full_folder_path)
    if folder_name not in folder_names_and_paths:
        folder_names_and_paths[folder_name] = ([], supported_pt_extensions)
    paths, exts = folder_names_and_paths[folder_name]
    if full_folder_path in paths:
        return
    if is_default:
        paths.insert(0, full_folder_path)
    else:
        paths.append(full_folder_path)


def get_folder_paths(folder_name: str) -> list[str]:
    if folder_name not in folder_names_and_paths:
        raise KeyError(f"Unknown folder name: {folder_name}")
    return list(folder_names_and_paths[folder_name][0])


def get_output_directory() -> str:
    os.makedirs(output_directory, exist_ok=True)
    return output_directory


def get_temp_directory() -> str:
    os.makedirs(temp_directory, exist_ok=True)
    return temp_directory


def get_input_directory() -> str:
    os.makedirs(input_directory, exist_ok=True)
    return input_directory


# ---------------------------------------------------------------- registration
def register_model_roots(cfg: dict | None = None) -> None:
    """Point the model registries at the on-disk weight tree (config.json:
    "models_dir" already covers <models>/<name>; "model_roots" overrides per key)."""
    cfg = cfg if cfg is not None else _load_config()
    roots = cfg.get("model_roots", {})
    for name, path in roots.items():
        add_model_folder_path(name, os.path.abspath(path), is_default=True)


register_model_roots(_cfg)
