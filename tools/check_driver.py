"""Stop the install early when the NVIDIA driver is too old for CUDA 13.

OS Studio's PyTorch build (torch 2.10 + CUDA 13.0) covers RTX 20, 30, 40 and 50
series cards, but CUDA 13 only runs on NVIDIA driver 580 or newer. With an older
driver the install would succeed and every song would then fail with a cryptic
CUDA error, so check first, before the multi-GB PyTorch download.

install.js watches for the two marker lines printed below.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

MIN_DRIVER = 580
FAIL = "OS Studio can't install:"
PASS = "NVIDIA driver check passed"


def find_nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for path in (os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
                 "/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi"):
        if os.path.isfile(path):
            return path
    return None


def driver_major(text: str) -> int | None:
    """'617.42' (Windows) or '580.65.06' (Linux) -> 617 / 580."""
    match = re.search(r"(\d+)\.\d+", text or "")
    return int(match.group(1)) if match else None


def update_hint() -> str:
    if sys.platform == "win32":
        return ("Update it from https://www.nvidia.com/en-us/drivers/ "
                "(or the NVIDIA App), restart the PC, then press Install again.")
    return ("Install driver 580 or newer (on Ubuntu: sudo ubuntu-drivers install, or "
            "https://www.nvidia.com/en-us/drivers/), reboot, then press Install again.")


def check(version_text: str | None) -> tuple[bool, str]:
    if version_text is None:
        return False, (f"{FAIL} no NVIDIA driver was found (nvidia-smi is missing). "
                       "OS Studio needs an NVIDIA graphics card (RTX 20 series or newer) "
                       "with driver 580 or newer. " + update_hint())
    major = driver_major(version_text)
    if major is None:
        return True, f"{PASS} (could not read the driver version '{version_text.strip()}'; continuing)"
    if major < MIN_DRIVER:
        return False, (f"{FAIL} your NVIDIA driver is version {version_text.strip()}, and OS Studio "
                       f"needs {MIN_DRIVER} or newer (its CUDA 13 build requires it). " + update_hint())
    return True, f"{PASS}: driver {version_text.strip()} (needs {MIN_DRIVER}+)"


def read_driver() -> str | None:
    smi = find_nvidia_smi()
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    return lines[0] if out.returncode == 0 and lines else None


def main() -> int:
    ok, message = check(read_driver())
    print(message, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
