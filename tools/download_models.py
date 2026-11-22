"""Make sure the song model this PC will use is on disk; reuse existing weights.

12 GB+ cards get the full YuE2-3B + VAE (about 7.8 GB); smaller cards get the
single-file INT8 model (about 4 GB). Lyrics and voice models download on first
use inside the app.
"""
from pathlib import Path
import sys

APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))
import library  # noqa: E402  (reads app/config.json)

total_gb = 99.0
try:
    import torch
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
except Exception:
    pass

print(f'Models folder: {library.MODELS_DIR}', flush=True)
if library.pick_song_model(total_gb) == library.INT8_FILE:
    print(f'{total_gb:.1f} GB graphics card: using the INT8 song model.', flush=True)
    library.download_int8_song_model()
else:
    library.download_full_song_model()
print('Song model ready. Lyrics and voice models download on first use if missing.', flush=True)
