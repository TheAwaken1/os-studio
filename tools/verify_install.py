"""Check the isolated environment before showing Start in Pinokio."""
import importlib
import json
from pathlib import Path
import sys
import shutil
import time
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'app'
sys.path.insert(0, str(APP))
import torch
if sys.version_info[:2] != (3, 10):
    raise RuntimeError('This package needs Python 3.10.')
if not torch.cuda.is_available():
    raise RuntimeError('NVIDIA CUDA is not available. Check your NVIDIA driver before starting.')
if not shutil.which('ffmpeg'):
    raise RuntimeError('FFmpeg CLI is missing. Run Install in Pinokio to provision it.')
for name in ['fastapi', 'uvicorn', 'transformers', 'soundfile', 'lameenc', 'librosa', 'torchsde', 'whisperx']:
    importlib.import_module(name)
import library
marker = {'python':sys.version.split()[0], 'torch':torch.__version__,
          'gpu':torch.cuda.get_device_name(0), 'verified':time.time(),
          'models_dir':library.MODELS_DIR}
(APP / '.installed.json').write_text(json.dumps(marker, indent=2), encoding='utf-8')
print('OS Studio installation verified: ' + json.dumps(marker), flush=True)
