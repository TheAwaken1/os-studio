import sys, time
sys.path.insert(0,'shims'); sys.path.insert(0,'.')
import faulthandler, torch
faulthandler.enable()
import service
from service import _ensure_yue2
from yue2_repo.models.yue2 import force_attention_backend
from yue2_repo.models import yue2 as ym
import folder_paths

BACKEND = sys.argv[1] if len(sys.argv) > 1 else "auto"
force_attention_backend(BACKEND)

pipe = _ensure_yue2()
# inspect internals to find the AR
import yue2.cuda_graph as cg
print("GraphAR patched:", getattr(cg, "_comfy_yue2_forced_backend", None), flush=True)

style = 'English, rock, strong male vocals, electric guitar, 120 BPM'
lyrics = "[verse]\nshort test line one\nshort test line two\n\n[chorus]\ntest the chorus here\nmore chorus words\n\n[outro]\nend it now"

from yue2_repo.nodes.yue2 import YuE2Sampler
t0 = time.time()
(audio, abc, info) = YuE2Sampler().generate(
    pipeline=pipe, style=style, lyrics=lyrics, cot="off",
    seed=999, cfg_scale=-1, ode_steps=32,
    abc_max_tokens=1024, semantic_max_tokens=2000,
    abc_temperature=0.7, semantic_temperature=1.0,
    save_flac=False, save_abc=False, vae_decode="tiled", vae_tile_frames=0,
    abc=None, abort_after_plan=False, sampling_settings=None, save_artifacts=False)
print("GEN_OK", time.time()-t0, audio["waveform"].shape, flush=True)
