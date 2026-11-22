"""Bisect: offload_ar True vs False on the standalone runtime."""
import sys, time
sys.path.insert(0,'shims'); sys.path.insert(0,'.')
import faulthandler; faulthandler.enable()
import service
from service import _ensure_yue2, _yue2_state
from yue2_repo.nodes.yue2 import YuE2Sampler, YuE2Loader

OFFLOAD = sys.argv[1] == "1"
lyrics = "[verse]\ntiny test verse line one\nsecond line here\n\n[chorus]\nchorus line one\nchorus line two\n\n[outro]\nbye"
style = 'English, rock, strong vocals, 120 BPM'

(pipe,) = YuE2Loader().load(
    model="YuE2-3B", vae="YuE2-Vae", device="cuda",
    memory_budget_gib=16, offload_ar=OFFLOAD, offline=True,
    attention_backend="auto", quantization="none")
print("loaded, offload_ar=", OFFLOAD, flush=True)
t0 = time.time()
(audio, abc, info) = YuE2Sampler().generate(
    pipeline=pipe, style=style, lyrics=lyrics, cot="off",
    seed=999, cfg_scale=-1, ode_steps=32,
    abc_max_tokens=1024, semantic_max_tokens=2000,
    abc_temperature=0.7, semantic_temperature=1.0,
    save_flac=False, save_abc=False, vae_decode="tiled", vae_tile_frames=0,
    abc=None, abort_after_plan=False, sampling_settings=None, save_artifacts=False)
print("GEN_OK", round(time.time()-t0,1), tuple(audio["waveform"].shape), flush=True)
