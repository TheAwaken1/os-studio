"""Ground truth: does YuE2Sampler still work on the LIVE ComfyUI engine?"""
import json, sys, time, urllib.request

API = "http://127.0.0.1:8188"

wf_path = r"E:\awaken\workflows\YuE2\SongLRC-Voice\yue2_song_to_lrc_qwen3_my_voice_native_yue2.json"
wf = json.load(open(wf_path, encoding="utf-8"))

# graph -> api map: keep only nodes we need, wire by [id, slot]
# Nodes: 26 loader, 27 sampler (lyrics link from node2). We bypass lyrics nodes
# by giving node 2 a fixed lyrics text and node 27 lyrics from node2 clean out.
prompt = {
    "2": {"class_type": "SongLyricsClean",
          "inputs": {"text": "[verse]\ntiny test verse line one\nsecond line here\n\n[chorus]\nchorus line one\nchorus line two\n\n[outro]\nbye"}},
    "26": {"class_type": "YuE2Loader",
           "inputs": {"model": "YuE2-3B", "vae": "YuE2-Vae", "device": "cuda",
                      "memory_budget_gib": 16, "offload_ar": True, "offline": True,
                      "attention_backend": "auto", "quantization": "none"}},
    "27": {"class_type": "YuE2Sampler",
           "inputs": {"pipeline": ["26", 0], "abc": "", "sampling_settings": None,
                      "lyrics": ["2", 0],
                      "style": "English, rock, strong vocals, 120 BPM", "cot": "off",
                      "seed": 999, "control_after_generate": "fixed",
                      "cfg_scale": -1, "ode_steps": 32, "abc_max_tokens": 1024,
                      "semantic_max_tokens": 2000, "abc_temperature": 0.7,
                      "semantic_temperature": 1.0, "save_flac": False,
                      "save_abc": False, "vae_decode": "tiled",
                      "vae_tile_frames": False, "abort_after_plan": False,
                      "save_artifacts": False}},
    "7": {"class_type": "SaveAudioAdvanced",
          "inputs": {"audio": ["27", 0], "filename_prefix": "audio/songs/probe_standalone",
                     "format": "mp3", "format.quality": "V0"}},
}

data = json.dumps({"prompt": prompt, "client_id": "probe-standalone"}).encode()
req = urllib.request.Request(API + "/prompt", data=data,
                             headers={"Content-Type": "application/json"})
pid = json.load(urllib.request.urlopen(req))["prompt_id"]
print("prompt_id", pid, flush=True)

t0 = time.time()
while True:
    hist = json.load(urllib.request.urlopen(API + f"/history/{pid}")).get(pid)
    if hist:
        st = hist.get("status", {})
        print("status", st.get("status_str"), "completed", st.get("completed"),
              f"{time.time()-t0:.0f}s", flush=True)
        if st.get("status_str") != "success":
            print(json.dumps(hist, indent=1)[:4000])
        break
    if time.time() - t0 > 900:
        print("TIMEOUT", flush=True)
        break
    time.sleep(5)
