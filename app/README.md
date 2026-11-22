# OS Studio

Create. Sing. Listen. with open-source song models (YuE2 by default; ACE-Step 1.5 as an experimental, lower-quality option). The Pinokio listing is **OS Studio**; OS means Open Source. Model and dependency licenses still apply. Existing project folder names are preserved.

Type an idea, get a finished song: lyrics, music, vocals, word-timed karaoke
lyrics — optionally re-sung in **your own voice**. Runs entirely on your PC.

It is the ComfyUI workflow `yue2_song_to_lrc_qwen3_my_voice_native_yue2` turned
into a standalone app: same models, same settings, no ComfyUI server and no
node graph.

| Stage | Model |
|---|---|
| Lyrics writer | Qwen3-VL-4B-Instruct (or its 4-bit / 8-bit forms, or Qwen3-VL-2B — pick in step 1) |
| Song | YuE2-3B — the full model, or the small INT8 single file (pick in step 3; Auto chooses by graphics card) |
| Lyric timing | SongLRC + WhisperX word alignment |
| Voice clone | Seed-VC + Hybrid Demucs (train once from a talking clip) |

## Requirements

- Windows or Linux, NVIDIA RTX 20 series or newer, NVIDIA driver 580+
  (Install checks this first). Pick the song model in step 3 (Auto does it for you):
  - **12 GB and up:** *YuE2 3B · full* — 10.6 GB peak, a 100 s song in about 55 s.
  - **8 GB:** *YuE2 3B · INT8* (`yue2_3b_int8_convrot.safetensors` in
    `models\checkpoints`, 3.7 GB, downloaded on first use if missing) — 4.5 GB
    of model memory, 6.5 GB reserved, about 85 s for the same song.
  - The lyrics writer follows the same rule: under 11.5 GB, Auto uses
    *Qwen 4B · 4-bit* (4.1 GB, same speed) instead of the full 8.4 GB model.
    Only one model is on the card at a time, so 8 GB cards can run everything.
  - Tested on an RTX 5080 only; the small-card numbers come from hard-capping
    its memory, not from a real 8 GB or 12 GB card.
- Python 3.10 (setup creates its own `env\`; nothing else on the PC is touched).
- The model weights. The Pinokio launcher reuses a `YuE2\YuE2-3B` folder that another
  Pinokio app already has, otherwise it downloads the song model during Install
  (about 7.8 GB full, or 4 GB INT8 on smaller cards); it also downloads on the first
  song if still missing. Lyrics and voice models download on first use.

## Install and run

```bat
setup\setup.bat     :: once: creates .\env and installs requirements.txt (~4 GB)
start.bat           :: opens OS Studio in its own window
```

Keep the black console window open while you use the app; close it to quit.
`start.bat` uses `.\env` when it exists and falls back to the Pinokio
environment otherwise.

## Using it

1. **What's the song about?** Type an idea and press *Write lyrics* (Ctrl+Enter).
   The lyrics stream in as they are written. Or skip this and paste your own.
2. **Lyrics** — first choose the **song structure**: a length preset (Short,
   Standard, Long, Verses only) or your own order of intro / verse /
   pre-chorus / chorus / bridge / outro, and how many lines each section gets.
   The estimate next to it shows about how long the song will be. *Write
   lyrics* follows that structure exactly. Then edit freely; section tags go
   on their own line: `[verse]` `[chorus]` `[bridge]` …
3. **Style** — pick a preset or describe the sound in your own words.
   *Save to my styles* keeps a sound you made, and *Delete this style* (or the
   trash button in *Browse & manage styles*) removes one, with Undo. *Make a
   style with AI* has the lyrics writer (Qwen) turn a few words like "dreamy
   late-night R&B" into a style prompt for the song model you picked.
4. **Who sings it?** — the AI singer, or one of your trained voices.
   *New voice* teaches it a voice from 30 s – 3 min of someone talking
   (no music): record it right there with your microphone, or choose a file.
   That takes about 2.5 minutes and happens once per voice. The × on a voice
   deletes it (to the Recycle Bin).
5. **Make the song.** When it is done it loads into the player with synced
   lyrics — click any line to jump there. Songs sung in your voice keep the
   AI-singer version too; switch between them in the player.

*More options* holds the song title, the seed, and melody planning.
*Remix* on any song loads its lyrics and settings back into the composer.

Finished songs live in `output\songs\` as `Title_00001.mp3` with a matching
`.lrc` (timed lyrics), `.json` (the settings used) and, for cloned voices,
`.ai.mp3` (the original AI singer).

## How long things take (RTX 5080)

| Step | Time |
|---|---|
| Write lyrics | ~1 min to wake the writer, then 15–30 s |
| Make a song (1.5–2 min long) | 1–2 min |
| …sung in your voice | add ~1¼ min |
| Learn a new voice | ~2.5 min, once |

The lyrics writer and the song model cannot share a 16 GB card, so the app
swaps them in and out; that swap is the "waking up" wait.

## If it is slow

A song needs about 11 GB of free VRAM. If other programs are holding the card
(ComfyUI, a game, lots of video tabs) Windows quietly spills into normal RAM
and a 1-minute render becomes 10. The top bar shows how full the card is, and
the app warns you when a song starts without enough room.

## Layout

```
app.py            web server + launcher        web/        the page (HTML/CSS/JS, no build step)
service.py        the pipeline                 engine/     vendored ComfyUI core + YuE2/LRC/voice nodes
library.py        song list, config            shims/      folder_paths + Windows fixes
config.json       paths, port, window mode     setup/      setup.bat, get_weights.py
requirements.txt  tested package versions      output/     your songs
scripts_test/     api_e2e.py (full test), bench_song.py, dump_deps.py
legacy/           the old Gradio app, kept for reference
```

`config.json` options: `port` (default 7861), `window` (`"app"` = own window
via Edge, `"browser"` = a tab in your default browser), `models_dir`,
`output_dir`.

## Checking it works

With the app running:

```bat
env\Scripts\python scripts_test\api_e2e.py --voice my_voice
```

runs lyrics → song → lyrics → song-in-your-voice through the same API the page
uses and prints the timings. `logs\studio.log` has the engine output; the
≡ button in the app shows it live.

## Notes

- `shims/yue2_standalone_patch.py` materializes YuE2's lazily-shared weights
  so torch cu130 on Windows does not crash moving them between RAM and GPU.
- MP3s are 320 kbps constant bitrate so the lyric sync stays exact when seeking.
- Trained voices are stored in this app's own `voices\` folder (`voices_dir` in
  `config.json`), separate from the model weights. A new install starts with none.
- First use of lyric timing and of voice cloning downloads a few small helper
  models (alignment, vocal separation); that needs internet once.
