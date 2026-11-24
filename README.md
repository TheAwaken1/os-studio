# OS Studio — Pinokio launcher



https://github.com/user-attachments/assets/fbbdc2db-596c-4fd4-9fa3-b6920026a5d9



**OS Studio**: OS means Open Source. Create. Sing. Listen. with open-source song models: **YuE2** (default) and **ACE-Step 1.5** (experimental, lower quality). The name does not replace the model/dependency licenses described below.

Windows or Linux, NVIDIA 8 GB+. Write lyrics, create a song, sync lyrics, and re-sing it using a voice you have permission to use.

OS Studio has two modes: **Create** for making music and **Listen** for playing your saved songs. The composer includes a guided lyrics editor, target-length controls, a searchable style library, microphone recording, and saved voice-reference previews. The player supports a continuous queue, shuffle/repeat, a mini-player, and supported phone lock-screen controls.

## Use
1. Open **OS Studio** in Pinokio. Click **Install** once, then **Start**.
2. Click **Open OS Studio** when the local server is ready.
3. Use Pinokio's native Stop control on the running start script to release the server/GPU.

### Create a song
Create walks you through five steps. Only the step you are on is open; the ones after it are greyed out until the step before is finished, and finished steps fold up to one line with a **Change** link.

1. **What's the song about?** Type an idea and press **Write lyrics**, or press **I have my own lyrics**.
2. **Lyrics.** Edit the words. To get a different draft, pick a song shape and optional target length, then press **Write lyrics** again.
3. **Pick the sound.** Choose a quick style, open **Browse more styles**, or describe your own.
4. **Who sings it?** Choose the **AI singer** or a trained voice. With no voices yet, **Add my voice** records or uploads a reference and trains one. Or switch to **No vocals (instrumental)**.
5. **Make the song.** Check the recap, pick how many **Takes** you want (1–4), and press **Make the song** (or **Make 3 takes**). Each take is a new performance of the same song: same lyrics, style and singer, with a new melody, arrangement and delivery. When a title is already used, the new song is numbered: "I'm a Roomba Not a Horse", then "I'm a Roomba Not a Horse 2", "… 3" (this applies to takes, remakes and remixes alike). Takes render one after another, each is saved to your songs as it finishes, and Cancel stops the rest. When it finishes you get **Listen now**, **Make another take** and **Start a new song**.

Finished songs are kept in **Listen** with timed lyrics and downloads. Voice-converted songs keep the original AI-singer version too; switch versions in the player.

### Covers (SheetSage2)
**Cover a song** in step 1 makes a new version of a song that follows its melody, in any style and with any singer, including your own trained voice.

1. Choose a song file (MP3, WAV, FLAC, OGG, M4A, AAC or Opus) or one of your songs, and how to get the lyrics:
   - **Write down its lyrics:** the vocals are separated and Whisper writes the words down by ear. This adds about 2 minutes, and the words need checking (it misheard "Shih Tzu" in testing).
   - **Keep its lyrics:** for your own songs, which already know their words.
   - **I'll write new lyrics:** step 2 gets the song's sections to fill. **Write lyrics** follows that shape.
2. **Listen to the song.** SheetSage2 listens once and writes down the melody, chords, key, tempo and sections as a score: about 10 seconds for a 2-minute song, under 2 GB of graphics memory. The lyrics land in step 2, sorted into the song's sections, and a banner shows what is being covered.
3. Pick any sound and singer and make the song as usual. Takes, Remix, instrumentals and your own voice all work with covers.

The banner chooses what the cover keeps: **Melody** (the default, recommended by SheetSage2's authors; the band plays its own chords) or **Melody + chords**. **Stop covering** goes back to a normal song. In testing, a cover in a different genre kept the original's key, tempo and length (96.5 s against 96 s) and matched 0.69 of its melody, against 0.26 for the same lyrics without the score. Covers always use YuE2; ACE-Step can't follow a score.

Analyses are kept in `app/input/covers/` (git-ignored) and listed under **Analysed before** in the cover dialog, so a song only has to be listened to once; the × there deletes one. Deleting an analysis never deletes your songs. **Covers of other people's songs are for personal use unless you have the rights to them.**

### Instrumentals, music only and vocals only
- **Make an instrumental:** press **Instrumental (no lyrics)** in step 1, or pick **No vocals (instrumental)** in step 4. Lyrics are not needed: the guide goes straight to the sound, and steps 1 and 2 show **Skip to the sound**. Anything typed in step 1 becomes the title (e.g. "late night rain" → "Late Night Rain"). YuE2 always sings something (with blank lyrics it makes up its own words), so the app renders the track and then takes the vocals out with the same separator the voice cloning uses. The result is music only, a few seconds slower than a normal song. Your lyrics and singer choice stay saved; **Add vocals back** switches it off.
- **Split any song:** **Music & vocals** in the player saves the song's music and its vocals as two extra versions (a few seconds, about a minute the first time while the separator loads). Switch between **Full song**, **Music only** and **Vocals only** at the top of the player; the MP3 button downloads the one you are on, and the timed lyrics keep running, so Music only works as karaoke.
- Songs sung in a trained voice get **Music only** and **Vocals only** automatically, since they are split anyway.
- With melody planning on (More options), the song's melody score is also saved next to it as `Title_00001.abc`, ready for remixes and covers in a later version.

**Step by step: on/off** (top right of Create) shows every step at once for people who prefer that; the choice is remembered in this browser. The lyrics-writer and song-model pickers are under **Advanced** in steps 1 and 3.

The app handles one GPU job at a time. Every time the app opens or the page is refreshed, Create starts clean at step 1 with nothing typed. If you left a song unfinished, a **Continue that song** bar brings it back exactly where you were (or **Dismiss** forgets it); a song you already made is not offered again. Only the lyrics writer, song model and graphics-memory picks carry over. This is kept in the current browser and is not synced between your PC and phone.

### Remix
**Remix this song** (in the player) or **Remix** (in the song list) starts a new song from an old one. It asks what you want to change and opens that step in Create with everything else already filled in:

- **New take** — same lyrics, sound and singer; goes straight to Make the song.
- **Change the lyrics**, **Change the sound**, **Change the singer** — open step 2, 3 or 4.

The original song is never changed. A remix always uses a new random seed. **Start a brand-new song instead** leaves the remix and clears the draft.

### Guided lyrics and song structure
Step 2 shows your lyrics first (**Make it yours**), then **Want a different draft?** with the song shape, target length and **Write lyrics**. Short / Standard / Long / Verses only choose the section layout, not an exact audio duration. **Customize structure** opens the advanced controls for adding, removing, and dragging sections. **Editing tools** contains section-tag insertion, Clear, and Undo.

Changing a shape or target does not rearrange or replace existing lyrics. Clicking **Write lyrics** deliberately replaces the draft; Undo in Editing tools can restore the previous draft. The Long shape includes an instrumental `[intro]`: its tag is retained, but it has no lyric lines. To control lines per section manually, leave target length on Auto.

### Target song length
Step 2 keeps the existing Short / Standard / Long song shapes and adds **Target song length**: Auto, 1 min, 2 min, 3 min, or Custom minutes/seconds. Auto preserves the original lines-per-section behavior. A target adjusts Qwen's next draft using a per-section line plan; it never edits pasted/current lyrics just because you change the control. The preview shows the planned length and a separate rough estimate for your current words. Tempo and instrumental passages affect the result, so this is not an exact-duration guarantee, an audio crop, or a loop/padding feature.

Custom targets range from 0:30 to 6:00, subject to the chosen memory profile: Low allows up to 3:00, Standard up to 6:00, and Auto follows the card's memory profile. Invalid/over-limit targets must be corrected before Write lyrics starts. Targets, custom fields and existing lyrics stay in the browser's saved draft. To set section line counts manually, choose Auto and open Customize structure. The instrumental intro and repeated-chorus behavior remain unchanged.

### Styles without a crowded composer
The original quick-pick styles stay visible in Step 3. **Browse & manage styles** opens a searchable library with category filters, including Rock & Heavy and Electronic & Dance. Choosing a library entry fills the style field; **Type your own** lets you describe a custom sound.

### My styles, deleting styles, and AI-made styles
- **Save to my styles**: when the style box holds a sound of your own, name it and save it. Saved styles show up first among the quick picks (marked ★) and in the library under *My styles*.
- **Delete**: *Delete this style* under the chips removes the selected one; the trash button next to each entry in *Browse & manage styles* does the same. Every delete offers **Undo**. A deleted built-in style is only hidden, and *Bring them back* in the library restores all of them.
- **Make a style with AI**: describe the vibe in plain words, for example "dreamy late-night R&B with a soft female singer". The lyrics writer (Qwen, the same model as *Write lyrics*) turns it into a style prompt worded for the song model you picked: short tags for YuE2, or a plain description with a key for ACE-Step. The first use loads the writer, which takes about a minute; after that a style takes a few seconds. The result lands in the style box with a suggested name, so you can edit it and save it.
- Saved and deleted styles live in `app/my_styles.json`. The file is git-ignored, and Update and Reset leave it alone. The shipped `style_prompts*.txt` files are never changed.

Extra entries come from `app/style_prompts.txt`. The library excludes duplicate names/prompts already present in the quick picks or earlier library entries, while retaining distinct subgenres such as Country-rock. Prompts are preserved as written. Entries that request **no vocals** show an instrumental warning rather than silently changing the prompt.

### Listen mode and Pinokio Remote
Switch to **Listen** for the music player, then tap **Play all** or a saved song. Each song gets its own cover colours, live sound bars and a waveform of the whole song; tap or drag the waveform (or the slider under it) to jump. Continuous play is on by default; use Previous/Next, Shuffle, and Repeat off/all/one to control the rotation. Create mode remains available without stopping the music. A mini-player follows you in Create mode and when the main playback controls scroll offscreen in Listen mode. Player preferences are saved in this browser; reloading never autoplays.

To listen on your phone, turn on **Remote** in Pinokio, then scan the QR code with your phone (or open the https link it shows) and open OS Studio. Music streams from your PC, so keep it awake with Pinokio and OS Studio running; songs are not downloaded to the phone. Pinokio's Remote guide: https://desktop.pinokio.co/docs/remote/. The app itself stays bound to 127.0.0.1; no public port is opened.

The player registers supported Media Session controls/metadata for the OS and plays music through a plain audio element on every device. The sound bars and waveform are drawn from loudness and pitch data the server measures from the song file (`GET /api/songs/{id}/viz`, cached under `app/output/temp/viz`), looked up at the current playback time. They do not use the browser's Web Audio or animation frames, so they behave the same on a phone, in a browser and in the Pinokio window. The first play of a song measures it (well under a second); with reduced-motion turned on in the OS the bars update a few times a second instead of smoothly. Visual updates sleep when the page is hidden, without deliberately pausing audio. Lock-screen metadata uses the OS Studio name and full-bleed, 1024 × 1024 amber-to-mint music-note artwork, without a baked-in black border.

**Continuous playback and next-track handoff while a real iPhone is locked remain device-test dependent, not guaranteed.** First tap Play in the foreground; then test a track boundary with the phone locked over your actual Remote connection. Interrupted audio requires a foreground Play tap. If the phone still shows old branding/artwork after an update, refresh the app page and start playback again.

### Singer type

In **Who sings it?**, pick **Female**, **Male**, **Duet**, **Choir**, or **Describe your own** (e.g. "raspy female alto"). Any male/female words in the chosen sound are swapped for your pick. **Match the sound** keeps the style prompt as written. When a cloned voice is selected, the singer type sets the AI take that gets re-sung, so pick the one closest to your own voice.

### New voice: recording type
Record a reference directly in the app or upload an existing recording. Use one person's clear voice, preferably 30 seconds to 3 minutes, without background music or other speakers. Train only voices you own or have permission to use. Training happens once per saved voice; the result can be selected for later songs.

Choose **Talking**, **Singing**, or **Rapping** for the reference clip before training. This describes the recording, not the style of the song you want to generate. The choice is saved in that voice's JSON metadata. Talking keeps the speech-to-song pitch offset; singing and rapping use the existing performance-reference path without that extra offset. These are not three separate voice models. Existing voices with no saved type keep the original Talking behavior; they are not automatically reclassified or retrained.

### Listen to a saved voice reference
Use the separate listen button next to a saved voice in Step 4. It opens the voice's saved reference recording without changing the voice selected for the song or starting another training job. The button is unavailable when that voice has no saved reference recording.

**This preview is the original reference recording, not a generated singing demo or a guarantee of cloning quality.** Close the preview to return to the composer. Deleting a shared trained voice affects the other apps that use the same voice folder; see the file-safety notes below.

### Re-clone voice & studio vocal separation
Voice cloning and **Music & vocals** separation use **BS-RoFormer** (vocals vs. instrumental) followed by **Mel-Band RoFormer Karaoke** (lead vocals vs. chorus harmonies), with automatic fallback to Hybrid Demucs if RoFormer weights cannot be downloaded. Only the lead vocal goes through Seed-VC while chorus harmonies and backing vocals are kept intact and blended back in, then finished with studio EQ, gentle compression, stereo plate reverb, and a soft-clip bus limiter.

Use **Re-clone voice** in the player on any vocal song to re-sing it from its preserved `.ai.mp3` take in any trained voice — or tweak vocal volume, chorus harmony level, reverb, studio EQ/compression, voice likeness, diffusion steps, CFG, and pitch matching — without regenerating the song. Verified BS-RoFormer and Karaoke stems are cached under `app/output/temp/stems/` (LRU-pruned) so repeat mix adjustments skip separation.

## Requirements and installation details
Python 3.10 and the app's pinned CUDA 13.0 PyTorch build are installed into `app/env`. The original project's environment is not used or copied. Optional `uv_executable` and `uv_cache_dir` keys in the ignored `settings.local.json` can reuse an existing compatible UV package cache during installation; packages are copied into the isolated environment (`UV_LINK_MODE=copy`), not shared with the original app at runtime. Without these keys, Pinokio's managed UV and cache are used. Runs on **Windows and Linux with an NVIDIA card**: RTX 20, 30, 40 or 50 series (GTX 10 series and older are not supported by CUDA 13), with **NVIDIA driver 580 or newer**. Install checks the driver first and stops with a clear message if it is too old, before downloading anything big. Linux was tested on Ubuntu 24.04 (WSL2, RTX 5080): a fresh install, lyrics, a song, and a song in a cloned voice all worked. On Linux, *Show file* opens the songs folder with `xdg-open`; without a desktop it tells you where the folder is instead. macOS, AMD and CPU-only machines are not supported: the song engine needs CUDA. Hardware requirements depend on model selection; the INT8 option is the smaller-card route.

## Your files are safe
- App code lives in `app/`. Everything personal is git-ignored and never touched by **Update** or **Reset**:
  - songs in `app/output/songs/` (each song's `.mp3` plus its `.lrc`, `.json`, and when present `.abc`, `.ai.mp3`, `.instrumental.mp3` and `.vocals.mp3`; deleting a song deletes all of them)
  - recordings in `app/input/`
  - trained voices in `app/voices/` (each voice is a `.pth`, `.wav` and `.json` with the same name)
  - settings in `app/config.json`, generated on first Install and kept afterward
  - your saved and deleted styles in `app/my_styles.json`
- A fresh install starts with **no voices**. Train your own in the app.
- **Reset dependencies only** removes `app/env` and its install marker, never songs, models, voices, or configuration. Install again afterward.
- Do not run GPU jobs in two apps at the same time.

## Models
Models download automatically; nothing has to be fetched by hand.

- **Reuse first.** On first Install the launcher looks through your other Pinokio apps (`<PINOKIO_HOME>/api/*/app/models` and `*/models`) for an existing `YuE2/YuE2-3B` folder and uses that models folder in place. Weights are referenced, not copied. If none is found, models go to `app/models`. To pick a folder yourself, put `{"models_dir": "D:/path/to/models"}` in `settings.local.json` before the first Install.
- **Song model.** Install finishes by making sure the song model is on disk: the full YuE2-3B + YuE2-Vae (about 7.8 GB, from `m-a-p` on Hugging Face) on 12 GB+ cards, or the single-file INT8 model (about 4 GB) on smaller cards. If it is still missing when you make a song, it downloads then. **Download missing song models** in the Pinokio menu does the same on demand.
- **ACE-Step 1.5 XL (experimental)** is a second song engine, picked in step 3's *Song model* list. **It doesn't come out as good as YuE2** (the vocals sound noticeably lower quality), so YuE2 stays the default: Auto never picks ACE-Step, and the app goes back to Auto each time it opens. It reuses ComfyUI's `diffusion_models/ace-step1.5`, `text_encoders/ace-step1.5` and `vae/ace-step1.5` files if your models folder has them, otherwise the first ACE song downloads about 20 GB from `Comfy-Org/ace_step_1.5_ComfyUI_files`. It runs the same settings as ComfyUI's ACE-Step 1.5 XL SFT workflow (50 steps, cfg 7, euler/simple, shift 3); the song length comes from your lyrics and the BPM from your style text (`110 BPM`), otherwise 120. Timed lyrics and your trained voices work with it unchanged. Melody planning is YuE2-only. Picking ACE-Step swaps the style presets and style library to versions written for it (`app/style_prompts_ace.txt`; YuE2's are in `app/style_prompts.txt`); a chosen preset follows the switch by name, and a style you typed yourself is left alone. A key in the style text (`E minor`) and a time signature (`6/8`) are passed to ACE as song settings.
- **Lyrics writers, voice models and lyric timing** download the first time you use them. Voice separation downloads SHA-256-verified BS-RoFormer (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`) and Mel-Band RoFormer Karaoke (`mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt`) into `<models>/seed_vc/roformer/` on first use (falling back to Hybrid Demucs if unavailable).
- **Covers** download two models the first time: SheetSage2 (`audio_encoders/sheetsage2_bf16.safetensors`, 1.4 GB, from `Comfy-Org/YuE2`, reused if your models folder has it) and, only if you ask it to write lyrics down, Whisper large-v3 turbo (about 1.6 GB, from `openai/whisper-large-v3-turbo`, kept in the Hugging Face cache).
- Seed-VC's sample clips, which voice training mixes in as "other voices", are fetched from the pinned upstream Seed-VC commit on first training instead of being shipped here.
- FFmpeg is installed with Conda if it is missing.

## License
OS Studio is released under the **GNU General Public License v3.0** — see [LICENSE](LICENSE). It bundles GPLv3 code (Seed-VC and parts of ComfyUI), so changes you share must stay GPLv3 as well. The license covers this code only, not the model weights below.

## Third-party licensing — before sharing
The YuE2 checkpoint weights are **CC BY-NC 4.0** (see `YuE2-3B/LICENSE`): no commercial use. The SheetSage2 weights used for covers (`m-a-p/SheetSage2`) are also **CC BY-NC 4.0**. Whisper and the BS-RoFormer / Mel-Band RoFormer inference code (based on lucidrains/BS-RoFormer and ZFTurbo/Music-Source-Separation-Training) are MIT-licensed. Vendored Seed-VC is GPLv3 and the vendored ComfyUI core carries GPLv3-or-later notices; original notices are preserved. This repository does not relicense those components or authorize commercial use of the weights.

## Updates
**Update** in the Pinokio menu runs `git pull` on this launcher and then re-runs Install, which picks up new dependencies and any missing models. Songs, voices, models and `app/config.json` are git-ignored, so updates never touch them.

**Update notice:** a few seconds after the app opens, it asks GitHub once whether this repository has newer commits than your install (the answer is cached for 6 hours). If it does, a banner says so, links to the release notes, and reminds you that Update shows in Pinokio's menu once you stop OS Studio. **Later** hides it until the next new version. The request sends only the public repository name and your install's commit id. Offline, or with no git checkout, nothing is shown. To turn it off, add `"update_check": false` to `app/config.json`.

Release notes for each version are on the [Releases page](https://github.com/TheAwaken1/os-studio/releases); press **Watch → Custom → Releases** on GitHub to be emailed when one comes out.

## Troubleshooting
- Launcher logs: `logs/api/`; engine log: `app/logs/studio.log`.
- Missing NVIDIA CUDA: update the NVIDIA driver and rerun Install.
- Missing models: check `app/config.json`, then Download missing song models.
- Slow generation: stop other GPU apps. Changing to a Pinokio launcher does not lower the model's GPU memory requirement.
- Microphone: use the local Open OS Studio URL and allow microphone access. Uploaded recordings remain an alternative.
- New controls are visible but length planning says unavailable: finish any running job, stop/start OS Studio in Pinokio, then refresh the page. Backend changes require an app restart; static-only branding/artwork changes normally need a page refresh.
- Phone playback stops: check that the serving PC is awake, the app is running, and the phone's Remote connection is still working. Return to the app and tap Play. This player does not provide offline listening.

## Local API examples
Use the actual URL shown by Pinokio; the port is selected automatically. No authentication is provided: keep the app bound to loopback and do not expose it publicly.

JavaScript (from the app's browser console):
```js
const state = await fetch('/api/state').then(r => r.json());
const job = await fetch('/api/job').then(r => r.json());

// Read-only length preview: does not start a GPU job or edit lyrics.
const plan = await fetch('/api/lyrics-plan', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    structure: ['verse', 'chorus', 'verse', 'chorus'],
    lines: 4, target_seconds: 150, memory: 'standard'
  })
}).then(r => r.json());
```

Python:
```python
import json, urllib.request
base = 'http://127.0.0.1:YOUR_PORT'
state = json.load(urllib.request.urlopen(base + '/api/state'))
body = json.dumps({'idea':'a song about a dog', 'structure':['verse','chorus'], 'lines':4}).encode()
request = urllib.request.Request(base + '/api/lyrics', data=body, headers={'Content-Type':'application/json'})
job = json.load(urllib.request.urlopen(request))
# Poll GET /api/job. Only one GPU job may run at a time.
```

Curl:
```sh
curl http://127.0.0.1:YOUR_PORT/api/state
curl http://127.0.0.1:YOUR_PORT/api/songs
curl -X POST http://127.0.0.1:YOUR_PORT/api/cancel
# Newer version on GitHub? {available, current, latest, behind, url, release:{tag,name,url}}
curl http://127.0.0.1:YOUR_PORT/api/update

# Styles: ask Qwen for one (a GPU job; poll /api/job, result.style = {name, style})
curl -X POST http://127.0.0.1:YOUR_PORT/api/style-ai -H "Content-Type: application/json" \
  -d '{"idea":"gritty 90s grunge with a raspy guy singing","engine":"yue2"}'
# Save one, delete one (yours are removed; built-ins are hidden), bring hidden ones back
curl -X POST http://127.0.0.1:YOUR_PORT/api/styles -H "Content-Type: application/json" \
  -d '{"name":"Night drive","style":"English, synthwave, breathy female vocals, 104 BPM","engine":"yue2"}'
curl -X POST http://127.0.0.1:YOUR_PORT/api/styles/delete -H "Content-Type: application/json" -d '{"name":"Pop"}'
curl -X POST http://127.0.0.1:YOUR_PORT/api/styles/restore -H "Content-Type: application/json" -d '{}'
curl http://127.0.0.1:YOUR_PORT/api/styles
```

Style endpoints: `GET /api/styles` returns `engine_styles` (`yue2` and `ace15`, each with `presets`, `style_library` and `hidden_count`). The save, delete and restore calls return the same thing. `POST /api/styles/restore` takes an optional `name` to bring back a single style. `POST /api/style-ai` takes `idea` (up to 400 characters), `engine` (`yue2` or `ace15`) and an optional `writer`.

`POST /api/lyrics-plan` returns per-section `counts`, `total_lines`, `estimated_seconds`, `target_seconds`, and `limit_seconds`. Send an integer `target_seconds` from 30 to 360, or null/omit it for Auto, plus `memory` (`auto`, `low`, or `standard`). Over-limit targets are rejected; an estimate is not a promised final song duration. The same optional target/memory fields are accepted by `POST /api/lyrics` when starting a draft.

Covers: `POST /api/covers?name=Song&lyrics=transcribe` with the audio file as the raw body (header `X-Filename: song.mp3`) starts an analysis job; `lyrics` is `transcribe` or `none`. `POST /api/covers/song/{id}?lyrics=song` analyses one of your songs (`song` reuses its lyrics). The job result's `cover` has `id`, `key`, `bpm`, `meter`, `sections` (tag, start, end, sung, lines) and `lyrics`. `GET /api/covers` lists analyses; `DELETE /api/covers/{id}` forgets one. Then `POST /api/song` with `"cover_id": "<id>"` and `"cover_follow": "melody"` (or `"full"` for melody + chords) makes the cover.

```sh
curl -X POST "http://127.0.0.1:YOUR_PORT/api/covers?name=My%20Song&lyrics=transcribe" \
  -H "X-Filename: my-song.mp3" -H "Content-Type: application/octet-stream" --data-binary @my-song.mp3
# poll /api/job until done, then:
curl -X POST http://127.0.0.1:YOUR_PORT/api/song -H "Content-Type: application/json" \
  -d '{"lyrics":"[verse]\n...","style":"English, country, acoustic guitar, warm male vocal","cover_id":"COVER_ID"}'
```

Instrumentals, splits and re-cloning: `POST /api/song` with `"instrumental": true` makes a music-only song (lyrics may be empty), and accepts an optional `"voice_settings"` object when cloning (`keep_backing`, `backing_gain_db`, `voice_strength`, `diffusion_steps`, `cfg_rate`, `pitch_mode`, `pitch_shift`, `vocal_gain_db`, `reverb`, `polish`). `POST /api/songs/{id}/reclone` (`{"voice": "<name>", "voice_settings": {...}}`) re-sings an existing song from its preserved `.ai.mp3` take with BS-RoFormer + Karaoke separation and studio mix polish (poll `/api/job`). `POST /api/songs/{id}/split` starts a GPU job that saves the song's music and vocals (poll `/api/job`; it returns `job: null` if both already exist). `GET /media/{id}?v=inst` and `?v=vocals` play them (`?v=ai` is the AI singer of a voice-cloned song); `GET /api/songs` reports `has_inst`, `has_vocals`, `has_score`, `instrumental`, `voice_settings`, `separator` and `updated` per song.

Other endpoints: `POST /api/song` accepts lyrics, style, voice and model settings, plus an optional `singer` (`female`, `male`, `duet`, `choir`, or free text such as `"raspy female alto"`; empty keeps whatever singer the style describes); `GET /api/voices` lists your trained voices; `GET /api/voices/{name}/preview` plays a saved reference recording, not a generated singing demo. Use only voices you own or are authorized to use.

## Validation
Run these from the Pinokio package directory, not from `app/`:

```sh
node --test tests/*.test.cjs
app/env/Scripts/python.exe -m unittest discover -s tests
```

The checks cover launcher/menu behavior, branding, artwork, queue behavior, the guided-step rules, measured player visuals, target-length validation/planning, voice-reference types, and non-destructive configuration. Backend unit tests use isolated/stubbed inference paths; passing them does not prove model quality or an exact song duration.

For browser checks, replace `<running-app-url>` with the URL currently shown by Pinokio. The scripts use Playwright and the installed Microsoft Edge browser on Windows:

```sh
uv run --with playwright python tests/browser_smoke.py --base <running-app-url>
uv run --with playwright python tests/target_length_browser.py --base <running-app-url>
uv run --with playwright python tests/workout_player_smoke.py --base <running-app-url>
uv run --with playwright python tests/recording_type_browser.py --base <running-app-url>
uv run --with playwright python tests/branding_browser.py --base <running-app-url>
uv run --with playwright python tests/guided_studio_browser.py --base <running-app-url>
```

- **Composer:** style search/categories, saved-reference audio playback, and mobile layout (run with every step open).
- **Guided steps, Remix and visuals:** first visit opens step 1 only, locked steps stay closed, Next through all five steps, progress survives a reload, Remix opens the chosen step, the finished-song panel (driven by a scripted job, not the engine), sound bars that keep changing with animation frames frozen and no Web Audio, waveform seeking, and phone layout. Lyrics and song POSTs are aborted or scripted; no GPU job runs.
- **Target length:** actual read-only planning API, preserved pasted lyrics, saved custom inputs, memory-limit validation, and the outgoing lyrics-request format. The test aborts the lyrics-generation POST before it reaches the server.
- **Player:** actual MP3 playback, real ended events, queue navigation/shuffle/repeat, mini-player visibility, supported Media Session handlers, and HTTP byte-range seeking. The test requires at least two saved songs.
- **Voice types:** Talking/Singing/Rapping request format; training POSTs are deliberately aborted.
- **Branding:** displayed OS Studio name, actual served artwork, Media Session metadata, and mobile header.

Run browser checks when no generation/training job is active. Some checks require existing saved songs or voice references. iPhone user-agent and simulated-visibility checks run in Chromium: they are not physical iPhone lock-screen or private Tailscale-network tests. The app's `scripts_test` folder contains additional checks; actual GPU pipeline tests write new songs and must be run deliberately.
