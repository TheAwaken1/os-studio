"""Unit tests for RoFormer separation, vocal mix polish, voice settings validation, and re-clone API/service."""
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

APP = Path(__file__).resolve().parents[1] / "app"
for p in (str(APP / "shims"), str(APP / "engine"), str(APP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import app as web_app  # noqa: E402
import library  # noqa: E402
import service  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from song_lrc import song_lrc_node as lrc_mod  # noqa: E402
from song_voice import roformer  # noqa: E402
from song_voice.song_voice_node import (  # noqa: E402
    SongVoiceMix,
    _follow_original,
    _polish_vocals,
    _resolve_pitch_shift,
)


class VoiceSettingsTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        defaults = library.clean_voice_settings(None)
        self.assertEqual(defaults["keep_backing"], True)
        self.assertEqual(defaults["voice_strength"], 0.8)
        self.assertEqual(defaults["diffusion_steps"], 40)
        self.assertEqual(defaults["reverb"], 0.18)

        custom = library.clean_voice_settings({
            "keep_backing": False,
            "voice_strength": 0.75,
            "diffusion_steps": 50,
            "pitch_mode": "exact to my range",
            "pitch_shift": -2,
            "reverb": 0.25,
            "vocal_gain_db": 1.5,
        })
        self.assertFalse(custom["keep_backing"])
        self.assertEqual(custom["voice_strength"], 0.75)
        self.assertEqual(custom["diffusion_steps"], 50)
        self.assertEqual(custom["pitch_mode"], "exact to my range")
        self.assertEqual(custom["pitch_shift"], -2)
        self.assertEqual(custom["reverb"], 0.25)
        self.assertEqual(custom["vocal_gain_db"], 1.5)

    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):
            library.clean_voice_settings({"pitch_mode": "invalid"})
        with self.assertRaises(ValueError):
            library.clean_voice_settings({"voice_strength": 2.0})
        with self.assertRaises(ValueError):
            library.clean_voice_settings({"diffusion_steps": 5})
        with self.assertRaises(ValueError):
            library.clean_voice_settings({"reverb": -0.1})


class RoformerAndMixTests(unittest.TestCase):
    def test_small_bs_and_mel_roformer_forward_and_overlap_add(self):
        # Small synthetic instances validate architecture & overlap-add math without needing weights
        bs = roformer.BSRoformer(
            dim=16, depth=1, stereo=True, num_stems=1,
            time_transformer_depth=1, freq_transformer_depth=1,
            freqs_per_bands=(256, 256, 256, 257),
            dim_head=8, heads=2, flash_attn=False,
            stft_n_fft=2048, stft_hop_length=441, stft_win_length=2048,
            mask_estimator_depth=2, mlp_expansion_factor=2,
        ).eval()
        mel = roformer.MelBandRoformer(
            dim=16, depth=1, stereo=True, num_stems=1,
            time_transformer_depth=1, freq_transformer_depth=1,
            num_bands=8, dim_head=8, heads=2, flash_attn=False,
            stft_n_fft=2048, stft_hop_length=441, stft_win_length=2048,
            mask_estimator_depth=1, mlp_expansion_factor=2,
        ).eval()
        audio = torch.randn(2, 16000) * 0.1
        for model in (bs, mel):
            target, residual = roformer.separate_roformer(
                model, {"audio": {"chunk_size": 8820}}, audio, torch.device("cpu"), num_overlap=2
            )
            self.assertEqual(tuple(target.shape), tuple(audio.shape))
            self.assertEqual(tuple(residual.shape), tuple(audio.shape))
            self.assertTrue(torch.isfinite(target).all())
            self.assertTrue(torch.allclose(target + residual, audio, atol=1e-5))

    def test_follow_original_polish_and_mix_with_backing(self):
        sr = 44100
        t = torch.linspace(0, 1.5, int(sr * 1.5))
        orig = 0.25 * torch.sin(2 * 3.14159 * 220 * t)
        conv = 0.08 * torch.sin(2 * 3.14159 * 220 * t)
        matched = _follow_original(conv, orig, sr)
        self.assertEqual(matched.shape, conv.shape)
        self.assertTrue(torch.isfinite(matched).all())
        self.assertGreater(matched.pow(2).mean().sqrt().item(), 0.15)

        voc_st = matched[None].repeat(2, 1)
        polished = _polish_vocals(voc_st, sr, reverb_amount=0.2)
        self.assertEqual(polished.shape, voc_st.shape)
        self.assertTrue(torch.isfinite(polished).all())
        # Stereo reverb decorrelates L/R so the channels are no longer identical mono duplicates
        self.assertGreater((polished[0] - polished[1]).abs().max().item(), 1e-4)

        # Reverb must still apply when Studio EQ & compression is disabled (eq_and_comp=False)
        reverb_only = _polish_vocals(voc_st, sr, reverb_amount=0.2, eq_and_comp=False)
        self.assertGreater((reverb_only[0] - reverb_only[1]).abs().max().item(), 1e-4)

        mixer = SongVoiceMix()
        voc_audio = {"waveform": voc_st[None], "sample_rate": sr}
        inst_audio = {"waveform": (0.15 * torch.randn(1, 2, int(sr * 1.5))), "sample_rate": sr}
        back_audio = {"waveform": (0.10 * torch.randn(1, 2, int(sr * 1.5))), "sample_rate": sr}
        (mixed,) = mixer.mix(voc_audio, inst_audio, backing_vocals=back_audio, polish=True, reverb_amount=0.18)
        self.assertEqual(mixed["waveform"].shape, inst_audio["waveform"].shape)
        self.assertLessEqual(mixed["waveform"].abs().max().item(), 0.991)
        (voc_only,) = mixer.combine_vocals(voc_audio, backing_vocals=back_audio, polish=True, reverb_amount=0.18)
        self.assertEqual(voc_only["waveform"].shape, voc_audio["waveform"].shape)
        (voc_reverb_only,) = mixer.combine_vocals(voc_audio, backing_vocals=None, polish=False, reverb_amount=0.2)
        self.assertGreater((voc_reverb_only["waveform"][0, 0] - voc_reverb_only["waveform"][0, 1]).abs().max().item(), 1e-4)

    def test_octave_pitch_shift_avoids_male_sub_bass_drop(self):
        deep_male = torch.full((50,), 90.9)
        male_rock_1 = torch.full((50,), 191.5)
        male_rock_2 = torch.full((50,), 218.7)
        female_lead = torch.full((50,), 320.0)
        female_talk = torch.full((50,), 200.0)
        male_low = torch.full((50,), 125.0)

        # Deep baritone talking ref + male rock tenor should NOT snap down -12 st into slurred sub-bass
        self.assertEqual(_resolve_pitch_shift(deep_male, male_rock_1, "octave to my range", 0, 5.0), 0.0)
        self.assertEqual(_resolve_pitch_shift(deep_male, male_rock_2, "octave to my range", 0, 5.0), 0.0)
        # True female -> male and male -> female octave gaps still snap cleanly by +/-12 st
        self.assertEqual(_resolve_pitch_shift(deep_male, female_lead, "octave to my range", 0, 5.0), -12.0)
        self.assertEqual(_resolve_pitch_shift(female_talk, male_low, "octave to my range", 0, 5.0), 12.0)
        # Keep song key respects explicit pitch_shift only
        self.assertEqual(_resolve_pitch_shift(deep_male, female_lead, "keep song key", -2, 5.0), -2.0)

    def test_lrc_line_windows_preserves_lyric_order_with_fewer_score_sections(self):
        # Cover ABC score has only intro / verse / outro while lyrics have Intro, Verse 1, Chorus, Verse 2, Chorus, Outro
        structure = "0.0\t13.0\tintro\n13.0\t170.0\tverse\n170.0\t190.0\toutro\n"
        lyrics = (
            "[Intro]\nIntro line\n\n"
            "[Verse 1]\nVerse one line\n\n"
            "[Chorus]\nChorus one line\n\n"
            "[Verse 2]\nVerse two line\n\n"
            "[Chorus]\nChorus two line\n\n"
            "[Outro]\nOutro line\n"
        )
        windows = lrc_mod.line_windows(lyrics, seconds=190.0, structure=structure)
        lines = [w[0] for w in windows]
        starts = [w[1] for w in windows]
        self.assertEqual(
            lines,
            ["Intro line", "Verse one line", "Chorus one line", "Verse two line", "Chorus two line", "Outro line"],
        )
        self.assertEqual(starts, sorted(starts))
        self.assertGreaterEqual(starts[-1], 170.0)

    def test_parse_abc_structure_counts_multibar_rests(self):
        # Rap covers from SheetSage2 emit Z4| multi-bar rests; Z4| must count as 4 bars, not 1
        abc = (
            "X:1\nM:2/4\nL:1/16\nQ:1/4=73\n"
            "% intro\nV: Vocal\nZ4|\nZ4|\n"
            "% verse\nV: Vocal\n" + "Z4|\n" * 24 +
            "% outro\nV: Vocal\nZ4|\nZ4|\nZ4|\nZ|\n"
        )
        spans = lrc_mod.parse_abc_structure(abc, duration=162.14)
        self.assertEqual([n for _s, _e, n in spans], ["intro", "verse", "outro"])
        self.assertAlmostEqual(spans[-1][1], 192.3, delta=0.5)

    def test_lrc_needs_order_repair_detects_scrambled_legacy_lrc(self):
        lyrics = "[Verse 1]\nLine A\n\n[Chorus]\nLine B\n\n[Outro]\nLine C\n"
        with tempfile.TemporaryDirectory() as folder:
            good = Path(folder) / "good.lrc"
            good.write_text("[ti:Demo]\n\n[00:01.00]<00:01.00>Line A\n[00:05.00]<00:05.00>Line B\n[00:10.00]<00:10.00>Line C\n", encoding="utf-8")
            bad = Path(folder) / "bad.lrc"
            bad.write_text("[ti:Demo]\n\n[00:01.00]<00:01.00>Line A\n[00:05.00]<00:05.00>Line C\n[00:10.00]<00:10.00>Line B\n", encoding="utf-8")
            trunc = Path(folder) / "trunc.lrc"
            trunc.write_text("[ti:Demo]\n[by:section-timing]\n\n[00:01.00]<00:01.00>Line A\n[00:20.00]<00:20.00>Line B\n[00:40.00]<00:40.00>Line C\n", encoding="utf-8")
            self.assertFalse(service._lrc_needs_order_repair(str(good), lyrics, duration=15.0))
            self.assertTrue(service._lrc_needs_order_repair(str(bad), lyrics, duration=15.0))
            self.assertTrue(service._lrc_needs_order_repair(str(trunc), lyrics, duration=162.0))


class RecloneApiAndServiceTests(unittest.TestCase):
    def test_reclone_api_validation_and_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            mp3 = Path(folder) / "Demo_00001.mp3"
            mp3.write_bytes(b"fake-mp3")
            (Path(folder) / "Demo_00001.json").write_text(json.dumps({"title": "Demo", "duration": 5}))
            with patch.object(library, "OUTPUT_DIR", folder):
                sid = library.song_id(str(mp3))
                # Missing voice -> 400
                with patch.object(web_app.library, "list_voices", return_value=[{"name": "Me"}]):
                    with self.assertRaises(HTTPException) as ctx:
                        web_app.api_reclone_song(sid, web_app.RecloneBody(voice="Nobody"))
                    self.assertEqual(ctx.exception.status_code, 400)

                    # Invalid setting -> 400
                    with self.assertRaises(HTTPException) as ctx2:
                        web_app.api_reclone_song(
                            sid, web_app.RecloneBody(voice="Me", voice_settings={"reverb": 9.0})
                        )
                    self.assertEqual(ctx2.exception.status_code, 400)

                    # Valid request schedules "reclone" job
                    def fake_start(kind, cb):
                        self.assertEqual(kind, "reclone")
                        svc = SimpleNamespace(reclone_song=lambda s, **kw: {"id": s, **kw})
                        return cb(svc, {})

                    with patch.object(web_app, "_start", side_effect=fake_start):
                        res = web_app.api_reclone_song(
                            sid, web_app.RecloneBody(voice="Me", voice_settings={"reverb": 0.22, "keep_backing": True})
                        )
                    self.assertEqual(res["song"]["voice"], "Me")
                    self.assertEqual(res["song"]["voice_settings"]["reverb"], 0.22)

    def test_reclone_service_preserves_ai_and_caches_stems(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            out_dir = Path(folder) / "songs"
            tmp_dir = Path(folder) / "temp"
            voices_dir = Path(folder) / "voices"
            out_dir.mkdir(); tmp_dir.mkdir(); voices_dir.mkdir()
            (voices_dir / "Me.pth").write_bytes(b"ckpt")
            (voices_dir / "Me.json").write_text(json.dumps({"recording_type": "singing"}))

            mp3 = out_dir / "Track_00001.mp3"
            mp3.write_bytes(b"original-ai-mp3")
            (out_dir / "Track_00001.json").write_text(json.dumps({"title": "Track", "duration": 1.0}))

            stack.enter_context(patch.object(library, "OUTPUT_DIR", str(out_dir)))
            stack.enter_context(patch.object(library, "VOICE_DIR", str(voices_dir)))
            stack.enter_context(patch.dict(library.CFG, {"temp_dir": str(tmp_dir)}))
            for fn in ("_begin", "_check", "_make_room", "_park_voice_models", "_flush"):
                stack.enter_context(patch.object(service, fn))

            dummy_audio = {"waveform": torch.zeros(1, 2, 4410), "sample_rate": 44100}
            stack.enter_context(patch.object(service, "_read_clip", return_value=dummy_audio))
            written = []
            def fake_write_mp3(audio, path):
                Path(path).write_bytes(b"mp3-out")
                written.append(os.path.basename(path))

            calls = {"split": 0, "lead": 0, "clone": 0}
            class FakeSplit:
                def split(self, audio, segment_seconds=10):
                    calls["split"] += 1
                    return dummy_audio, dummy_audio
                def split_lead(self, vocals):
                    calls["lead"] += 1
                    return dummy_audio, dummy_audio
            class FakeLoad:
                def load(self, voice_name):
                    return ({"name": voice_name},)
            class FakeClone:
                def clone(self, vocals, trained_voice, my_voice=None, **kw):
                    calls["clone"] += 1
                    return dummy_audio, 0.0
            class FakeMix:
                def mix(self, **kw):
                    return (dummy_audio,)
                def combine_vocals(self, **kw):
                    return (dummy_audio,)

            stack.enter_context(patch.dict(sys.modules, {
                "mp3_save": SimpleNamespace(write_mp3=fake_write_mp3),
                "song_voice.song_voice_node": SimpleNamespace(
                    SongVoiceSplit=FakeSplit, SongVoiceLoad=FakeLoad,
                    SongVoiceClone=FakeClone, SongVoiceMix=FakeMix,
                ),
            }))

            sid = library.song_id(str(mp3))
            # First re-clone: preserves original as .ai.mp3, runs split + lead + clone
            res1 = service.reclone_song(sid, voice="Me", voice_settings={"reverb": 0.2})
            self.assertTrue((out_dir / "Track_00001.ai.mp3").is_file())
            self.assertEqual((out_dir / "Track_00001.ai.mp3").read_bytes(), b"original-ai-mp3")
            self.assertEqual(res1["voice"], "Me")
            self.assertEqual(res1["voice_settings"]["reverb"], 0.2)
            self.assertEqual((calls["split"], calls["lead"], calls["clone"]), (1, 1, 1))

            # Second re-clone on same song: hits stem cache, skips split & lead!
            res2 = service.reclone_song(sid, voice="Me", voice_settings={"reverb": 0.3})
            self.assertEqual(res2["voice_settings"]["reverb"], 0.3)
            self.assertEqual(res2["separator"], "roformer")
            self.assertNotIn("warnings", res2)
            self.assertEqual((calls["split"], calls["lead"], calls["clone"]), (1, 1, 2))

    def test_demucs_fallback_not_cached_and_surfaces_warning(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            out_dir = Path(folder) / "songs"
            tmp_dir = Path(folder) / "temp"
            voices_dir = Path(folder) / "voices"
            out_dir.mkdir(); tmp_dir.mkdir(); voices_dir.mkdir()
            (voices_dir / "Me.pth").write_bytes(b"ckpt")
            (voices_dir / "Me.json").write_text(json.dumps({"recording_type": "singing"}))

            mp3 = out_dir / "Track_00001.mp3"
            mp3.write_bytes(b"original-ai-mp3")
            (out_dir / "Track_00001.json").write_text(json.dumps({"title": "Track", "duration": 1.0}))

            stack.enter_context(patch.object(library, "OUTPUT_DIR", str(out_dir)))
            stack.enter_context(patch.object(library, "VOICE_DIR", str(voices_dir)))
            stack.enter_context(patch.dict(library.CFG, {"temp_dir": str(tmp_dir)}))
            for fn in ("_begin", "_check", "_make_room", "_park_voice_models", "_flush"):
                stack.enter_context(patch.object(service, fn))

            dummy_audio = {"waveform": torch.zeros(1, 2, 4410), "sample_rate": 44100}
            stack.enter_context(patch.object(service, "_read_clip", return_value=dummy_audio))
            def fake_write_mp3(audio, path):
                Path(path).write_bytes(b"mp3-out")

            calls = {"split": 0, "lead": 0}
            mode = {"sep": "demucs", "lead": "none"}
            class FlakySplit:
                def split(self, audio, segment_seconds=10):
                    calls["split"] += 1
                    self.last_separator = mode["sep"]
                    return dummy_audio, dummy_audio
                def split_lead(self, vocals):
                    calls["lead"] += 1
                    self.last_lead_separator = mode["lead"]
                    return (dummy_audio, dummy_audio if mode["lead"] == "karaoke" else None)
            class FakeLoad:
                def load(self, voice_name):
                    return ({"name": voice_name},)
            class FakeClone:
                def clone(self, vocals, trained_voice, my_voice=None, **kw):
                    return dummy_audio, 0.0
            class FakeMix:
                def mix(self, **kw):
                    return (dummy_audio,)
                def combine_vocals(self, **kw):
                    return (dummy_audio,)

            stack.enter_context(patch.dict(sys.modules, {
                "mp3_save": SimpleNamespace(write_mp3=fake_write_mp3),
                "song_voice.song_voice_node": SimpleNamespace(
                    SongVoiceSplit=FlakySplit, SongVoiceLoad=FakeLoad,
                    SongVoiceClone=FakeClone, SongVoiceMix=FakeMix,
                ),
            }))

            sid = library.song_id(str(mp3))
            # Run 1: Demucs + no karaoke fallback -> warns user AND does NOT populate RoFormer cache
            res1 = service.reclone_song(sid, voice="Me")
            self.assertEqual(res1["separator"], "demucs")
            self.assertEqual(len(res1.get("warnings", [])), 2)
            self.assertEqual((calls["split"], calls["lead"]), (1, 1))

            # Run 2: RoFormer recovers -> MUST re-run separation (not serve stale Demucs stems!)
            mode["sep"], mode["lead"] = "roformer", "karaoke"
            res2 = service.reclone_song(sid, voice="Me")
            self.assertEqual(res2["separator"], "roformer")
            self.assertNotIn("warnings", res2)
            self.assertEqual((calls["split"], calls["lead"]), (2, 2))

            # Run 3: Now cached RoFormer + Karaoke stems ARE reused
            res3 = service.reclone_song(sid, voice="Me")
            self.assertEqual((calls["split"], calls["lead"]), (2, 2))
            self.assertNotIn("warnings", res3)

    def test_corrupt_cached_file_removed_and_redownloaded_with_hash_check(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            good_bytes = b"valid-model-content-" * 20
            good_sha = hashlib.sha256(good_bytes).hexdigest()
            local = os.path.join(folder, "model.ckpt")
            Path(local).write_bytes(b"corrupt-truncated-file-" * 10)

            downloads = []
            def fake_hf(repo, filename, local_dir, force_download=False):
                downloads.append((repo, filename, force_download))
                out = os.path.join(local_dir, "model.ckpt")
                Path(out).write_bytes(good_bytes)
                return out

            with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf):
                roformer._ensure_file("org/repo", "model.ckpt", local, folder, [], expected_sha256=good_sha)
            self.assertEqual(len(downloads), 1)
            self.assertTrue(downloads[0][2])  # force_download=True after corrupt file removed
            self.assertEqual(Path(local).read_bytes(), good_bytes)

            # Second call with valid file does NOT re-download
            with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf):
                roformer._ensure_file("org/repo", "model.ckpt", local, folder, [], expected_sha256=good_sha)
            self.assertEqual(len(downloads), 1)

    def test_tampered_download_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            local = os.path.join(folder, "model.ckpt")
            def fake_hf(repo, filename, local_dir, force_download=False):
                out = os.path.join(local_dir, "model.ckpt")
                Path(out).write_bytes(b"tampered-bytes-" * 20)
                return out
            with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf):
                with self.assertRaises(RuntimeError):
                    roformer._ensure_file("org/repo", "model.ckpt", local, folder, [], expected_sha256="0" * 64)
            self.assertFalse(os.path.exists(local))


if __name__ == "__main__":
    unittest.main()
