"""Measured player visuals: synthetic audio in, known picture out. No model, no GPU."""
import base64
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
APPS = [ROOT / 'app']


def load_library(app, songs_dir):
    """Import one copy of library.py against a throwaway songs folder."""
    config = json.loads((app / 'config.json').read_text(encoding='utf-8'))
    config.update(output_dir=str(songs_dir), temp_dir=str(Path(songs_dir).parent / 'temp'))
    spec = importlib.util.spec_from_file_location('library_under_test', app / 'library.py')
    module = importlib.util.module_from_spec(spec)
    real_open = open

    def fake_open(path, *args, **kwargs):
        if os.path.abspath(str(path)) == os.path.abspath(str(app / 'config.json')):
            import io
            return io.StringIO(json.dumps(config))
        return real_open(path, *args, **kwargs)

    import builtins
    builtins.open = fake_open
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.open = real_open
    return module


def bands_of(result):
    raw = np.frombuffer(base64.b64decode(result['spectrum']), dtype=np.uint8)
    return raw.reshape(-1, result['bands'])


class SongVizTests(unittest.TestCase):
    def test_outline_and_spectrum_follow_the_audio(self):
        rate = 24000
        t = np.arange(rate * 4) / rate
        # 1 s quiet low tone, 1 s silence, 2 s loud high tone
        audio = np.concatenate([
            0.1 * np.sin(2 * np.pi * 110 * t[:rate]),
            np.zeros(rate),
            0.8 * np.sin(2 * np.pi * 3000 * t[:rate * 2]),
        ]).astype(np.float32)
        for app in APPS:
            with self.subTest(app=str(app)), tempfile.TemporaryDirectory() as tmp:
                library = load_library(app, Path(tmp) / 'songs')
                result = library.analyze_audio(audio, rate)
                self.assertEqual(result['fps'], library.VIZ_FPS)
                self.assertEqual(len(result['peaks']), library.VIZ_PEAKS)
                self.assertAlmostEqual(result['duration'], 4.0, places=2)
                self.assertEqual(result['frames'], 4 * library.VIZ_FPS)
                peaks = np.array(result['peaks'])
                quarter = len(peaks) // 4
                self.assertLess(peaks[quarter + 5:2 * quarter - 5].max(), 5, 'silence draws flat')
                self.assertGreater(peaks[2 * quarter + 5:].min(), 240, 'the loud part draws tall')
                self.assertLess(peaks[5:quarter - 5].max(), peaks[2 * quarter + 5:].min())
                spectrum = bands_of(result)
                self.assertEqual(spectrum.shape, (result['frames'], library.VIZ_BANDS))
                fps = result['fps']
                low, quiet, high = spectrum[5:fps - 5], spectrum[fps + 5:2 * fps - 5], spectrum[2 * fps + 5:4 * fps - 5]
                self.assertLess(int(quiet.max()), 10)
                self.assertLess(low.mean(axis=0).argmax(), library.VIZ_BANDS // 3, 'a low note lights the low bars')
                self.assertGreater(high.mean(axis=0).argmax(), library.VIZ_BANDS // 2, 'a high note lights the high bars')
                self.assertGreater(int(high.max()), 200)

    def test_silence_and_empty_input_are_safe(self):
        for app in APPS:
            with self.subTest(app=str(app)), tempfile.TemporaryDirectory() as tmp:
                library = load_library(app, Path(tmp) / 'songs')
                silent = library.analyze_audio(np.zeros(24000, dtype=np.float32), 24000)
                self.assertEqual(max(silent['peaks']), 0)
                self.assertEqual(int(bands_of(silent).max()), 0)
                empty = library.analyze_audio(np.zeros(0, dtype=np.float32), 24000)
                self.assertEqual(empty['frames'], 0)
                json.dumps(silent), json.dumps(empty)

    def test_song_file_is_measured_once_then_served_from_cache(self):
        rate = 48000
        t = np.arange(rate * 3) / rate
        stereo = np.stack([0.5 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
        for app in APPS:
            with self.subTest(app=str(app)), tempfile.TemporaryDirectory() as tmp:
                songs = Path(tmp) / 'songs'
                songs.mkdir()
                library = load_library(app, songs)
                self.assertTrue(Path(library.VIZ_DIR).is_relative_to(tmp), 'the cache must stay out of the real app folders')
                song = songs / 'Tone_00001.mp3'
                sf.write(str(song), stereo, rate, format='MP3')
                first = library.song_viz(str(song))
                self.assertAlmostEqual(first['duration'], 3.0, delta=0.15)
                self.assertGreater(max(first['peaks']), 200)
                cached = list(Path(library.VIZ_DIR).glob('*.json'))
                self.assertEqual(len(cached), 1)
                self.assertEqual(sorted(p.name for p in songs.iterdir()), ['Tone_00001.mp3'], 'nothing is written beside the songs')
                real_read = sf.read
                sf.read = lambda *a, **k: self.fail('a cached song must not be decoded again')
                try:
                    self.assertEqual(library.song_viz(str(song)), first)
                finally:
                    sf.read = real_read
                # a changed file is measured again
                sf.write(str(song), stereo[: rate * 2], rate, format='MP3')
                again = library.song_viz(str(song))
                self.assertAlmostEqual(again['duration'], 2.0, delta=0.15)


if __name__ == '__main__':
    sys.exit(unittest.main())
