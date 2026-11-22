"""Voice recording preview lookup without touching shared voices."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library


class VoicePreviewTests(unittest.TestCase):
    def test_existing_recording_is_advertised_and_resolved(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(library, 'VOICE_DIR', folder):
            root = Path(folder)
            (root / 'Example.pth').write_bytes(b'fixture checkpoint')
            (root / 'Example.wav').write_bytes(b'fixture recording')
            self.assertTrue(library.list_voices()[0].get('has_preview'))
            self.assertEqual(library.voice_preview_path('Example'), str(root / 'Example.wav'))

    def test_missing_recording_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(library, 'VOICE_DIR', folder):
            (Path(folder) / 'Example.pth').write_bytes(b'fixture checkpoint')
            self.assertFalse(library.list_voices()[0].get('has_preview'))
            for name in ('Example', '../Example', '..\\Example', 'unknown', ''):
                with self.assertRaises(FileNotFoundError):
                    library.voice_preview_path(name)


if __name__ == '__main__':
    unittest.main()
