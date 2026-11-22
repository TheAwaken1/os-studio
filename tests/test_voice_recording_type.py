"""Recording-type persistence and legacy defaults. All voice files are temporary fixtures."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

APP = Path(sys.argv[1]).resolve() if __name__ == "__main__" and len(sys.argv) > 1 and Path(sys.argv[1]).is_dir() else Path(__file__).resolve().parents[1] / 'app'
if __name__ == "__main__" and len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    del sys.argv[1]
sys.path.insert(0, str(APP))
import library

class RecordingTypeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.override = patch.object(library, 'VOICE_DIR', str(self.root))
        self.override.start()
        (self.root / 'Example.pth').write_bytes(b'test checkpoint, not a real model')
        (self.root / 'Example.json').write_text(json.dumps({'clip':'fingerprint','steps':300,'seconds':44.2}))
    def tearDown(self):
        self.override.stop()
        self.temp.cleanup()
    def test_legacy_voice_defaults_to_talking(self):
        self.assertEqual(library.get_voice_recording_type('Example'), 'talking')
        self.assertEqual(library.list_voices()[0]['recording_type'], 'talking')
        self.assertEqual(library.voice_clip_mode('Example'), 'talking')
    def test_each_type_persists_without_losing_training_metadata(self):
        for kind in ('talking','singing','rapping'):
            library.set_voice_recording_type('Example', kind)
            meta = json.loads((self.root / 'Example.json').read_text())
            self.assertEqual(meta, {'clip':'fingerprint','steps':300,'seconds':44.2,'recording_type':kind})
            self.assertEqual(library.list_voices()[0]['recording_type'], kind)
            self.assertEqual(library.get_voice_recording_type('Example'), kind)
            self.assertEqual(library.voice_clip_mode('Example'), 'talking' if kind == 'talking' else 'singing')
    def test_invalid_types_and_traversal_do_not_write(self):
        before = (self.root / 'Example.json').read_bytes()
        for kind in ('unknown','Singing','','mixed',None):
            with self.assertRaises(ValueError):
                library.set_voice_recording_type('Example', kind)
        for name in ('../Example','..\\Example','unknown',''):
            with self.assertRaises(FileNotFoundError):
                library.set_voice_recording_type(name, 'singing')
        self.assertEqual((self.root / 'Example.json').read_bytes(), before)
    def test_invalid_legacy_metadata_falls_back_safely(self):
        for value in ('not json','[]','{"recording_type":"unknown"}'):
            (self.root / 'Example.json').write_text(value)
            self.assertEqual(library.get_voice_recording_type('Example'), 'talking')
            self.assertEqual(library.list_voices()[0]['recording_type'], 'talking')

if __name__ == '__main__':
    unittest.main()
