"""Service integration with a stub GPU trainer; validates real metadata/settings code."""
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
APP=Path(sys.argv[1]).resolve() if __name__ == "__main__" and len(sys.argv)>1 and Path(sys.argv[1]).is_dir() else Path(__file__).resolve().parents[1]/'app'
if __name__ == "__main__" and len(sys.argv)>1 and Path(sys.argv[1]).is_dir():
    del sys.argv[1]
sys.path.insert(0,str(APP))
sys.path.insert(0,str(APP/'shims'))
import service
import library

class ServiceRecordingTests(unittest.TestCase):
    def test_training_persists_each_selected_type(self):
        for kind in ('talking','singing','rapping'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root=Path(folder)
                stack.enter_context(patch.object(library,'VOICE_DIR',folder))
                for name in ('_begin','_check','_make_room','_park_voice_models','_flush'):
                    stack.enter_context(patch.object(service,name))
                stack.enter_context(patch.object(service,'_read_clip',return_value=object()))
                stack.enter_context(patch.object(service,'_seconds',return_value=40))
                class Trainer:
                    def train(self, **kwargs):
                        (root/'Fixture.pth').write_bytes(b'unit test checkpoint')
                        (root/'Fixture.json').write_text(json.dumps({'clip':'fingerprint','steps':300,'seconds':40}))
                        return ({'name':'Fixture'},)
                stack.enter_context(patch.dict(sys.modules, {'song_voice.song_voice_node':SimpleNamespace(SongVoiceTrain=Trainer)}))
                result=service.train_voice('not-a-real-clip.wav','Fixture',recording_type=kind)
                self.assertEqual(result['recording_type'],kind)
                self.assertEqual(library.get_voice_recording_type('Fixture'),kind)
                self.assertEqual(json.loads((root/'Fixture.json').read_text())['clip'],'fingerprint')
    def test_conversion_options_follow_metadata_without_mutating_defaults(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(library,'VOICE_DIR',folder):
            root=Path(folder)
            (root/'Fixture.pth').write_bytes(b'unit test checkpoint')
            for kind,mode in [('singing','singing'),('talking','talking'),('rapping','singing'),('talking','talking')]:
                library.set_voice_recording_type('Fixture',kind)
                settings=service.voice_clone_settings('Fixture')
                self.assertEqual(settings['my_clip_is'],mode)
                self.assertEqual(settings['pitch_mode'],'octave to my range')
                self.assertEqual(service.VOICE_CLONE['my_clip_is'],'talking')
                settings['cfg_rate']=999
                self.assertEqual(service.VOICE_CLONE['cfg_rate'],0.7)

if __name__=='__main__':
    unittest.main()
