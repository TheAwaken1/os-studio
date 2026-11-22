"""API boundary tests; GPU job scheduling is stubbed and clips stay in a temporary directory."""
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
APP = Path(sys.argv[1]).resolve() if __name__ == "__main__" and len(sys.argv)>1 and Path(sys.argv[1]).is_dir() else Path(__file__).resolve().parents[1]/'app'
if __name__ == "__main__" and len(sys.argv)>1 and Path(sys.argv[1]).is_dir():
    del sys.argv[1]
sys.path.insert(0,str(APP))
import app
from fastapi import HTTPException

class RecordingApiTests(unittest.TestCase):
    def test_all_types_reach_training_job(self):
        for kind in ('talking','singing','rapping'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                request = SimpleNamespace(body=AsyncMock(return_value=b'unit-test-upload' * 2000), headers={'x-filename':'fixture.wav'})
                def start(job_kind, callback):
                    self.assertEqual(job_kind,'voice')
                    service = SimpleNamespace(train_voice=lambda *args, **kwargs: kwargs)
                    return callback(service,{})
                with patch.object(app,'CLIP_DIR',folder), patch.object(app.library,'list_voices',return_value=[]), patch.object(app,'_start',side_effect=start):
                    result = asyncio.run(app.api_train_voice(request,name='Fixture',recording_type=kind))
                self.assertEqual(result['voice']['recording_type'],kind)
                self.assertFalse(result['voice']['retrain'])
    def test_invalid_type_rejected_before_upload(self):
        request=SimpleNamespace(body=AsyncMock(),headers={})
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(app.api_train_voice(request,name='Fixture',recording_type='opera'))
        self.assertEqual(raised.exception.status_code,400)
        request.body.assert_not_awaited()
    def test_old_clients_default_to_talking(self):
        with tempfile.TemporaryDirectory() as folder:
            request=SimpleNamespace(body=AsyncMock(return_value=b'unit-test-upload'*2000),headers={})
            def start(kind, callback):
                return callback(SimpleNamespace(train_voice=lambda *args,**kwargs:kwargs),{})
            with patch.object(app,'CLIP_DIR',folder), patch.object(app.library,'list_voices',return_value=[]), patch.object(app,'_start',side_effect=start):
                result=asyncio.run(app.api_train_voice(request,name='Fixture'))
            self.assertEqual(result['voice']['recording_type'],'talking')

if __name__=='__main__':
    unittest.main()
