"""Real GPU training smoke test on a uniquely named temporary voice; never replaces a real voice."""
import argparse
import json
import time
import uuid
import urllib.request
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--base',required=True)
args=parser.parse_args()
BASE=args.base.rstrip('/')
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'app/config.json').read_text(encoding='utf-8'))
voices=Path(cfg['models_dir'])/'seed_vc/voices'
name='zz_recording_type_test_'+uuid.uuid4().hex[:10]
assert not any((voices/(name+ext)).exists() for ext in ('.pth','.json','.wav'))
reference=voices/'my_voice.wav'
assert reference.is_file()

def call(path,body=None,method=None,headers=None):
    req=urllib.request.Request(BASE+path,data=body,method=method,headers=headers or {})
    with urllib.request.urlopen(req,timeout=30) as response:
        return json.load(response)

existing=[v['name'] for v in call('/api/voices')['voices']]
last=None
terminal=False
try:
    result=call('/api/voices?name='+name+'&recording_type=rapping',reference.read_bytes(),'POST',{'X-Filename':'reference.wav','Content-Type':'application/octet-stream'})
    jid=result['job']
    started=time.monotonic()
    while time.monotonic()-started<900:
        job=call('/api/job')['job']
        assert job['id']==jid,job
        label=(job.get('progress') or {}).get('label')
        if label and label!=last:
            print(label,flush=True);last=label
        if job['state']!='running':
            terminal=True
            assert job['state']=='done',job
            break
        time.sleep(2)
    else:
        raise TimeoutError('Training still running; temporary files are intentionally not removed mid-job.')
    saved=next(v for v in call('/api/voices')['voices'] if v['name']==name)
    meta=json.loads((voices/(name+'.json')).read_text(encoding='utf-8'))
    assert saved['recording_type']=='rapping' and meta['recording_type']=='rapping'
    assert meta['clip'] and meta['steps']==300 and meta['seconds']>=10
    with urllib.request.urlopen(BASE+'/api/voices/'+name+'/preview',timeout=30) as response:
        assert response.read(4)==b'RIFF'
    assert job['result']['voice']['recording_type']=='rapping'
    print(json.dumps({'status':'PASS','voice':name,'selected_type':saved['recording_type'],'training_metadata':{'steps':meta['steps'],'seconds':meta['seconds']},'note':'Real training/persistence test; the owned talking reference was intentionally labelled rapping to test plumbing, not rap quality.'}),flush=True)
finally:
    if terminal:
        if any(v['name']==name for v in call('/api/voices')['voices']):
            call('/api/voices/'+name,method='DELETE')
        for ext in ('.pth','.json','.wav'):
            path=voices/(name+ext)
            if path.exists():
                raise RuntimeError('Temporary voice cleanup did not complete: '+str(path))
        assert all(v in [entry['name'] for entry in call('/api/voices')['voices']] for v in existing)
        for clip in (Path(cfg['input_dir'])/'voice_clips').glob(name+'_*.wav'):
            clip.unlink()
        print('Temporary test voice removed; existing voices retained.',flush=True)
