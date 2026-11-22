"""Exercise the real Pinokio-launched pipeline; writes one clearly named test song."""
import argparse
import json
import time
import urllib.request
from pathlib import Path
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument('--base', required=True)
parser.add_argument('--voice', default='my_voice')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
BASE = args.base.rstrip('/')

def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(BASE + path, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)

def run(path, body):
    started = time.monotonic()
    jid = call(path, body)['job']
    last = None
    while time.monotonic() - started < 900:
        job = call('/api/job')['job']
        assert job['id'] == jid, job
        label = (job.get('progress') or {}).get('label')
        if label and label != last:
            print(label, flush=True)
            last = label
        if job['state'] != 'running':
            assert job['state'] == 'done', job
            print(f'{path}: done in {time.monotonic()-started:.1f}s', flush=True)
            return job['result']
        time.sleep(2)
    raise TimeoutError(path)

for _ in range(90):
    state = call('/api/state')
    if state['engine']['state'] != 'starting':
        break
    time.sleep(2)
assert state['engine']['state'] == 'ready', state['engine']
assert not [m for m in state['models'] if m['required'] and not m['ok']]
assert args.voice in [v['name'] for v in state['voices']], 'Expected your existing authorized voice'
lyrics = run('/api/lyrics', {'idea':'a cheerful song about coffee and making music on my computer',
                            'structure':['verse','chorus'], 'lines':2})['lyrics']
assert '[verse]' in lyrics and '[chorus]' in lyrics, lyrics
song = run('/api/song', {'lyrics':lyrics,'style':state['presets'][0]['style'], 'voice':args.voice,
                         'title':'Pinokio Packaging Test', 'idea':'Pinokio packaging verification',
                         'seed':271828, 'memory':'auto', 'model':'auto'})['song']
assert song['has_lrc'] and song['has_ai'] and song['voice'] == args.voice, song
config = json.loads((ROOT / 'app/config.json').read_text(encoding='utf-8'))
path = Path(config['output_dir']) / song['file']
wave, rate = sf.read(path, always_2d=True)
duration = len(wave) / rate
assert 5 < duration < 240, duration
assert abs(duration-song['duration']) < 2, (duration,song['duration'])
assert wave.shape[1] in (1,2) and abs(wave).max() > .001
assert path.with_suffix('.lrc').is_file()
assert path.with_suffix('.ai.mp3').is_file()
# Guard the global-transformers-patch ordering regression in this fresh server.
lyrics2 = run('/api/lyrics', {'idea':'a little thank you for a song that finally works',
                             'structure':['verse'], 'lines':2})['lyrics']
assert '[verse]' in lyrics2 and len(lyrics2) > 15
report = {'status':'PASS', 'base':BASE, 'song':song, 'decoded':{'duration':duration,'rate':rate,'channels':wave.shape[1]},
          'checks':['isolated CUDA engine','lyrics generation','song generation','existing voice conversion','AI original preserved','timed lyrics','MP3 decode and duration','song-to-lyrics model handoff']}
(ROOT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
