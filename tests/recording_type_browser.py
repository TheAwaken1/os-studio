"""Real browser UI checks. Training POSTs are intentionally BLOCKED, not simulated as successes."""
import argparse
import io
import json
import math
import struct
import wave
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright
parser=argparse.ArgumentParser()
parser.add_argument('--base',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
with io.BytesIO() as buffer:
    with wave.open(buffer,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
        wav.writeframes(b''.join(struct.pack('<h',int(3000*math.sin(i*2*math.pi*220/16000))) for i in range(32000)))
    fixture=buffer.getvalue()
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
    page=browser.new_page(viewport={'width':1280,'height':960})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.add_init_script("try{localStorage.setItem('yss.guide','off')}catch(e){}")  # every step open: this test is about the controls, not the guide
    page.goto(args.base)
    page.wait_for_selector('#voiceChips .add')
    current_job=page.request.get(args.base+'/api/job').json()['job']
    assert not current_job or current_job['state']!='running', 'Run UI request-capture checks when no GPU job is active.'
    real_voices=page.request.get(args.base+'/api/voices').json()['voices']
    assert all(v.get('recording_type') in ('talking','singing','rapping') for v in real_voices)
    captured=[]
    def block_training(route):
        captured.append(parse_qs(urlparse(route.request.url).query))
        route.abort('blockedbyclient')
    page.route('**/api/voices?*',block_training)
    for kind in ('talking','singing','rapping'):
        page.locator('#voiceChips .add').click()
        assert page.locator('#voiceRecordingType').input_value()=='talking'
        options=page.locator('#voiceRecordingType option').evaluate_all('(nodes)=>nodes.map(n=>n.value)')
        assert options==['talking','singing','rapping'],options
        page.locator('#voiceRecordingType').select_option(kind)
        before_hint=page.locator('#recHint').inner_text().lower()
        if kind!='talking':
            assert {'singing':'sing','rapping':'rap'}[kind] in before_hint,before_hint
        page.locator('#clip').set_input_files({'name':'unit-test-tone.wav','mimeType':'audio/wav','buffer':fixture})
        page.locator('#voiceName').fill('Blocked UI Test')
        with page.expect_request('**/api/voices?*'):
            page.locator('#voiceTrain').click()
        page.wait_for_timeout(250)
        assert captured[-1]['recording_type']==[kind],captured[-1]
        assert not page.locator('#voiceProgress').is_visible()
        page.locator('#voiceCancel').click()
    page.locator('#voiceChips .add').click()
    page.set_viewport_size({'width':390,'height':844})
    assert page.locator('#voiceRecordingType').is_visible()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path=str(root/'logs/voice-recording-types.png'))
    assert not errors,errors
    browser.close()
    print(json.dumps({'status':'PASS','checked_types':['talking','singing','rapping'],'blocked_training_requests':captured,'existing_voices_have_type':True,'page_errors':errors,'note':'Wire-format test only; POSTs intentionally aborted before reaching the real server.'}))
