"""Guided steps, Remix and the measured player visuals in a real browser. No GPU job is started:
the lyrics and song POSTs are aborted before they reach the server."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--base', required=True)
args = parser.parse_args()
base = args.base.rstrip('/')
root = Path(__file__).resolve().parents[1]
(root / 'logs').mkdir(exist_ok=True)
EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
LYRICS = '[verse]\nBrowser test line one\nBrowser test line two\n\n[chorus]\nBrowser test hook'
# Counts Web Audio use and freezes animation frames, as an embedded or background view can.
# (Every wait below polls on a timer for the same reason.)
SPY = """window.__contexts=0;
for (const name of ['AudioContext','webkitAudioContext']) if (window[name]) {
  const Original=window[name];
  window[name]=new Proxy(Original,{construct(t,a){window.__contexts++;return Reflect.construct(t,a)}});
}
window.requestAnimationFrame=()=>0;"""
INK = """(id)=>{const c=document.getElementById(id);const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
  let lit=0,sum=0;for(let i=3;i<d.length;i+=4){if(d[i]>0)lit++;sum=(sum+d[i]*(i%9973))%1000000007}
  return {lit,sum,mode:c.dataset.mode,width:c.width,height:c.height}}"""


def states(page):
    return page.locator('.step').evaluate_all('(n)=>n.map(x=>x.dataset.state)')


def open_bodies(page):
    return page.locator('.step-body').evaluate_all('(n)=>n.map(x=>x.offsetParent!==null)')


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=EDGE, headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    errors, blocked = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.add_init_script(SPY)
    page.route('**/api/lyrics', lambda route: (blocked.append(('lyrics', route.request.post_data_json)), route.abort()))
    page.route('**/api/song', lambda route: (blocked.append(('song', route.request.post_data_json)), route.abort()))
    songs = page.request.get(base + '/api/songs').json()['songs']
    remixable = [s for s in songs if s['lyrics']]
    assert len(songs) >= 2 and remixable, 'Need saved songs (one with lyrics) for the player and Remix checks.'

    # ---- first visit: only step 1 is open, the rest are greyed out and closed
    page.goto(base)
    page.wait_for_selector('#idea')
    page.wait_for_function("document.querySelector('.step').dataset.state==='current'", polling=100)
    assert page.locator('body').get_attribute('data-mode') == 'create'
    assert states(page) == ['current', 'locked', 'locked', 'locked', 'locked'], states(page)
    assert open_bodies(page) == [True, False, False, False, False]
    assert not page.locator('#listeningStage').is_visible(), 'Create shows the steps only.'
    assert 'Opens after step 1' in page.locator('#sumLyrics').inner_text()
    page.screenshot(path=str(root / 'logs/guided-step1.png'))
    # a locked step cannot be opened by clicking it
    page.locator('#stepStyle .step-head').click()
    assert states(page)[2] == 'locked'

    # ---- Write lyrics needs an idea; with one it sends the request (aborted here)
    page.locator('#btnLyrics').click()
    page.wait_for_selector('.toast.error')
    assert not blocked and states(page)[0] == 'current'
    page.locator('#idea').fill('a browser test, do not generate')
    page.locator('#btnLyrics').click()
    page.wait_for_timeout(300)
    assert blocked and blocked[0][0] == 'lyrics' and blocked[0][1]['idea'] == 'a browser test, do not generate'
    assert states(page)[0] == 'current', 'A failed start must not move on.'

    # ---- own lyrics path, then Next through every step
    page.locator('#btnOwnLyrics').click()
    assert states(page) == ['done', 'current', 'locked', 'locked', 'locked'], states(page)
    page.locator('#lyricsCard [data-guide="next"]').click()            # nothing typed yet
    assert states(page)[1] == 'current'
    page.locator('#lyrics').fill(LYRICS)
    page.locator('#lyricsCard [data-guide="next"]').click()
    assert states(page) == ['done', 'done', 'current', 'locked', 'locked'], states(page)
    assert '3 lines' in page.locator('#sumLyrics').inner_text()
    page.locator('#styleChips .chip').nth(1).click()
    page.locator('#stepStyle [data-guide="next"]').click()
    assert states(page)[3] == 'current'
    assert page.locator('#voiceChips .chip').first.is_visible()
    page.locator('#stepVoice [data-guide="next"]').click()
    assert states(page) == ['done', 'done', 'done', 'done', 'current'], states(page)
    assert page.locator('#btnMake').is_visible()
    assert '3 lines' in page.locator('#recapLyrics').inner_text()
    assert page.locator('#recapStyle').inner_text() == 'Pop'
    page.screenshot(path=str(root / 'logs/guided-step5.png'))
    page.locator('#btnMake').click()
    page.wait_for_timeout(300)
    song_posts = [body for kind, body in blocked if kind == 'song']
    assert len(song_posts) == 1 and song_posts[0]['lyrics'] == LYRICS and 'pop' in song_posts[0]['style'].lower()

    # ---- going back keeps later steps open; progress survives a reload
    page.locator('#lyricsCard .step-head').click()
    assert states(page) == ['done', 'current', 'ahead', 'ahead', 'ahead'], states(page)
    # ---- every start is fresh: step 1, nothing typed; the unfinished song is offered back, not forced
    page.reload()
    page.wait_for_function("document.querySelector('.step').dataset.state==='current'", polling=100)
    assert states(page) == ['current', 'locked', 'locked', 'locked', 'locked'], states(page)
    assert page.locator('#idea').input_value() == '' and page.locator('#lyrics').input_value() == ''
    assert page.locator('#resumeBar').is_visible()
    assert 'a browser test, do not generate' in page.locator('#resumeText').inner_text()
    page.reload()                                                       # still offered after another fresh start
    page.wait_for_function("document.querySelector('.step').dataset.state==='current'", polling=100)
    assert page.locator('#resumeBar').is_visible() and page.locator('#lyrics').input_value() == ''
    with page.expect_navigation():
        page.locator('#resumeGo').click()
    page.wait_for_function("document.querySelector('#lyricsCard').dataset.state==='current'", polling=100)
    assert page.locator('#lyrics').input_value() == LYRICS
    assert not page.locator('#resumeBar').is_visible()
    # emptying the lyrics locks everything after step 2 again
    page.locator('#lyrics').fill('')
    assert states(page) == ['done', 'current', 'locked', 'locked', 'locked'], states(page)
    page.locator('#lyrics').fill(LYRICS)

    # ---- the switch shows every step at once, and is remembered
    page.locator('#guideToggle').click()
    assert page.locator('#composer').get_attribute('data-guided') == 'false'
    assert open_bodies(page) == [True] * 5
    page.reload()
    page.wait_for_selector('#shapePresets button')
    assert open_bodies(page) == [True] * 5
    assert page.locator('#lyrics').input_value() == ''
    page.locator('#resumeDrop').click()                                 # Dismiss forgets the old draft for good
    assert not page.locator('#resumeBar').is_visible()
    page.reload()
    page.wait_for_selector('#shapePresets button')
    assert not page.locator('#resumeBar').is_visible() and page.locator('#idea').input_value() == ''
    page.locator('#guideToggle').click()
    assert page.locator('#composer').get_attribute('data-guided') == 'true'

    # ---- player: bars and waveform come from measured song data, without Web Audio or animation frames
    viz = page.request.get(base + '/api/songs/' + songs[0]['id'] + '/viz')
    assert viz.ok, viz.text()
    data = viz.json()
    assert data['frames'] > 0 and len(data['peaks']) == 240 and max(data['peaks']) > 100
    assert abs(data['frames'] / data['fps'] - data['duration']) < 0.2
    page.locator('#modeListen').click()
    assert not page.locator('#composer').is_visible()
    page.locator('#btnPlayAll').click()
    page.wait_for_function("document.querySelector('#audio').currentTime>1 && !document.querySelector('#audio').paused", timeout=15000, polling=100)
    page.wait_for_function("document.querySelector('#viz').dataset.mode==='measured'", timeout=15000, polling=100)
    page.wait_for_timeout(1500)                                         # past a silent intro
    samples = []
    for _ in range(8):
        samples.append(page.evaluate(INK, 'viz'))
        page.wait_for_timeout(180)
    assert len({s['sum'] for s in samples}) >= 3, 'Live bars must keep changing while a song plays.'
    assert max(s['lit'] for s in samples) > samples[0]['width'] * 6
    wave = page.evaluate(INK, 'wave')
    assert wave['mode'] == 'measured' and wave['lit'] > wave['width'] * 4, wave
    assert page.evaluate('__contexts') == 0, 'The player must not depend on Web Audio.'
    assert page.locator('#npState').inner_text().lower() == 'now playing'
    assert page.locator('#npCoverText').inner_text().strip()
    page.screenshot(path=str(root / 'logs/player-desktop.png'))
    # tap the middle of the waveform: playback jumps there
    box = page.locator('#wave').bounding_box()
    page.mouse.click(box['x'] + box['width'] * 0.5, box['y'] + box['height'] / 2)
    page.wait_for_function("Math.abs(document.querySelector('#audio').currentTime/document.querySelector('#audio').duration-0.5)<0.04", timeout=5000, polling=100)
    # paused: the bars settle to rest and stop moving
    page.locator('#btnPlay').click()
    page.wait_for_timeout(1500)
    rest = [page.evaluate(INK, 'viz') for _ in range(2)]
    assert rest[0]['sum'] == rest[1]['sum']
    assert page.locator('#npState').inner_text().lower() == 'paused'

    # ---- Remix asks what to change, then opens that step in Create
    title = page.locator('#npTitle').inner_text()
    page.locator('#btnReuse').click()
    assert page.locator('#remixDlg').is_visible()
    assert title in page.locator('#remixHint').inner_text()
    assert page.locator('.remix-choice').count() == 4
    page.screenshot(path=str(root / 'logs/remix-dialog.png'))
    page.locator('.remix-choice[data-remix="style"]').click()
    assert page.locator('body').get_attribute('data-mode') == 'create'
    assert page.locator('#stepStyle').get_attribute('data-state') == 'current'
    assert title in page.locator('#remixText').inner_text() and page.locator('#remixBanner').is_visible()
    assert page.locator('#guideTitle').inner_text() == 'Remix a song'
    current = next(s for s in songs if s['title'] == title)
    assert page.locator('#lyrics').input_value().strip() == current['lyrics'].strip()
    assert page.locator('#seed').input_value() == ''
    assert page.locator('#miniPlayer').is_visible()
    page.screenshot(path=str(root / 'logs/remix-create.png'))
    # from the song list, New take lands on Make the song
    page.locator('#modeListen').click()
    page.locator('#songs .song .remix').first.click()
    page.locator('.remix-choice[data-remix="take"]').click()
    assert page.locator('#stepMake').get_attribute('data-state') == 'current'
    assert page.locator('#btnMake').is_visible()
    # leaving the remix starts clean at step 1
    page.locator('#remixStop').click()
    assert states(page) == ['current', 'locked', 'locked', 'locked', 'locked'], states(page)
    assert page.locator('#lyrics').input_value() == '' and not page.locator('#remixBanner').is_visible()

    # ---- a finished song: the page logic is driven by a scripted job, not by the engine
    target = remixable[0]
    polls = {'n': 0, 'started': False}

    def scripted_job(route):
        if not polls['started']:                                    # idle until Make is pressed
            return route.fulfill(json={'job': None, 'engine': {'state': 'ready', 'error': ''}, 'gpu': None, 'log': [], 'log_seq': 0})
        polls['n'] += 1
        running = polls['n'] < 3
        job = {'id': 9001, 'kind': 'song', 'state': 'running' if running else 'done', 'partial': None, 'error': None,
               'elapsed': polls['n'] * 0.7, 'result': None if running else {'song': target},
               'progress': {'overall': 0.4, 'label': 'Scripted test job', 'steps': []} if running else None}
        route.fulfill(json={'job': job, 'engine': {'state': 'ready', 'error': ''}, 'gpu': None, 'log': [], 'log_seq': 0})

    page.route('**/api/song', lambda route: (polls.update(started=True), route.fulfill(json={'job': 9001})))
    page.route('**/api/job*', scripted_job)
    page.locator('#modeListen').click()
    page.locator('#songs .song .remix').first.click()
    page.locator('.remix-choice[data-remix="take"]').click()
    page.locator('#btnMake').click()
    page.wait_for_function("!document.querySelector('#progress').hidden", polling=100, timeout=5000)
    assert not page.locator('#btnMake').is_visible()
    page.wait_for_function("!document.querySelector('#made').hidden", polling=100, timeout=10000)
    assert target['title'] in page.locator('#madeTitle').inner_text()
    assert page.locator('#btnMake').is_visible() and not page.locator('#remixBanner').is_visible()
    page.screenshot(path=str(root / 'logs/guided-made.png'))
    page.locator('#madeListen').click()
    assert page.locator('body').get_attribute('data-mode') == 'listen'
    page.wait_for_function("!document.querySelector('#audio').paused", polling=100, timeout=8000)
    assert page.locator('#npTitle').inner_text() == target['title']
    page.locator('#modeCreate').click()
    page.locator('#madeNew').click()
    assert states(page) == ['current', 'locked', 'locked', 'locked', 'locked'], states(page)
    assert not page.locator('#made').is_visible()
    page.unroute('**/api/job*')
    assert not errors, errors

    # ---- phone
    mobile = browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True,
        user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1')
    phone = mobile.new_page()
    phone_errors = []
    phone.on('pageerror', lambda error: phone_errors.append(str(error)))
    phone.add_init_script(SPY)
    phone.goto(base)
    phone.wait_for_function("document.querySelector('.step').dataset.state==='current'", polling=100)
    assert phone.evaluate('document.documentElement.scrollWidth<=innerWidth')
    assert phone.locator('#btnLyrics').evaluate('(e)=>{const r=e.getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight}'), 'Step 1 button should fit on the first phone screen.'
    phone.screenshot(path=str(root / 'logs/guided-phone.png'))
    phone.locator('#modeListen').click()
    phone.locator('#btnPlayAll').click()
    phone.wait_for_function("document.querySelector('#audio').currentTime>1", timeout=15000, polling=100)
    phone.wait_for_function("document.querySelector('#viz').dataset.mode==='measured'", timeout=15000, polling=100)
    phone.wait_for_timeout(1500)
    marks = []
    for _ in range(8):
        marks.append(phone.evaluate(INK, 'viz')['sum'])
        phone.wait_for_timeout(180)
    assert len(set(marks)) >= 3, 'Phone bars must move too.'
    assert phone.evaluate('__contexts') == 0
    phone.evaluate('window.scrollTo(0,0)')
    phone.wait_for_timeout(150)
    assert phone.evaluate('document.documentElement.scrollWidth<=innerWidth')
    assert phone.locator('#btnPlay').evaluate('(e)=>{const r=e.getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight}')
    phone.screenshot(path=str(root / 'logs/player-phone.png'))
    assert not phone_errors, phone_errors
    browser.close()
    print(json.dumps({'status': 'PASS', 'songs': len(songs), 'blocked_generation_requests': [kind for kind, _ in blocked],
                      'checks': ['first visit shows step 1 only', 'locked steps stay closed', 'idea required', 'own-lyrics path',
                                 'Next through all five steps', 'recap', 'back keeps later steps', 'every start is fresh at step 1', 'Continue that song restores the draft', 'Dismiss forgets it',
                                 'emptied lyrics re-lock', 'step-by-step switch remembered', 'measured visuals API',
                                 'live bars change with animation frames frozen', 'waveform drawn', 'no Web Audio',
                                 'waveform tap seeks', 'bars rest when paused', 'Remix dialog', 'Remix opens chosen step in Create',
                                 'Remix clears seed', 'list Remix', 'start fresh', 'finished-song panel (scripted job, no engine)', 'phone layout and bars', 'no page errors'],
                      'limitation': 'Chromium with an iPhone user agent, not a physical phone or the Pinokio window.'}))
