"""Target length UI + real read-only planning API. Training/generation POST is aborted."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--base', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    errors, intercepted = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.add_init_script("try{localStorage.setItem('yss.guide','off')}catch(e){}")  # every step open: this test is about the controls, not the guide
    page.goto(args.base)
    page.wait_for_selector('#shapePresets button')
    assert page.locator('#targetLength').input_value() == 'auto'
    assert not page.locator('#lengthCustom').is_visible()
    assert not page.locator('#shapeLines').is_disabled()
    original = '[verse]\nKeep my original words exactly\nA second line of my own\n\n[chorus]\nMy original chorus'
    page.locator('#lyrics').fill(original)
    page.locator('#options').evaluate('(e)=>e.open=true')
    page.locator('#memory').select_option('standard')
    page.locator('#targetLength').select_option('custom')
    page.locator('#lengthMinutes').fill('2')
    page.locator('#lengthSeconds').fill('30')
    assert page.locator('#lengthMinutes').evaluate('(e)=>getComputedStyle(e).backgroundColor') != 'rgb(255, 255, 255)', 'Number fields should match the dark app theme.'
    page.wait_for_function('document.querySelector("#lengthHelp").textContent.includes("next draft plans")')
    assert page.locator('#lyrics').input_value() == original
    assert page.locator('#shapeLines').is_disabled()
    page.get_by_role('button', name='Long', exact=True).click()
    sections = page.locator('#shapeSeq .sec').evaluate_all('(nodes)=>nodes.map(n=>n.dataset.s)')
    response = page.request.post(args.base + '/api/lyrics-plan', data={
        'structure': sections, 'lines': 4, 'target_seconds': 150, 'memory': 'standard'})
    assert response.ok, response.text()
    plan = response.json()
    assert sections[0] == 'intro' and plan['counts'][0] == 0
    assert plan['target_seconds'] == 150 and plan['limit_seconds'] == 360
    page.wait_for_function('(n)=>document.querySelector("#shapeMeta").textContent.includes(n+" lines")', arg=plan['total_lines'])
    assert page.locator('#lyrics').input_value() == original
    page.reload()
    page.wait_for_selector('#shapePresets button')
    assert page.locator('#lyrics').input_value() == '' and page.locator('#targetLength').input_value() == 'auto'
    # every start is fresh; the unfinished song comes back through "Continue that song"
    with page.expect_navigation():
        page.locator('#resumeGo').click()
    page.wait_for_selector('#shapePresets button')
    assert page.locator('#targetLength').input_value() == 'custom'
    assert page.locator('#lengthMinutes').input_value() == '2'
    assert page.locator('#lengthSeconds').input_value() == '30'
    assert page.locator('#lyrics').input_value() == original
    page.route('**/api/lyrics', lambda route: (intercepted.append(route.request.post_data_json), route.abort()))
    page.locator('#idea').fill('A browser test, do not generate')
    page.locator('#options').evaluate('(e)=>e.open=true')
    page.locator('#memory').select_option('low')
    page.locator('#lengthMinutes').fill('4')
    page.locator('#btnLyrics2').click()
    assert not intercepted
    assert '3:00' in page.locator('#lengthHelp').inner_text()
    page.locator('#lengthMinutes').fill('2')
    page.locator('#lengthSeconds').fill('60')
    page.locator('#btnLyrics2').click()
    assert not intercepted
    assert '0 to 59' in page.locator('#lengthHelp').inner_text()
    page.locator('#lengthSeconds').fill('30')
    page.wait_for_function('document.querySelector("#lengthHelp").textContent.includes("next draft plans")')
    page.locator('#btnLyrics2').click()
    page.wait_for_function('document.querySelector("#toasts").children.length > 0')
    page.wait_for_timeout(200)
    assert len(intercepted) == 1, intercepted
    assert intercepted[0]['target_seconds'] == 150 and intercepted[0]['memory'] == 'low'
    assert page.locator('#lyrics').input_value() == original
    page.locator('#targetLength').select_option('auto')
    assert not page.locator('#shapeLines').is_disabled()
    page.locator('#targetLength').select_option('custom')
    page.wait_for_function('document.querySelector("#lengthHelp").textContent.includes("next draft plans")')
    page.locator('#toasts').evaluate('(e)=>e.replaceChildren()')
    page.locator('#lyricsCard').screenshot(path=str(root / 'logs/target-length-desktop.png'))
    page.set_viewport_size({'width': 390, 'height': 844})
    page.locator('.length-control').scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert page.locator('#lengthMinutes').is_visible() and page.locator('#lengthSeconds').is_visible()
    page.locator('.length-control').screenshot(path=str(root / 'logs/target-length-phone.png'))
    assert not errors, errors
    browser.close()
    print(json.dumps({'status':'PASS','plan':plan,'pasted_lyrics_unchanged':True,'custom_target_persists':True,
                      'blocked_generation_requests':intercepted,'page_errors':errors,
                      'note':'Actual planning API; the single lyrics POST was aborted before reaching the server.'}))
