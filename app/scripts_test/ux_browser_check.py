"""Browser checks against the real app, without running GPU jobs."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.add_init_script("try{localStorage.setItem('yss.guide','off')}catch(e){}")  # every step open: this test is about the controls, not the guide
    page.goto('http://127.0.0.1:7862')
    page.wait_for_selector('#shapePresets button')
    assert not page.locator('#shapeCustomize').evaluate('(e)=>e.open')
    assert not page.locator('#lyricsEditingTools').evaluate('(e)=>e.open')
    page.locator('#lyrics').fill('[verse]\nKeep my words\n\n[chorus]\nKeep my hook')
    page.locator('#shapePresets button').filter(has_text='Short').click()
    assert 'Keep my words' in page.locator('#lyrics').input_value()
    page.locator('#shapeCustomize summary').click()
    before = page.locator('#shapeSeq .sec').count()
    page.locator('#shapeAdd button').filter(has_text='+ bridge').click()
    assert page.locator('#shapeSeq .sec').count() == before + 1
    assert 'undefined' not in page.locator('#lyrics').input_value()
    page.locator('#shapeLines').select_option('6')
    page.locator('#btnBrowseStyles').click()
    catalog = page.request.get('http://127.0.0.1:7862/api/state').json()
    assert page.locator('.style-result').count() == len(catalog['style_library'])
    names = [p['name'] for p in catalog['style_library']]
    assert 'Country (straight)' not in names and 'Pop (modern radio)' not in names
    page.locator('#styleCategory').select_option('Rock & Heavy')
    assert page.locator('.style-result').count() == 3
    page.locator('#styleCategory').select_option('Electronic & Dance')
    assert page.locator('.style-result').count() == 2
    page.locator('#styleCategory').select_option('Soul & R&B')
    assert page.locator('.style-result').count() == 2
    page.locator('#styleSearch').fill('nomatchzzz')
    assert page.locator('.style-result').count() == 0
    page.locator('#styleSearch').fill('Motown')
    page.locator('.style-result').click()
    assert 'Motown' in page.locator('#style').input_value()
    page.reload()
    page.wait_for_selector('#shapePresets button')
    assert page.locator('#lyrics').input_value() == ''
    # every start is fresh; the unfinished song comes back through "Continue that song"
    with page.expect_navigation():
        page.locator('#resumeGo').click()
    page.wait_for_selector('#shapePresets button')
    assert 'Motown' in page.locator('#styleSelection').inner_text()
    assert 'Keep my words' in page.locator('#lyrics').input_value()
    assert page.locator('#shapeLines').input_value() == '6'
    selected = page.locator('#voiceChips .chip.on').inner_text()
    page.locator('.voice-listen').first.click()
    page.wait_for_function("document.querySelector('#voiceAudio').readyState >= 2", timeout=20000)
    page.wait_for_timeout(1200)
    media = page.locator('#voiceAudio').evaluate('(a)=>({duration:a.duration,time:a.currentTime,paused:a.paused,error:a.error})')
    assert media['duration'] > 0 and media['time'] > 0 and not media['paused'] and media['error'] is None, media
    assert page.locator('#voiceChips .chip.on').inner_text() == selected
    page.locator('#voicePreviewClose').click()
    assert page.locator('#voiceAudio').evaluate('(a)=>a.paused')
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'mobile horizontal overflow'
    page.locator('#btnBrowseStyles').click()
    assert page.locator('#styleDlg').is_visible()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.locator('#styleClose').click()
    page.set_viewport_size({'width':1440,'height':1000})
    page.locator('#lyricsCard').scroll_into_view_if_needed()
    page.screenshot(path=str(ROOT / 'logs/ux-guided-lyrics.png'))
    assert not errors, errors
    print(json.dumps({'status':'PASS', 'checks':['guided defaults','non-destructive structure','add section','line count autosave','deduplicated library styles','rock and dance categories','category filter','empty search','style selection autosave','real saved recording playback','preview does not select voice','preview close','390px responsive layout','no page errors'], 'media':media}))
    browser.close()
