"""Browser smoke check of the actual Pinokio-served UI, no GPU jobs."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
parser = argparse.ArgumentParser()
parser.add_argument('--base', required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless=True)
    page = browser.new_page(viewport={'width':1440,'height':1000})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.add_init_script("try{localStorage.setItem('yss.guide','off')}catch(e){}")  # every step open: this test is about the controls, not the guide
    page.goto(args.base)
    page.wait_for_selector('#shapePresets button')
    assert not page.locator('#shapeCustomize').evaluate('(x)=>x.open')
    page.locator('#btnBrowseStyles').click()
    names = page.locator('.style-result strong').all_text_contents()
    assert len(names) == len(set(names)) == 19, names
    assert 'Country (straight)' not in names and 'Pop (modern radio)' not in names
    page.locator('#styleCategory').select_option('Electronic & Dance')
    assert page.locator('.style-result').count() == 2
    page.locator('.style-result').first.click()
    assert 'Selected:' in page.locator('#styleSelection').inner_text()
    page.locator('.voice-listen').last.click()
    page.wait_for_function("document.querySelector('#voiceAudio').readyState >= 2")
    page.wait_for_timeout(1100)
    media = page.locator('#voiceAudio').evaluate('(a)=>({duration:a.duration,time:a.currentTime,error:a.error})')
    assert media['duration'] > 0 and media['time'] > 0 and media['error'] is None
    page.locator('#voicePreviewClose').click()
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.set_viewport_size({'width':1440,'height':1000})
    page.evaluate('window.scrollTo(0,0)')
    (ROOT / 'logs').mkdir(exist_ok=True)
    page.screenshot(path=str(ROOT / 'logs/pinokio-ui.png'),full_page=True)
    assert not errors, errors
    print(json.dumps({'status':'PASS','base':args.base,'style_count':len(names),'recording_playback':media,'page_errors':errors}))
    browser.close()
