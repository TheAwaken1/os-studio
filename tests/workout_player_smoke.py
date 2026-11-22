"""Real-media player checks. iPhone UA/visibility tests are NOT a physical iOS lock test."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
parser=argparse.ArgumentParser()
parser.add_argument('--base',required=True)
args=parser.parse_args()
base=args.base.rstrip('/')
root=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.add_init_script("""window.__mediaHandlers={};if(navigator.mediaSession){const original=navigator.mediaSession.setActionHandler.bind(navigator.mediaSession);navigator.mediaSession.setActionHandler=(name,handler)=>{window.__mediaHandlers[name]=handler;return original(name,handler)}}""")
    songs=page.request.get(base+'/api/songs').json()['songs']
    assert len(songs)>=2, 'Need two real library songs for queue transition verification.'
    page.goto(base)
    page.wait_for_selector('#songs .song',state='attached')  # Create shows the steps only; the list lives in Listen
    page.locator('#modeListen').click()
    assert page.locator('body').get_attribute('data-mode')=='listen'
    page.locator('#btnPlayAll').click()
    page.wait_for_function("document.querySelector('#audio').currentTime>0.15 && !document.querySelector('#audio').paused",timeout=8000)
    first=page.locator('#audio').get_attribute('src')
    page.evaluate('window.__persistentAudio=document.querySelector("#audio")')
    page.locator('#btnNext').click()
    page.wait_for_function('(old)=>document.querySelector("#audio").getAttribute("src")!==old',arg=first)
    page.wait_for_function('document.querySelector("#audio").currentTime>0.1')
    page.locator('#btnPrev').click()
    page.wait_for_function('(first)=>document.querySelector("#audio").getAttribute("src")===first',arg=first)
    # all -> one: real ended event must replay this track.
    page.locator('#btnRepeat').click()
    assert 'one' in page.locator('#btnRepeat').inner_text().lower()
    page.wait_for_function('Number.isFinite(document.querySelector("#audio").duration)')
    page.evaluate('document.querySelector("#audio").currentTime=document.querySelector("#audio").duration-0.12')
    page.wait_for_function('document.querySelector("#audio").currentTime<3 && !document.querySelector("#audio").paused',timeout=8000)
    assert page.locator('#audio').get_attribute('src')==first
    # one -> off: end of library stops.
    page.locator('#btnRepeat').click()
    assert 'off' in page.locator('#btnRepeat').inner_text().lower()
    page.locator('#songs .song').last.click()
    page.wait_for_function('Number.isFinite(document.querySelector("#audio").duration)')
    last=page.locator('#audio').get_attribute('src')
    page.evaluate('document.querySelector("#audio").currentTime=document.querySelector("#audio").duration-0.12')
    page.wait_for_function('document.querySelector("#audio").ended',timeout=8000)
    assert page.locator('#audio').get_attribute('src')==last
    # Continuous off stops after any track, but manual next remains usable.
    page.locator('#btnRepeat').click()
    page.locator('#btnContinuous').click()
    page.locator('#songs .song').first.click()
    page.wait_for_function('Number.isFinite(document.querySelector("#audio").duration)')
    page.evaluate('document.querySelector("#audio").currentTime=document.querySelector("#audio").duration-0.12')
    page.wait_for_function('document.querySelector("#audio").ended',timeout=8000)
    assert page.locator('#audio').get_attribute('src')==first
    page.locator('#btnContinuous').click()
    page.evaluate('Math.random=()=>0')
    page.locator('#btnShuffle').click()
    if len(songs)>=3:
        # Starting a different song in shuffle must start a full, nonrepeating rotation.
        page.locator('#songs .song').nth(2).click()
        rotation=[page.locator('#audio').get_attribute('src')]
        for _ in range(len(songs)-1):
            page.locator('#btnNext').click()
            rotation.append(page.locator('#audio').get_attribute('src'))
        assert len(set(rotation))==len(songs), rotation
    page.locator('#btnNext').click()
    page.wait_for_function('(old)=>document.querySelector("#audio").getAttribute("src")!==old',arg=first)
    # Lock-screen API handlers are exercised directly, not claimed as real OS control tests.
    assert page.evaluate("['play','pause','nexttrack','previoustrack','seekto'].every(k=>typeof __mediaHandlers[k]==='function')")
    page.evaluate('__mediaHandlers.pause()')
    assert page.locator('#audio').evaluate('(a)=>a.paused')
    page.evaluate('__mediaHandlers.play()')
    page.wait_for_function('!document.querySelector("#audio").paused')
    assert page.evaluate('navigator.mediaSession.metadata.title')==page.locator('#npTitle').inner_text()
    before=page.locator('#audio').get_attribute('src')
    page.evaluate('__mediaHandlers.nexttrack()')
    page.wait_for_function('(old)=>document.querySelector("#audio").getAttribute("src")!==old',arg=before)
    assert page.evaluate('window.__persistentAudio===document.querySelector("#audio")')
    page.locator('#modeCreate').click()
    assert page.locator('#miniPlayer').is_visible()
    page.locator('#miniPlay').click()
    assert page.locator('#audio').evaluate('(a)=>a.paused')
    page.locator('#miniPlay').click()
    page.wait_for_function('!document.querySelector("#audio").paused')
    page.locator('#miniExpand').click()
    assert page.locator('body').get_attribute('data-mode')=='listen'
    page.screenshot(path=str(root/'logs/listen-desktop.png'),full_page=True)
    page.reload()
    page.wait_for_selector('#songs .song')
    assert page.locator('body').get_attribute('data-mode')=='listen'
    assert page.locator('#audio').evaluate('(a)=>a.paused'), 'Reload must not autoplay without user gesture.'
    assert not errors,errors
    # Check direct native-element route under iPhone identification, using real MP3 data.
    mobile=browser.new_context(viewport={'width':390,'height':844},is_mobile=True,has_touch=True,user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1')
    phone=mobile.new_page()
    phone.add_init_script("""window.__contexts=0;if(window.AudioContext){const Original=window.AudioContext;window.AudioContext=new Proxy(Original,{construct(t,args){window.__contexts++;return Reflect.construct(t,args)}})}""")
    phone.goto(base)
    phone.wait_for_selector('#songs .song',state='attached')
    phone.locator('#modeListen').click()
    phone.locator('#btnPlayAll').click()
    phone.wait_for_function('document.querySelector("#audio").currentTime>0.2',timeout=8000)
    assert phone.evaluate('__contexts')==0, 'iPhone music must not depend on Web Audio.'
    phone.evaluate('window.scrollTo(0,0)')
    phone.wait_for_timeout(150)
    assert phone.locator('#btnPlay').evaluate('(e)=>{const r=e.getBoundingClientRect();return r.top>=0 && r.bottom<=innerHeight}'), 'Phone playback controls should fit above the fold.'
    phone.locator('#songs').scroll_into_view_if_needed()
    phone.wait_for_function('!document.querySelector("#miniPlayer").hidden',timeout=3000)
    assert phone.locator('#miniPlayer').is_visible(), 'Mini-player should follow library browsing in Listen mode.'
    assert phone.evaluate('document.documentElement.scrollWidth<=innerWidth')
    phone.evaluate('Object.defineProperty(document,"hidden",{configurable:true,get:()=>true});document.dispatchEvent(new Event("visibilitychange"))')
    t=phone.locator('#audio').evaluate('(a)=>a.currentTime')
    phone.wait_for_timeout(700)
    assert phone.locator('#audio').evaluate('(a)=>a.currentTime')>t
    before=phone.locator('#audio').get_attribute('src')
    phone.evaluate('document.querySelector("#audio").currentTime=document.querySelector("#audio").duration-0.12')
    phone.wait_for_function('(old)=>document.querySelector("#audio").getAttribute("src")!==old',arg=before,timeout=8000)
    phone.evaluate('Object.defineProperty(document,"hidden",{configurable:true,get:()=>false});document.dispatchEvent(new Event("visibilitychange"))')
    phone.screenshot(path=str(root/'logs/listen-phone.png'),full_page=True)
    response=phone.request.get(base+'/media/'+songs[0]['id'],headers={'Range':'bytes=0-1023'})
    assert response.status==206 and len(response.body())==1024
    browser.close()
    print(json.dumps({'status':'PASS','real_library_tracks':len(songs),'checks':['Listen/Create','real MP3 play','next/previous','real ended repeat one','repeat off end of library','continuous off','shuffle','Media Session handler registration and invocation','single persistent audio element','mini-player','preferences','iPhone native audio route','visibility-path playback and next track','mobile layout','HTTP byte-range seeking','no page errors'],'limitation':'Not a physical iPhone lock-screen or private Tailscale-network test.'}))
