"""Verify live OS Studio branding and the actual Media Session artwork response."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--base', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(args.base)
    page.wait_for_selector('#songs .song', state='attached')  # Create shows the steps only
    assert page.title() == 'OS Studio'
    assert page.locator('.brand h1').inner_text() == 'OS Studio'
    assert 'Open-source music models' in page.locator('.brand p').inner_text()
    page.locator('#modeListen').click()
    page.locator('#btnPlayAll').click()
    page.wait_for_function('document.querySelector("#audio").currentTime > 0.1')
    artwork = page.evaluate('''async () => {
      const metadata = navigator.mediaSession.metadata;
      const art = metadata.artwork[0];
      const response = await fetch(art.src);
      if (!response.ok) throw new Error('Artwork fetch failed: ' + response.status);
      const image = await createImageBitmap(await response.blob());
      const canvas = document.createElement('canvas');
      canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext('2d'); context.drawImage(image, 0, 0);
      const pixel = (x,y) => [...context.getImageData(x,y,1,1).data];
      return {album:metadata.album,src:art.src,sizes:art.sizes,width:image.width,height:image.height,
        sameOrigin:new URL(art.src).origin === location.origin,
        topLeft:pixel(0,0),bottomRight:pixel(image.width-1,image.height-1)};
    }''')
    assert artwork['album'] == 'OS Studio'
    assert 'v=os-studio-1' in artwork['src'] and artwork['sameOrigin']
    assert artwork['sizes'] == '1024x1024' and artwork['width'] == artwork['height'] == 1024
    assert artwork['topLeft'] == [254, 189, 73, 255]
    assert artwork['bottomRight'] == [154, 221, 216, 255]
    page.evaluate('window.scrollTo(0,0)')
    page.screenshot(path=str(root / 'logs/os-studio-desktop.png'))
    page.set_viewport_size({'width': 390, 'height': 844})
    page.wait_for_timeout(200)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert page.locator('.brand h1').is_visible()
    page.screenshot(path=str(root / 'logs/os-studio-phone.png'))
    assert not errors, errors
    browser.close()
    result = {'status': 'PASS', 'branding': 'OS Studio', 'artwork': artwork, 'page_errors': errors}
    (root / 'logs/branding-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
