const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
for (const app of [path.join(root, 'app')]) {
  test(`${app}: OS Studio app identity keeps YuE2 attribution`, () => {
    const html = fs.readFileSync(path.join(app, 'web/index.html'), 'utf8');
    const js = fs.readFileSync(path.join(app, 'web/app.js'), 'utf8');
    assert.match(html, /<title>OS Studio<\/title>/);
    assert.match(html, /<h1\b[^>]*>OS Studio<\/h1>/);
    assert.match(html, /Create\. Sing\. Listen\./);
    assert.match(html, /Open-source music models/);
    assert.match(html, /OS means Open Source/);
    assert.match(js, /album: "OS Studio"/);
    assert.doesNotMatch(js, /Song Studio isn't responding/);
  });
}
test('Pinokio listing is OS Studio with an Open OS Studio action', async () => {
  const launcher = require(path.join(root, 'pinokio.js'));
  const metadata = JSON.parse(fs.readFileSync(path.join(root, 'pinokio.json'), 'utf8'));
  assert.equal(launcher.title, 'OS Studio');
  assert.equal(metadata.title, 'OS Studio');
  const menu = await launcher.menu({}, {
    exists: () => true, running: script => script === 'start.js',
    local: () => ({url: 'http://127.0.0.1:42003'}),
  });
  assert.equal(menu[0].text, 'Open OS Studio');
});
