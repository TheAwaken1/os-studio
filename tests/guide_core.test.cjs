const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const roots = [path.resolve(__dirname, '../app')];
const none = { idea: false, lyrics: false, style: false };
const idea = { idea: true, lyrics: false, style: true };
const full = { idea: true, lyrics: true, style: true };

for (const root of roots) {
  const Guide = require(path.join(root, 'web/guide-core.js'));
  const all = (g, facts) => Guide.STEPS.map((_, i) => g.status(i, facts));

  test(`${root}: a first visit opens step 1 only`, () => {
    const g = new Guide();
    assert.deepEqual(all(g, none), ['current', 'locked', 'locked', 'locked', 'locked']);
    assert.equal(g.go(1, none), false);
    assert.match(g.advance(none), /about/i);
    assert.equal(g.current(none), 0);
  });
  test(`${root}: each finished step opens exactly the next one`, () => {
    const g = new Guide();
    assert.equal(g.advance(idea), '');
    assert.deepEqual(all(g, idea), ['done', 'current', 'locked', 'locked', 'locked']);
    assert.match(g.advance(idea), /lyrics/i);
    assert.equal(g.advance(full), '');
    assert.equal(g.advance(full), '');
    assert.equal(g.advance(full), '');
    assert.deepEqual(all(g, full), ['done', 'done', 'done', 'done', 'current']);
    assert.equal(g.advance(full), '', 'the last step has nowhere further to go');
    assert.equal(g.current(full), 4);
  });
  test(`${root}: own lyrics skip the idea, and a missing sound blocks the singer`, () => {
    const g = new Guide();
    g.ownLyrics = true;
    assert.equal(g.advance(none), '');
    assert.equal(g.current(none), 1);
    const noSound = { idea: false, lyrics: true, style: false };
    assert.equal(g.advance(noSound), '');
    assert.match(g.advance(noSound), /sound/i);
    assert.equal(g.current(noSound), 2);
  });
  test(`${root}: going back keeps later steps reachable; removing content locks them again`, () => {
    const g = new Guide({ reached: 4, at: 4 });
    assert.equal(g.go(1, full), true);
    assert.deepEqual(all(g, full), ['done', 'current', 'ahead', 'ahead', 'ahead']);
    assert.deepEqual(all(g, idea), ['done', 'current', 'locked', 'locked', 'locked']);
    assert.equal(g.go(3, idea), false);
    g.go(4, full);
    assert.equal(g.current(idea), 1, 'the open step falls back to the first one still missing something');
    assert.equal(g.back(full), 3);
  });
  test(`${root}: remix jumps, restart, saved state and switched-off mode`, () => {
    const g = new Guide();
    assert.equal(g.jump(2, full), 2);
    assert.deepEqual(all(g, full), ['done', 'done', 'current', 'ahead', 'ahead']);
    assert.equal(g.jump(4, { idea: true, lyrics: false, style: true }), 1, 'a song without lyrics cannot land past step 2');
    const copy = new Guide(JSON.parse(JSON.stringify(g)));
    assert.deepEqual(copy.toJSON(), g.toJSON());
    g.restart();
    assert.deepEqual(all(g, none), ['current', 'locked', 'locked', 'locked', 'locked']);
    g.on = false;
    assert.deepEqual(all(g, none), ['open', 'open', 'open', 'open', 'open']);
    for (const junk of [null, 'x', { reached: 99, at: -3 }, { reached: 1, at: 4 }]) {
      const safe = new Guide(junk);
      assert.ok(safe.at <= safe.reached && safe.reached <= 4 && safe.at >= 0);
    }
  });
  test(`${root}: an older saved draft resumes where its content leaves off`, () => {
    const withLyrics = new Guide(); withLyrics.adopt(full);
    assert.equal(withLyrics.current(full), 4);
    const withIdea = new Guide(); withIdea.adopt(idea);
    assert.deepEqual(all(withIdea, idea), ['current', 'ahead', 'locked', 'locked', 'locked']);
    const empty = new Guide(); empty.adopt(none);
    assert.equal(empty.reached, 0);
  });
  test(`${root}: page contract for the guided steps, Remix and the measured player`, () => {
    const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
    const js = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
    const steps = [...html.matchAll(/class="card step[^"]*"[^>]*data-step="(\d)"/g)].map(m => m[1]);
    assert.deepEqual(steps, ['0', '1', '2', '3', '4']);
    for (const id of ['composer', 'guideToggle', 'btnOwnLyrics', 'voiceInvite', 'recap', 'made', 'remixDlg', 'remixBanner', 'wave', 'viz', 'npCover', 'miniBar'])
      assert.ok(html.includes(`id="${id}"`), id);
    assert.deepEqual([...html.matchAll(/data-remix="([a-z]+)"/g)].map(m => m[1]), ['take', 'lyrics', 'style', 'voice']);
    assert.ok(html.indexOf('guide-core.js') < html.indexOf('/static/app.js'));
    assert.doesNotMatch(html, /Loaded into the composer/);
    assert.doesNotMatch(js, /Loaded into the composer/);
    // the player draws from measured data on a timer: no Web Audio graph, no animation frames
    assert.doesNotMatch(js, /createMediaElementSource|createAnalyser\(\)[\s\S]{0,200}frequencyBinCount|requestAnimationFrame/);
    assert.match(js, /\/viz/);
    const css = fs.readFileSync(path.join(root, 'web/guide.css'), 'utf8');
    assert.doesNotMatch(css, /backdrop-filter|animation:.*infinite/);
    assert.match(css, /body\[data-mode="create"\] \.stage\s*\{[^}]*display:\s*none/);
  });
}
