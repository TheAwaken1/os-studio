'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const roots = [path.resolve(__dirname, '../app')];
const newIds = ['modeCreate','modeListen','listeningStage','btnPrev','btnNext','btnContinuous','btnShuffle','btnRepeat','queueMeta','upNext','btnPlayAll','miniPlayer','miniExpand','miniTitle','miniSub','miniPrev','miniPlay','miniNext'];
function tag(html, id) {
  const found = html.match(new RegExp('<[a-z][^>]*\\bid="' + id + '"[^>]*>', 'i'));
  assert.ok(found, `Missing #${id}`);
  return found[0];
}
for (const root of roots) {
  test(`${root}: focused, responsive low-GPU Listen stylesheet`, () => {
    const css = fs.readFileSync(path.join(root, 'web/listen.css'), 'utf8');
    assert.match(css, /body\[data-mode=["']?listen["']?\] \.composer\s*\{[^}]*display:\s*none/);
    assert.match(css, /body\[data-mode=["']?listen["']?\] \.stage/);
    assert.match(css, /position:\s*fixed/);
    assert.match(css, /min-height:\s*44px/);
    assert.match(css, /@media \(max-width: 720px\)/);
    assert.doesNotMatch(css, /backdrop-filter|animation:.*infinite/);
  });
  test(`${root}: complete accessible Listen markup contract`, () => {
    const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
    for (const id of newIds) tag(html, id);
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
    assert.equal(ids.length, new Set(ids).size, 'IDs must be unique');
    for (const id of ['modeCreate','btnContinuous']) assert.match(tag(html,id), /aria-pressed="true"/);
    for (const id of ['modeListen','btnShuffle']) assert.match(tag(html,id), /aria-pressed="false"/);
    for (const id of ['btnPrev','btnNext','miniExpand','miniPrev','miniPlay','miniNext']) {
      assert.match(tag(html,id), /aria-label="[^"]+"/);
      assert.match(tag(html,id), /type="button"/);
    }
    assert.ok(html.indexOf('id="btnPrev"') < html.indexOf('id="btnPlay"'));
    assert.ok(html.indexOf('id="btnPlay"') < html.indexOf('id="btnNext"'));
    assert.match(tag(html,'queueMeta'), /role="status"/);
    assert.match(tag(html,'miniPlayer'), /\bhidden\b/);
    assert.match(tag(html,'audio'), /preload="auto"/);
    assert.match(tag(html,'audio'), /\bplaysinline\b/);
    assert.doesNotMatch(tag(html,'audio'), /\bautoplay\b/);
    assert.equal((html.match(/<audio\b/g) || []).length, 3, 'No new audio elements');
    assert.match(html, /href="\/static\/listen\.css\?v=\{\{v\}\}"/);
    assert.match(html, /Repeat all/);
    assert.match(html, /turn on Remote in Pinokio/i);
    assert.match(html, /scan the QR code/i);
  });
}
