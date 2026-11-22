const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const roots = [path.resolve(__dirname, '../app')];
function harness(root) {
  const nodes = new Map();
  function node() { return { value: '', hidden: true, textContent: '', children: [], dataset: {}, style: {},
    classList: { add() {}, remove() {}, toggle() {} },
    firstElementChild: { style: {} }, lastElementChild: {},
    append(...items) { this.children.push(...items); }, replaceChildren() { this.children = []; },
    setAttribute() {}, removeAttribute() {}, addEventListener() {}, focus() {}, showModal() {}, pause() {} }; }
  const $ = id => { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id); };
  const requests = [];
  class AudioContext {
    resume() { return Promise.resolve(); } close() { return Promise.resolve(); }
    createAnalyser() { return { fftSize: 1024, getFloatTimeDomainData() {}, connect() { return this; } }; }
    createGain() { return { gain: {}, connect() {} }; }
    createMediaStreamSource() { return { connect() {} }; }
    decodeAudioData() { return Promise.reject(new Error('test unreadable recording')); }
  }
  class MediaRecorder { constructor() { this.mimeType = 'audio/webm'; } start() {} }
  const context = { document: { getElementById: $, createElement: node }, window: { MediaRecorder, YSSPlaybackQueue: require(path.join(root, 'web/player-core.js')), YSSLengthUI: require(path.join(root, 'web/length-ui.js')) },
    navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [] }) } },
    AudioContext, MediaRecorder, performance: { now: () => 0 }, setInterval: () => 1, clearInterval() {},
    Blob, File, URL, setTimeout() {}, clearTimeout() {},
    fetch: async (url, init) => { requests.push({ url, init }); return { ok: false, status: 400, json: async () => ({ detail: 'intercepted; no backend' }) }; } };
  const code = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
  vm.runInNewContext(code.replace('  boot();', '  globalThis.ui = { openVoiceDialog, recReset, startRecording, finishRecording, trainVoice, renderVoices, wire, state, rec };'), context);
  return { ...context.ui, $, requests };
}
for (const root of roots) {
  test(`${root}: labelled reference recording choices and correction explanation`, () => {
    const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
    assert.match(html, /<label[^>]*for="voiceRecordingType"/);
    const select = html.match(/<select[^>]*id="voiceRecordingType"[^>]*>([\s\S]*?)<\/select>/);
    assert.ok(select, 'recording type select missing');
    assert.deepEqual([...select[1].matchAll(/value="([^"]+)"/g)].map(m => m[1]), ['talking', 'singing', 'rapping']);
    assert.match(html, /reference recording[^<]*not[^<]*song genre/i);
    assert.match(html, /speech-to-song pitch correction/i);
    assert.match(html, /singing[^<]*rapping[^<]*skip/i);
  });
  test(`${root}: opening defaults talking; reset follows selection`, () => {
    const h = harness(root);
    h.$('voiceRecordingType').value = 'rapping'; h.openVoiceDialog();
    assert.equal(h.$('voiceRecordingType').value, 'talking');
    for (const [type, verb] of [['talking', 'talk'], ['singing', 'sing'], ['rapping', 'rap']]) {
      h.$('voiceRecordingType').value = type; h.recReset();
      assert.match(h.$('recHint').textContent, new RegExp(verb, 'i'));
    }
  });
  test(`${root}: active guidance and failed recording follow chosen type; script talking only`, async () => {
    const h = harness(root);
    for (const [type, verb] of [['talking', 'talk'], ['singing', 'sing'], ['rapping', 'rap']]) {
      h.$('voiceRecordingType').value = type; await h.startRecording();
      assert.equal(h.$('recScript').hidden, type !== 'talking');
      assert.match(h.$('recHint').textContent, new RegExp(verb, 'i'));
      await h.finishRecording();
      assert.match(h.$('recHint').textContent, new RegExp(verb, 'i'));
      h.rec.on = false;
    }
  });
  test(`${root}: training request sends exact chosen type (intercepted, no backend)`, async () => {
    const h = harness(root);
    for (const type of ['talking', 'singing', 'rapping']) {
      h.$('voiceRecordingType').value = type; h.$('voiceName').value = 'UI test';
      h.state.clip = { name: 'reference.wav' }; await h.trainVoice(false);
      const req = h.requests.at(-1);
      assert.equal(new URL(req.url, 'http://ui.test').searchParams.get('recording_type'), type);
      assert.equal(req.init.method, 'POST');
    }
  });
  test(`${root}: selected hint shows recording type with legacy talking fallback`, () => {
    const h = harness(root);
    for (const type of [undefined, 'singing', 'rapping']) {
      h.state.voices = [{ name: 'Existing voice', recording_type: type }]; h.state.voice = 'Existing voice';
      h.renderVoices();
      assert.match(h.$('voiceHint').textContent, new RegExp(type || 'talking', 'i'));
    }
  });
}
