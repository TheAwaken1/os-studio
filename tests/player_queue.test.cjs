const test = require('node:test');
const assert = require('node:assert/strict');
const Queue = require('../app/web/player-core.js');

test('continuous library queue wraps, with manual previous/next', () => {
  const q = new Queue(); q.setSongs(['a','b','c']); q.select('a');
  assert.equal(q.next(), 'b'); assert.equal(q.next(), 'c'); assert.equal(q.next(), 'a');
  assert.equal(q.previous(), 'c');
});
test('continuous off stops at song end but allows manual navigation', () => {
  const q = new Queue({continuous:false}); q.setSongs(['a','b']); q.select('a');
  assert.equal(q.next(), null); assert.equal(q.current, 'a');
  assert.equal(q.next({manual:true}), 'b');
});
test('repeat one affects ended playback, never manual next', () => {
  const q = new Queue({repeat:'one'}); q.setSongs(['a','b']); q.select('a');
  assert.equal(q.next(), 'a'); assert.equal(q.next({manual:true}), 'b');
  assert.equal(q.next({manual:true}), 'a');
  assert.equal(q.previous(), 'b');
});
test('repeat off stops at last track', () => {
  const q = new Queue({repeat:'off'}); q.setSongs(['a','b']); q.select('b');
  assert.equal(q.next(), null); assert.equal(q.next({manual:true}), null);
  assert.equal(q.previous(), 'a');
});
test('shuffle plays every track before a new cycle without immediate repeat', () => {
  const q = new Queue({shuffle:true, random:()=>0}); q.setSongs(['a','b','c','d']); q.select('a');
  const cycle = [q.current, q.next(), q.next(), q.next()];
  assert.equal(new Set(cycle).size, 4);
  const last = q.current; assert.notEqual(q.next(), last);
});
test('toggle shuffle preserves current track, disabling restores library order', () => {
  const q = new Queue({random:()=>0}); q.setSongs(['a','b','c']); q.select('b');
  q.setShuffle(true); assert.equal(q.current,'b');
  q.setShuffle(false); assert.equal(q.current,'b'); assert.equal(q.next(),'c');
});
test('library refresh removes deleted IDs and preserves selection', () => {
  const q = new Queue(); q.setSongs(['a','b','c']); q.select('b');
  q.setSongs(['x','b','c']); assert.equal(q.current,'b'); assert.equal(q.next(),'c');
  q.setSongs(['x']); assert.equal(q.current,null); assert.equal(q.select('x'),'x');
  assert.equal(q.select('missing'),null);
});
test('empty and one-song libraries are safe', () => {
  const q = new Queue(); q.setSongs([]); assert.equal(q.next(),null); assert.equal(q.previous(),null);
  q.setSongs(['a','a']); q.select('a'); assert.equal(q.next(),'a'); assert.equal(q.order.length,1);
});
test('peek never consumes queue position', () => {
  const q=new Queue(); q.setSongs(['a','b']); q.select('a');
  assert.equal(q.peek(),'b'); assert.equal(q.current,'a'); assert.equal(q.next(),'b');
});
