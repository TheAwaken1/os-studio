const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const roots = [path.resolve(__dirname, '../app')];
function harness(root, planResponse) {
  const nodes = new Map(), requests = [], saved = new Map();
  function node() { return {value:'', hidden:false, textContent:'', children:[], dataset:{}, style:{}, attrs:{},
    classList:{toggle(){},add(){},remove(){}}, append(...items){this.children.push(...items)},
    setAttribute(k,v){this.attrs[k]=v}, focus(){}, querySelectorAll(){return []}, firstElementChild:{style:{}}, lastElementChild:{}}; }
  const $ = id => { if (!nodes.has(id)) nodes.set(id,node()); return nodes.get(id); };
  const context = {document:{getElementById:$,createElement:node},
    window:{YSSPlaybackQueue:require(path.join(root,'web/player-core.js')),YSSLengthUI:require(path.join(root,'web/length-ui.js'))},navigator:{},
    localStorage:{getItem:k=>saved.get(k)||null,setItem:(k,v)=>saved.set(k,v)},
    setTimeout(){assert.ok(!this || !this.state,'browser timers cannot use LengthUI as receiver'); return 1},
    clearTimeout(){assert.ok(!this || !this.state,'browser timers cannot use LengthUI as receiver')}, performance:{now:()=>0},
    fetch:async(url,init)=>{requests.push({url,body:JSON.parse(init.body)});
      if(url==='/api/lyrics-plan') return planResponse ? planResponse(JSON.parse(init.body)) : {ok:true,json:async()=>({counts:[10,9,10,9],total_lines:38,estimated_seconds:148.2,target_seconds:150,limit_seconds:360})};
      return {ok:false,status:400,json:async()=>({detail:'Unit-test request intercepted; no GPU job'})};}};
  let code=fs.readFileSync(path.join(root,'web/app.js'),'utf8');
  code=code.replace('  boot();','  globalThis.ui = {state,lengthUI,writeLyrics,saveDraft,loadDraft,lyricsMeta,getLengthTarget:typeof getLengthTarget === "function" ? getLengthTarget : null,currentLengthLimit:typeof currentLengthLimit === "function" ? currentLengthLimit : null,renderLengthSummary:typeof renderLengthSummary === "function" ? renderLengthSummary : null,refreshLengthPlan:typeof refreshLengthPlan === "function" ? refreshLengthPlan : null};');
  vm.runInNewContext(code,context);
  $('targetLength').value='auto'; $('lengthMinutes').value='2'; $('lengthSeconds').value='30'; $('memory').value='standard';
  context.ui.state.structure=['verse','chorus','verse','chorus']; context.ui.state.lines=4;
  return {...context.ui,$,requests,saved};
}
for(const root of roots) {
  test(`${root}: accessible target presets plus custom minutes/seconds`,()=>{
    const html=fs.readFileSync(path.join(root,'web/index.html'),'utf8');
    for(const id of ['targetLength','lengthCustom','lengthMinutes','lengthSeconds','lengthHelp']) assert.ok(html.includes(`id="${id}"`),id);
    assert.ok(/<label[^>]*for="targetLength"/.test(html));
    for(const value of ['auto','60','120','180','custom']) assert.ok(html.includes(`value="${value}"`),value);
  });
  test(`${root}: target parsing rejects invalid times and follows memory capacity`,()=>{
    const h=harness(root); assert.equal(typeof h.getLengthTarget,'function');
    assert.equal(h.getLengthTarget().seconds,null);
    h.$('targetLength').value='custom'; assert.equal(h.getLengthTarget().seconds,150);
    for(const [minutes,seconds] of [['2','60'],['-1','30'],['0','10'],['6','01'],['2.5','00']]) {
      h.$('lengthMinutes').value=minutes; h.$('lengthSeconds').value=seconds;
      assert.ok(h.getLengthTarget().error,`${minutes}:${seconds} must be rejected`);
    }
    h.$('memory').value='low'; assert.equal(h.currentLengthLimit(),180);
    h.$('memory').value='auto'; h.state.totalGpuGb=8; assert.equal(h.currentLengthLimit(),180);
    h.state.totalGpuGb=16; assert.equal(h.currentLengthLimit(),360);
  });
  test(`${root}: selecting a target never edits pasted lyrics and survives draft restore`,()=>{
    const h=harness(root); assert.equal(typeof h.renderLengthSummary,'function');
    const original='[verse]\nMy exact original words\nAnother original line'; h.$('lyrics').value=original;
    h.$('targetLength').value='custom'; h.renderLengthSummary();
    assert.equal(h.$('lyrics').value,original); assert.equal(h.$('shapeLines').disabled,true);
    assert.match(h.$('lengthHelp').textContent,/lyrics|words/i);
    h.saveDraft(); h.$('targetLength').value='auto'; h.$('lengthMinutes').value='0'; h.$('lengthSeconds').value='0'; h.loadDraft();
    assert.equal(h.$('targetLength').value,'custom'); assert.equal(h.$('lengthMinutes').value,'2'); assert.equal(h.$('lengthSeconds').value,'30');
    h.$('targetLength').value='auto'; h.renderLengthSummary(); assert.equal(h.$('shapeLines').disabled,false);
  });
  test(`${root}: write request forwards target and memory (unit interception, no backend)`,async()=>{
    const h=harness(root); assert.equal(typeof h.getLengthTarget,'function');
    h.$('idea').value='A test song'; h.$('targetLength').value='custom';
    await h.writeLyrics();
    const request=h.requests.find(r=>r.url==='/api/lyrics'); assert.ok(request);
    assert.equal(request.body.target_seconds,150); assert.equal(request.body.memory,'standard');
    assert.deepEqual(request.body.structure,['verse','chorus','verse','chorus']);
    h.requests.length=0; h.$('memory').value='low'; h.$('lengthMinutes').value='4'; await h.writeLyrics();
    assert.equal(h.requests.length,0,'over-limit target must not start a request');
  });
  test(`${root}: returning to Auto invalidates pending plans and clears busy status`,async()=>{
    let finish;
    const h=harness(root,()=>new Promise(resolve=>{finish=resolve}));
    h.$('targetLength').value='custom'; const pending=h.refreshLengthPlan();
    assert.equal(h.$('lengthHelp').attrs['aria-busy'],'true');
    h.$('targetLength').value='auto'; h.lengthUI.schedule();
    finish({ok:true,json:async()=>({counts:[10,9,10,9],total_lines:38,estimated_seconds:148.2,target_seconds:150,limit_seconds:360})});
    await pending;
    assert.equal(h.$('lengthHelp').attrs['aria-busy'],'false');
    assert.match(h.$('lengthHelp').textContent,/^Auto/);
    assert.equal(h.lengthUI.plan,null);
  });
}
