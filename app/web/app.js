/* OS Studio — open-source song models (YuE2, ACE-Step). Plain JS, no build step. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  };

  const IDEAS = [
    "a stubborn Roomba that thinks it's a horse",
    "Monday morning coffee",
    "my dog who steals socks",
    "leg day regrets",
  ];
  const ICON = {
    play: '<svg viewBox="0 0 24 24"><path d="M8 5.5v13l11-6.5z"/></svg>',
    bars: '<svg viewBox="0 0 24 24"><path d="M5 10h3v9H5zM10.5 5h3v14h-3zM16 12h3v7h-3z"/></svg>',
    reuse: '<svg viewBox="0 0 24 24"><path d="M4 4v6h6M5.5 15a7 7 0 1 0 .8-8.5L4 10"/></svg>',
    trash: '<svg viewBox="0 0 24 24"><path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12"/></svg>',
    plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
    mic: '<svg viewBox="0 0 24 24"><path d="M12 4a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0V7a3 3 0 0 0-3-3zM6 11a6 6 0 0 0 12 0M12 17v3"/></svg>',
    spark: '<svg viewBox="0 0 24 24"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/></svg>',
    spin: '<svg viewBox="0 0 24 24"><path d="M12 4a8 8 0 1 0 8 8"/></svg>',
    pen: '<svg viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4"/></svg>',
    close: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  };

  const state = {
    presets: [], styleLibrary: [], styleSets: {}, styleEngine: "yue2", voices: [], singers: [], singer: "", singerOwn: false, songs: [], writers: [], writer: "", defaultWriter: "",
    songModels: [], songModel: "auto", autoSongModel: "",
    sections: [], structure: [], lines: 4, secondsPerLine: 3.9, engine: "starting",
    job: null, handledJob: 0, logSeq: 0, startedJobs: new Set(), takes: 1, takeRun: null, instrumental: false, cover: null, coverFile: null,   // jobs this page started (only it autoplays them)
    current: null, version: "", lrc: null, undo: null,
  };
  const audio = $("audio");
  /* Guided composer: the step rules live in guide-core.js, the page only paints them. */
  const Guide = window.YSSGuide || null;
  const guide = { core: Guide ? new Guide() : null, ready: false, fresh: true, remix: null, made: null };
  const lengthUI = new window.YSSLengthUI({ state, $, api, clock: (seconds) => clock(seconds),
    shapeStats, saveDraft, setTimer: setTimeout, clearTimer: clearTimeout });
  function getLengthTarget() { return lengthUI.target(); }
  function currentLengthLimit() { return lengthUI.limit(); }
  function renderLengthSummary() { lengthUI.summary(); }
  function refreshLengthPlan() { return lengthUI.refresh(); }

  /* ------------------------------------------------------------ helpers */
  async function api(path, opts = {}) {
    const init = { method: opts.method || (opts.body !== undefined ? "POST" : "GET"), headers: {} };
    if (opts.raw !== undefined) {
      init.body = opts.raw;
      Object.assign(init.headers, opts.headers || {});
    } else if (opts.body !== undefined) {
      init.body = JSON.stringify(opts.body);
      init.headers["Content-Type"] = "application/json";
    }
    let resp;
    try {
      resp = await fetch(path, init);
    } catch (err) {
      throw new Error("OS Studio isn't responding. Is the black console window still open?");
    }
    if (!resp.ok) {
      let detail = resp.statusText;
      try { detail = (await resp.json()).detail || detail; } catch (_) { /* not json */ }
      const error = new Error(typeof detail === "string" ? detail : "Something went wrong.");
      error.status = resp.status;
      throw error;
    }
    return resp.json();
  }

  function toast(message, kind = "ok", action) {
    const node = el("div", "toast " + kind);
    node.append(el("span", null, message));
    if (action) {
      const btn = el("button", null, action.label);
      btn.onclick = () => { action.run(); node.remove(); };
      node.append(btn);
    }
    $("toasts").append(node);
    setTimeout(() => node.remove(), kind === "error" ? 12000 : 4500);
  }

  const clock = (sec) => {
    sec = Math.max(0, Math.floor(sec || 0));
    return Math.floor(sec / 60) + ":" + String(sec % 60).padStart(2, "0");
  };
  const when = (ts) => {
    const d = new Date(ts * 1000), now = new Date();
    const time = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    if (d.toDateString() === now.toDateString()) return "Today " + time;
    return d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + time;
  };
  const findStyle = (style) => [...state.presets, ...state.styleLibrary].find((p) => p.style === style) || null;
  const styleName = (style) => { const hit = findStyle(style); return hit ? hit.name : ""; };

  /* AI singer type: "" = whatever the style says, a preset id, or free text when singerOwn. */
  const singerValue = () => (state.singerOwn ? $("singerCustom").value.trim() : state.singer || "");
  const singerLabel = () => {
    const value = singerValue();
    const hit = state.singers.find((p) => p.id === value);
    return hit ? hit.name : value;
  };
  const singerSummary = () => {
    if (state.instrumental) return "No vocals (instrumental)";
    const kind = singerLabel();
    const who = kind ? `${kind} AI singer` : "AI singer";
    return state.voice ? `${state.voice} (your voice) · ${kind ? kind + " base" : "AI base"}` : who;
  };

  /* ------------------------------------------------------------ draft (autosave) */
  const FIELDS = ["idea", "lyrics", "style", "title", "seed", "cot", "memory", "targetLength", "lengthMinutes", "lengthSeconds"];
  function saveDraft() {
    const draft = { voice: state.voice || "", singer: state.singer || "", singerOwn: Boolean(state.singerOwn),
                    singerCustom: $("singerCustom").value, ownStyle: state.ownStyle || "", styleOwn: Boolean(state.styleOwn),
                    writer: state.writer || "", songModel: state.songModel || "auto",
                    structure: state.structure, lines: state.lines,
                    guide: guide.core ? guide.core.toJSON() : null, remix: guide.remix,
                    done: Boolean(state.draftDone), writtenKey: state.writtenKey ?? null,
                    instrumental: Boolean(state.instrumental), cover: state.cover || null };
    FIELDS.forEach((f) => { draft[f] = $(f).value; });
    localStorage.setItem("yss.draft", JSON.stringify(draft));
  }
  function loadDraft() {
    let draft = {};
    try { draft = JSON.parse(localStorage.getItem("yss.draft") || "{}"); } catch (_) { /* fresh */ }
    FIELDS.forEach((f) => { if (draft[f] != null && draft[f] !== "") $(f).value = draft[f]; });
    if (!["auto", "60", "120", "180", "custom"].includes($("targetLength").value)) $("targetLength").value = "auto";
    state.voice = draft.voice || "";
    state.instrumental = Boolean(draft.instrumental);
    state.cover = draft.cover && draft.cover.id ? draft.cover : null;
    state.singer = typeof draft.singer === "string" ? draft.singer : "";
    state.singerOwn = Boolean(draft.singerOwn);
    $("singerCustom").value = draft.singerCustom || "";
    state.writer = draft.writer || "";
    // ACE-Step is experimental and sounds worse than YuE2: each visit starts back on Auto (YuE2).
    state.songModel = draft.songModel && draft.songModel !== "ace15-xl" ? draft.songModel : "auto";
    state.structure = Array.isArray(draft.structure) ? draft.structure.slice(0, 16) : [];
    state.lines = [2, 4, 6, 8].includes(Number(draft.lines)) ? Number(draft.lines) : 0;
    state.ownStyle = draft.ownStyle || "";
    state.writtenKey = typeof draft.writtenKey === "string" ? draft.writtenKey : null;
    state.styleOwn = Boolean(draft.styleOwn) && Boolean($("style").value.trim());
    if (Guide) {
      guide.core = new Guide(draft.guide);
      guide.fresh = !draft.guide;
      guide.remix = typeof draft.remix === "string" && draft.remix ? draft.remix : null;
    }
  }

  /* Every start is a clean one: step 1, nothing typed. An unfinished song from last time is
     set aside (not deleted) and offered back by the "Continue that song" bar. Only the
     PC-specific picks (lyrics writer, song model, graphics memory) carry over. */
  function freshStart() {
    try {
      if (sessionStorage.getItem("yss.resume")) { sessionStorage.removeItem("yss.resume"); return; }
      const draft = JSON.parse(localStorage.getItem("yss.draft") || "null");
      if (!draft || typeof draft !== "object") return;
      const typed = String(draft.idea || "").trim() || String(draft.lyrics || "").trim();
      if (draft.done) localStorage.removeItem("yss.lastDraft");            // that song got made
      else if (typed) localStorage.setItem("yss.lastDraft", JSON.stringify(draft));
      localStorage.setItem("yss.draft", JSON.stringify({
        writer: draft.writer || "", songModel: draft.songModel || "auto", memory: draft.memory || "auto" }));
    } catch (_) { /* no storage: every start is fresh anyway */ }
  }

  function renderResume() {
    let last = null;
    try { last = JSON.parse(localStorage.getItem("yss.lastDraft") || "null"); } catch (_) { /* nothing saved */ }
    const words = last ? String(last.lyrics || "").split("\n").find((l) => l.trim() && !/^\s*\[.*\]\s*$/.test(l)) : "";
    const label = last ? (brief(last.title, 50) || brief(last.idea, 50) || brief(words, 50)) : "";
    $("resumeBar").hidden = !label;
    $("resumeText").textContent = label ? `Last time you were working on “${label}”.` : "";
  }

  function resumeLast() {
    try {
      const last = localStorage.getItem("yss.lastDraft");
      if (!last) { renderResume(); return; }
      localStorage.setItem("yss.draft", last);
      localStorage.removeItem("yss.lastDraft");
      sessionStorage.setItem("yss.resume", "1");
    } catch (_) { return; }
    window.location.reload();
  }

  /* ------------------------------------------------------------ guided steps */
  const STEP_FOCUS = ["idea", "lyrics", "style"];
  const brief = (text, max = 64) => {
    text = String(text || "").replace(/\s+/g, " ").trim();
    return text.length > max ? text.slice(0, max - 1).trimEnd() + "…" : text;
  };
  function guideFacts() {
    return { idea: Boolean($("idea").value.trim()), lyrics: Boolean($("lyrics").value.trim()),
             style: Boolean($("style").value.trim()), instrumental: Boolean(state.instrumental) };
  }
  function soundLabel() {
    const style = $("style").value.trim();
    return style ? (styleName(style) || "Your own sound") : "";
  }

  function renderGuide() {
    if (!guide.ready) return;
    const core = guide.core, facts = guideFacts();
    $("composer").dataset.guided = String(core.on);
    $("guideToggle").setAttribute("aria-pressed", String(core.on));
    $("guideToggle").textContent = core.on ? "Step by step: on" : "Step by step: off";
    const lines = state.lyricLines || 0;
    const words = lines ? `${lines} line${lines === 1 ? "" : "s"}` : "";
    const summaries = [
      facts.idea ? brief($("idea").value) : (state.instrumental ? "Instrumental" : (core.ownLyrics ? "Using my own lyrics" : "")),
      state.instrumental ? "Not sung" : words, soundLabel(), singerSummary(), guide.made ? "Done" : "Ready when you are",
    ];
    document.querySelectorAll(".step").forEach((card) => {
      const index = Number(card.dataset.step);
      const status = core.status(index, facts);
      card.dataset.state = status;
      const summary = card.querySelector(".step-summary");
      if (summary) summary.textContent = status === "locked" ? `Opens after step ${index}` : summaries[index];
    });
    $("recapLyrics").textContent = [brief($("title").value, 40), words].filter(Boolean).join(" · ") || "No lyrics yet";
    $("recapStyle").textContent = soundLabel() || "No sound picked yet";
    $("recapVoice").textContent = singerSummary();
    $("guideTitle").textContent = guide.remix ? "Remix a song" : "Let's make a song";
    $("guideSub").textContent = guide.remix
      ? "Everything from the original is loaded. Change what you like, then make it."
      : "Five quick steps. Finish one and the next opens.";
    $("remixBanner").hidden = !guide.remix;
    $("remixText").textContent = guide.remix ? `Remixing “${guide.remix}”. The original stays in your songs.` : "";
    $("made").hidden = !guide.made || busy();
    $("madeTitle").textContent = guide.made ? `“${guide.made.title}” is ready.` : "";
  }

  function showStep(index) {
    if (!guide.ready || !guide.core.on) return;
    const card = document.querySelector(`.step[data-step="${index}"]`);
    if (!card) return;
    const header = document.querySelector(".top");
    const top = index === 0 ? 0
      : card.getBoundingClientRect().top + window.scrollY - (header ? header.offsetHeight : 0) - 16;
    window.scrollTo({ top: Math.max(0, top), behavior: reducedMotion ? "auto" : "smooth" });
  }

  function goStep(index, scroll = true) {
    if (!guide.ready || !guide.core.go(index, guideFacts())) return false;
    saveDraft(); renderGuide();
    if (scroll) showStep(index);
    return true;
  }

  function guideNext() {
    const facts = guideFacts(), at = guide.core.current(facts);
    const missing = guide.core.advance(facts);
    if (missing) {
      toast(missing, "error");
      if (STEP_FOCUS[at]) $(STEP_FOCUS[at]).focus();
      return false;
    }
    saveDraft(); renderGuide(); showStep(guide.core.current(guideFacts()));
    return true;
  }

  function startFresh() {
    $("idea").value = ""; $("title").value = ""; $("seed").value = "";
    guide.remix = null; guide.made = null;
    state.instrumental = false; renderVocals();
    state.cover = null; renderCover();
    if (guide.core) guide.core.restart();
    setLyrics("");
    saveDraft(); renderGuide();
    setMode("create");
    window.scrollTo({ top: 0, behavior: reducedMotion ? "auto" : "smooth" });
    $("idea").focus();
  }

  async function ideaWrite() {
    const was = busy();
    await writeLyrics();
    if (!was && busy() && guide.ready && guide.core.current(guideFacts()) === 0) guideNext();
  }

  /* ------------------------------------------------------------ composer */
  function renderIdeas() {
    const box = $("ideaChips");
    IDEAS.forEach((idea) => {
      const chip = el("button", "chip", idea);
      chip.onclick = () => { $("idea").value = "a song about " + idea; $("idea").focus(); saveDraft(); };
      box.append(chip);
    });
  }

  function renderWriters() {
    const select = $("writer");
    select.replaceChildren();
    if (!state.writers.some((w) => w.id === state.writer)) state.writer = state.defaultWriter;
    state.writers.forEach((w) => {
      const label = `${w.name} — ${w.note} (${w.vram_gb} GB)` + (w.installed ? "" : " · download");
      const option = el("option", null, label);
      option.value = w.id;
      select.append(option);
    });
    select.value = state.writer;
    const picked = state.writers.find((w) => w.id === state.writer);
    $("writerHint").textContent = picked && !picked.installed
      ? `${picked.name} is not on this PC yet — the first use downloads ${picked.download_gb} GB.`
      : "";
  }

  function renderStyles() {
    const box = $("styleChips");
    box.replaceChildren();
    const current = $("style").value.trim();
    const own = state.styleOwn || Boolean(current && !styleName(current));
    state.presets.forEach((preset) => {
      const chip = el("button", "chip" + (preset.mine ? " mine" : "") + (!own && preset.style === current ? " on" : ""), preset.name);
      if (preset.mine) chip.title = "One of your saved styles";
      chip.onclick = () => {
        state.styleOwn = false;
        $("style").value = preset.style;
        renderStyles(); saveDraft();
      };
      box.append(chip);
    });
    const mine = el("button", "chip own" + (own ? " on" : ""));
    mine.innerHTML = ICON.pen;
    mine.append(el("span", null, "Type your own"));
    mine.onclick = () => {
      state.styleOwn = true;
      $("style").value = state.ownStyle || "";     // bring back what you typed last time
      renderStyles(); saveDraft();
      $("style").focus();
    };
    box.append(mine);
    const picked = findStyle(current);
    $("styleSelection").textContent = picked ? "Selected: " + picked.name + (picked.mine ? " (saved)" : "") : (own ? "Your custom sound" : "");
    $("btnDeleteStyle").hidden = !picked;
    $("btnDeleteStyle").textContent = picked && picked.mine ? "Delete from my styles" : "Delete this style";
    $("styleSaveRow").hidden = Boolean(picked) || !current;
    $("styleWarning").hidden = !/\bno vocals\b|^instrumental\b/i.test(current);
    $("style").classList.toggle("own", own);
    const ace = state.styleEngine === "ace15";
    $("style").placeholder = own
      ? (ace
        ? "Describe what you want to hear (ACE-Step: say what's there, not \"no …\"). Add a key and tempo if you like.\nExample: English, 90s boom bap, deep male rap vocal, dusty drums, vinyl crackle, 92 BPM, A minor"
        : "Describe the sound in your own words — language, genre, who is singing, instruments, tempo.\nExample: English, 90s boom bap, deep male vocals, dusty drums, vinyl crackle, 92 BPM")
      : "Describe the sound: language, genre, vocals, instruments, tempo";
    renderGuide();
  }

  function styleTyped() {
    const current = $("style").value.trim();
    state.styleOwn = !styleName(current);          // editing a preset makes it yours
    if (state.styleOwn) state.ownStyle = $("style").value;
    renderStyles();
  }

  function openStyleLibrary() {
    const select = $("styleCategory");
    select.replaceChildren();
    const all = el("option", null, "All styles"); all.value = ""; select.append(all);
    [...new Set(state.styleLibrary.map((p) => p.category))].forEach((category) => {
      const option = el("option", null, category); option.value = category; select.append(option);
    });
    $("styleSearch").value = "";
    renderStyleLibrary();
    $("styleDlg").showModal();
    $("styleSearch").focus();
  }

  function renderStyleLibrary() {
    const query = $("styleSearch").value.trim().toLowerCase();
    const category = $("styleCategory").value;
    const results = state.styleLibrary.filter((p) => (!category || p.category === category) &&
      `${p.name} ${p.style} ${p.category}`.toLowerCase().includes(query));
    const box = $("styleResults"); box.replaceChildren();
    $("styleCount").textContent = `${results.length} style${results.length === 1 ? "" : "s"}`;
    if (!results.length) box.append(el("p", "hint", "No matches. Try another word or choose All styles."));
    const hidden = (state.styleSets[state.styleEngine] || {}).hidden || 0;
    $("styleRestoreRow").hidden = !hidden;
    $("styleRestoreText").textContent = `You deleted ${hidden} built-in style${hidden === 1 ? "" : "s"}.`;
    results.forEach((preset) => {
      const item = el("div", "style-item");
      const btn = el("button", "style-result" + (preset.style === $("style").value.trim() ? " selected" : ""));
      btn.type = "button";
      btn.setAttribute("aria-pressed", String(preset.style === $("style").value.trim()));
      btn.append(el("strong", null, preset.name), el("span", "hint", preset.category),
        el("span", "style-description", preset.style));
      if (preset.instrumental) btn.append(el("span", "instrumental-label", "Instrumental prompt · asks for no vocals"));
      if (preset.mine) btn.append(el("span", "mine-label", "★ Saved by you" + (preset.engine ? ` · written for ${engineLabel(preset.engine)}` : "")));
      btn.onclick = () => {
        state.styleOwn = false; $("style").value = preset.style;
        renderStyles(); saveDraft(); $("styleDlg").close();
        $("btnBrowseStyles").focus();
      };
      const del = el("button", "style-del");
      del.type = "button"; del.innerHTML = ICON.trash;
      del.title = `Delete “${preset.name}”`;
      del.setAttribute("aria-label", del.title);
      del.onclick = () => deleteStyle(preset);
      item.append(btn, del);
      box.append(item);
    });
  }

  /* ------------------------------------------------------------ my styles */
  const engineLabel = (kind) => (kind === "ace15" ? "ACE-Step" : "YuE2");

  function setStyleSets(sets) {
    Object.entries(sets || {}).forEach(([kind, set]) => {
      state.styleSets[kind] = { presets: set.presets || [], library: set.style_library || [], hidden: set.hidden_count || 0 };
    });
    const set = state.styleSets[state.styleEngine] || state.styleSets.yue2;
    if (set) { state.presets = set.presets; state.styleLibrary = set.library; }
  }

  function afterStyleChange(sets) {
    setStyleSets(sets);
    const current = $("style").value.trim();
    state.styleOwn = Boolean(current) && !styleName(current);
    if (state.styleOwn) state.ownStyle = $("style").value;
    renderStyles(); saveDraft();
    if ($("styleDlg").open) renderStyleLibrary();
  }

  async function deleteStyle(preset) {
    try {
      const res = await api("/api/styles/delete", { body: { name: preset.name } });
      afterStyleChange(res.engine_styles);
      toast(res.outcome === "mine" ? `Deleted “${preset.name}” from your styles.`
        : `Deleted “${preset.name}”. Built-in styles can come back from Browse & manage styles.`, "ok", {
        label: "Undo", run: async () => {
          try {
            const back = res.outcome === "mine"
              ? await api("/api/styles", { body: { name: preset.name, style: preset.style, engine: preset.engine || state.styleEngine } })
              : await api("/api/styles/restore", { body: { name: preset.name } });
            afterStyleChange(back.engine_styles);
          } catch (err) { toast(err.message, "error"); }
        },
      });
    } catch (err) { toast(err.message, "error"); }
  }

  async function saveStyle() {
    const name = $("styleSaveName").value.trim();
    const style = $("style").value.trim();
    if (!name) { $("styleSaveName").focus(); toast("Give the style a name first.", "error"); return; }
    try {
      const res = await api("/api/styles", { body: { name, style, engine: state.styleEngine } });
      $("style").value = res.saved.style;
      $("styleSaveName").value = "";
      afterStyleChange(res.engine_styles);
      toast(`Saved “${res.saved.name}” to your styles — it's now one of the quick picks.`, "ok");
    } catch (err) { toast(err.message, "error"); }
  }

  async function restoreStyles() {
    try {
      const res = await api("/api/styles/restore", { body: {} });
      afterStyleChange(res.engine_styles);
      toast(`Brought back ${res.restored} style${res.restored === 1 ? "" : "s"}.`, "ok");
    } catch (err) { toast(err.message, "error"); }
  }

  async function makeStyleAI() {
    if (busy()) {                       // the button doubles as "Stop"
      if (state.job.kind === "style") api("/api/cancel", { method: "POST" }).catch(() => {});
      return;
    }
    const idea = $("styleIdea").value.trim();
    if (!idea) { $("styleIdea").focus(); toast("Describe the sound first, e.g. dreamy late-night R&B with a soft female singer.", "error"); return; }
    await startJob("/api/style-ai", { body: { idea, engine: state.styleEngine, writer: state.writer || "" } }, "style");
  }

  function previewVoice(voice) {
    const player = $("voiceAudio");
    if (state.previewVoice === voice.name && !player.paused) { player.pause(); return; }
    player.pause(); audio.pause(); $("recPlay").pause();
    state.previewVoice = voice.name;
    $("voicePreviewName").textContent = voice.name;
    $("voicePreview").hidden = false;
    player.src = "/api/voices/" + encodeURIComponent(voice.name) + "/preview?t=" + encodeURIComponent(voice.trained);
    player.play().catch(() => toast("Press play below to listen. If playback fails, this voice may be missing its saved recording.", "error"));
    renderVoices();
  }

  function closeVoicePreview() {
    $("voiceAudio").pause(); $("voiceAudio").removeAttribute("src"); $("voiceAudio").load();
    $("voicePreview").hidden = true; state.previewVoice = null;
    renderVoices();
  }

  function renderSingers() {
    const box = $("singerChips");
    box.replaceChildren();
    // Only offer the picker when the running server understands it (an old, un-restarted server would ignore it).
    $("singerType").hidden = !state.singers.length;
    const pick = (value, own) => {
      state.singer = value; state.singerOwn = own;
      renderSingers(); saveDraft();
      if (own) $("singerCustom").focus();
    };
    const any = el("button", "chip" + (!state.singerOwn && !state.singer ? " on" : ""), "Match the sound");
    any.type = "button";
    any.title = "Use whatever singer the style prompt describes";
    any.onclick = () => pick("", false);
    any.setAttribute("aria-pressed", String(!state.singerOwn && !state.singer));
    box.append(any);
    state.singers.forEach((preset) => {
      const chip = el("button", "chip" + (!state.singerOwn && state.singer === preset.id ? " on" : ""), preset.name);
      chip.type = "button";
      chip.title = `Sung by a ${preset.prompt}`;
      chip.onclick = () => pick(preset.id, false);
      chip.setAttribute("aria-pressed", String(!state.singerOwn && state.singer === preset.id));
      box.append(chip);
    });
    const own = el("button", "chip" + (state.singerOwn ? " on" : ""), "Describe your own");
    own.type = "button";
    own.onclick = () => pick("", true);
    own.setAttribute("aria-pressed", String(state.singerOwn));
    box.append(own);
    $("singerCustom").hidden = !state.singerOwn;
    const kind = singerLabel();
    $("singerHint").textContent = state.voice
      ? (kind ? `The ${kind.toLowerCase()} AI take is re-sung in your voice. Pick the type closest to your own voice for the best result.`
              : "Tip: pick the singer type closest to your own voice for the best re-sing.")
      : (kind ? `Any male/female words in the sound are swapped for: ${kind.toLowerCase()}.`
              : "Uses whatever singer the sound describes. Many sounds say \"male\", so pick Female to switch.");
    renderGuide();
  }

  function renderVoices() {
    const box = $("voiceChips");
    box.replaceChildren();
    if (state.voice && !state.voices.some((v) => v.name === state.voice)) state.voice = "";
    const ai = el("button", "chip" + (state.voice ? "" : " on"), "AI singer");
    ai.onclick = () => { state.voice = ""; renderVoices(); saveDraft(); };
    box.append(ai);
    state.voices.forEach((voice) => {
      const wrap = el("span", "voice-chip");
      const chip = el("button", "chip" + (state.voice === voice.name ? " on" : ""));
      chip.innerHTML = ICON.mic;
      chip.append(el("span", null, voice.name));
      chip.onclick = () => { state.voice = voice.name; renderVoices(); saveDraft(); };
      const del = el("button", "chip-x");
      del.type = "button";
      del.title = `Delete the voice "${voice.name}"`;
      del.setAttribute("aria-label", del.title);
      del.innerHTML = ICON.close;
      del.onclick = () => askDeleteVoice(voice.name);
      wrap.append(chip);
      const listen = el("button", "voice-listen");
      listen.type = "button";
      listen.disabled = !voice.has_preview;
      listen.title = voice.has_preview ? `Listen to ${voice.name}'s saved recording` : "No saved recording for this voice";
      listen.setAttribute("aria-label", listen.title);
      listen.innerHTML = ICON.play;
      listen.onclick = () => previewVoice(voice);
      wrap.append(listen, del);
      box.append(wrap);
    });
    const add = el("button", "chip add");
    add.innerHTML = ICON.plus;
    add.append(el("span", null, "New voice"));
    add.onclick = openVoiceDialog;
    box.append(add);
    const selected = state.voices.find((voice) => voice.name === state.voice);
    $("voiceHint").textContent = state.voice
      ? `The song is made with the AI singer first, then re-sung in "${state.voice}". You get both versions. Reference recording: ${selected.recording_type || "talking"}.`
      : (state.voices.length ? "" : "Add your own voice from a short recording of you talking, singing, or rapping.");
    $("voiceInvite").hidden = state.voices.length > 0;
    renderSingers();                               // its hint depends on the chosen voice; also repaints the guide
  }

  function lyricsMeta() {
    const text = $("lyrics").value;
    const lines = text.split("\n").filter((l) => l.trim() && !/^\s*(\d+x\s*)?\[.*\]\s*$/.test(l)).length;
    const sections = (text.match(/^\s*(\d+x\s*)?\[.*\]\s*$/gm) || []).length;
    $("lyricsMeta").textContent = lines ? `${lines} lines · ${sections} sections` : "";
    state.lyricLines = lines;
    const filled = Boolean(text.trim());
    $("lyricsEditHint").textContent = filled
      ? "Your lyrics are ready to edit. Change any line or paste a replacement; these are the words used to make your song."
      : "Write or paste your own lyrics here, or use Write lyrics below to start a draft.";
    renderWriteButtons();
    renderLengthSummary();
    renderGuide();
  }

  /* "Write lyrics" while the box is empty, "Update lyrics" once there are words. When the
     idea, shape or length changed since the lyrics were written, step 2's button lights up
     and says so: the change only takes effect when you press it again. */
  function lyricsSettingsKey() {
    return JSON.stringify([$("idea").value.trim(), state.structure, state.lines,
      $("targetLength").value, $("lengthMinutes").value, $("lengthSeconds").value]);
  }

  function renderWriteButtons() {
    const writing = busy() && state.job.kind === "lyrics";
    const filled = Boolean($("lyrics").value.trim());
    const label = filled ? "Update lyrics" : "Write lyrics";
    if (filled && state.writtenKey == null) state.writtenKey = lyricsSettingsKey();   // older drafts: no nag
    const stale = !writing && filled && state.writtenKey !== lyricsSettingsKey();
    const paint = `${writing}|${label}`;
    if (state.paintedWrite !== paint) {
      state.paintedWrite = paint;
      [$("btnLyrics"), $("btnLyrics2")].forEach((btn) => {
        btn.innerHTML = (writing ? ICON.spin : ICON.spark) + `<span>${writing ? "Stop writing" : label}</span>`;
      });
    }
    $("btnLyrics2").classList.toggle("primary", stale);
    $("btnLyrics2").title = filled ? "Write a new draft from your idea, shape and length (replaces these lyrics; Undo brings them back)" : "";
    $("lyricsWriteHint").textContent = stale
      ? "You changed the idea, shape or length since these lyrics were written. Press Update lyrics to write a new draft that follows them — Undo in Editing tools brings this one back."
      : (filled
        ? "Change the shape or length, then press Update lyrics for a new draft. It uses your idea from step 1 and replaces these lyrics; Undo in Editing tools brings them back."
        : "Pick a shape and length, then press Write lyrics. It uses your idea from step 1.");
    $("lyricsWriteHint").classList.toggle("stale", stale);
    $("shapeApplyHint").textContent = `This shape is used the next time you press ${label}. It does not rearrange existing lyrics.`;
  }

  function insertTag(tag) {
    const box = $("lyrics");
    const start = box.selectionStart, value = box.value;
    const before = value.slice(0, start), after = value.slice(box.selectionEnd);
    const lead = before && !before.endsWith("\n") ? "\n\n" : (before && !before.endsWith("\n\n") ? "\n" : "");
    const text = lead + tag + "\n";
    box.value = before + text + after;
    box.selectionStart = box.selectionEnd = before.length + text.length;
    box.focus();
    lyricsMeta(); saveDraft();
  }

  function setLyrics(text, remember = true) {
    const box = $("lyrics");
    if (remember && box.value.trim() && box.value !== text) {
      state.undo = box.value;
      $("btnUndo").hidden = false;
    }
    box.value = text;
    lyricsMeta(); saveDraft();
  }

  /* ------------------------------------------------------------ jobs */
  const busy = () => state.job && state.job.state === "running";

  const STYLE_AI_HINT = "Say the vibe in your own words. The lyrics writer (Qwen) turns it into a style prompt worded for the song model you picked.";

  function reflectBusy() {
    const running = busy();
    const kind = running ? state.job.kind : null;
    $("btnMake").disabled = running;
    const writeButtons = [$("btnLyrics"), $("btnLyrics2")];
    writeButtons.forEach((btn) => {
      btn.disabled = running && kind !== "lyrics";
      btn.classList.toggle("busy", kind === "lyrics");
    });
    renderWriteButtons();
    const aiBtn = $("btnStyleAI");
    aiBtn.disabled = running && kind !== "style";
    aiBtn.classList.toggle("busy", kind === "style");
    aiBtn.lastElementChild.textContent = kind === "style" ? "Stop" : "Make style";
    if (kind !== "style") $("styleAIHint").textContent = STYLE_AI_HINT;
    $("btnMake").hidden = kind === "song";
    $("progress").hidden = kind !== "song";
    if (kind !== "song") { $("btnCancel").disabled = false; $("btnCancel").textContent = "Cancel"; state.stepSignature = null; }
    $("voiceProgress").hidden = kind !== "voice";
    $("voiceTrain").disabled = running;
    $("btnSplit").disabled = running;
    $("coverGo").disabled = running;
    $("coverProgress").hidden = kind !== "cover";
    $("btnSplit").classList.toggle("busy", kind === "split");
    if (kind !== "split") $("btnSplit").lastElementChild.textContent = "Music & vocals";
    const pill = $("enginePill");
    if (state.engine === "error") {
      pill.dataset.state = "error"; pill.lastElementChild.textContent = "Engine failed to start";
    } else if (running) {
      pill.dataset.state = "busy";
      pill.lastElementChild.textContent = { song: "Making a song", lyrics: "Writing lyrics", voice: "Learning a voice", style: "Making a style", split: "Splitting a song", cover: "Listening to a song" }[kind];
    } else if (state.engine === "ready") {
      pill.dataset.state = "ready"; pill.lastElementChild.textContent = "Ready";
    } else {
      pill.dataset.state = "starting"; pill.lastElementChild.textContent = "Starting the engine…";
    }
    if (guide.ready) $("made").hidden = !guide.made || running;
  }

  function paintProgress(job) {
    const prog = job.progress || { overall: 0, label: "Starting…", steps: [] };
    const pct = Math.round(prog.overall * 100) + "%";
    if (job.kind === "song") {
      const run = state.takeRun;
      $("progLabel").textContent = (run && run.total > 1 ? `Take ${run.done + 1} of ${run.total} · ` : "") + (prog.label || "Working…");
      $("progPct").textContent = pct;
      $("progTime").textContent = clock(job.elapsed);
      $("progBar").style.width = pct;
      $("progNote").textContent = prog.note || "You can keep this window in the background.";
      $("progNote").classList.toggle("warn", Boolean(prog.note));
      const signature = prog.steps.map((s) => s.key + ":" + s.state).join("|");
      if (signature !== state.stepSignature) {   // rebuilding every poll would restart the spinner
        state.stepSignature = signature;
        $("progSteps").replaceChildren(...prog.steps.map((s) => el("li", s.state, s.label)));
      }
    } else if (job.kind === "voice") {
      $("vpLabel").textContent = prog.label || "Working…";
      $("vpPct").textContent = pct;
      $("vpTime").textContent = clock(job.elapsed);
      $("vpBar").style.width = pct;
    } else if (job.kind === "split") {
      $("btnSplit").lastElementChild.textContent = `Splitting… ${pct}`;
    } else if (job.kind === "cover") {
      $("cpLabel").textContent = prog.label || "Working…";
      $("cpPct").textContent = pct;
      $("cpTime").textContent = clock(job.elapsed);
      $("cpBar").style.width = pct;
    } else if (job.kind === "style") {
      $("styleAIHint").textContent = `${prog.label || "Starting"}… ${clock(job.elapsed)}`;
    } else if (job.kind === "lyrics") {
      if (job.partial) {
        $("lyrics").value = job.partial;
        $("lyrics").scrollTop = $("lyrics").scrollHeight;
        lyricsMeta();
      } else {
        $("lyricsMeta").textContent = `${prog.label || "Starting"}… ${clock(job.elapsed)}`;
      }
    }
  }

  async function finishJob(job) {
    if (job.id <= state.handledJob) return;
    state.handledJob = job.id;
    if (job.state !== "done" && job.kind === "song") { state.takeRun = null; renderTakes(); }
    if (job.state === "error") {
      toast(job.error || "That didn't work.", "error", { label: "Details", run: () => openDrawer(true) });
      if (job.kind === "voice") { $("voiceError").textContent = job.error; $("voiceError").hidden = false; }
      if (job.kind === "cover") { $("coverError").textContent = job.error; $("coverError").hidden = false; }
      if (job.kind === "lyrics" && state.lyricsBefore != null) setLyrics(state.lyricsBefore, false);
      return;
    }
    if (job.state === "cancelled") {
      toast("Cancelled.");
      if (job.kind === "lyrics" && state.lyricsBefore != null) setLyrics(state.lyricsBefore, false);
      return;
    }
    if (job.kind === "lyrics") {
      $("lyrics").value = state.lyricsBefore || "";
      state.writtenKey = state.writingKey || lyricsSettingsKey();
      setLyrics(job.result.lyrics);
      toast(guide.ready && guide.core.on ? "Lyrics are ready. Change anything you like, then press Next."
        : "Lyrics are ready — edit anything you like, then make the song.");
      if (state.writers.some((w) => !w.installed)) refreshModels();   // one may just have been downloaded
    } else if (job.kind === "style") {
      const made = job.result.style;
      $("style").value = made.style;
      state.styleOwn = true; state.ownStyle = made.style;
      $("styleSaveName").value = made.name;
      renderStyles(); saveDraft();
      toast(`Here's “${made.name}”. Edit it if you like — press Save to my styles to keep it.`, "ok");
      if (state.writers.some((w) => !w.installed)) refreshModels();
    } else if (job.kind === "song") {
      if (state.songModels.some((m) => !m.installed)) refreshModels();
      await refreshSongs();
      const song = state.songs.find((s) => s.id === job.result.song.id) || job.result.song;
      // Every open copy of the app sees the finished job; only the one that started it plays it.
      const mine = state.startedJobs.has(job.id);
      if (!state.current || (mine && audio.paused && state.mode !== "listen")) loadSong(song, mine);
      guide.made = { id: song.id, title: song.title };
      guide.remix = null;
      state.draftDone = true;
      saveDraft(); renderGuide();
      const run = mine ? state.takeRun : null;
      if (run) run.done += 1;
      const label = run && run.total > 1 ? `Take ${run.done} of ${run.total}, "${song.title}",` : `"${song.title}"`;
      if (song.cut_short) {
        toast(`${label} is ready, but it ran into the length limit and may end abruptly. Shorten the lyrics, or raise Graphics memory in More options.`, "error");
      } else toast(`${label} is ready.`, "ok", { label: "Listen", run: () => listenTo(song) });
      if (run && run.done < run.total) {
        if (!await startTake()) { state.takeRun = null; renderTakes(); }
      } else {
        if (run && run.total > 1) toast(`All ${run.total} takes are ready in your songs.`, "ok", { label: "Listen", run: () => listenTo(song) });
        state.takeRun = null; renderTakes();
      }
    } else if (job.kind === "cover") {
      if (state.startedJobs.has(job.id)) {
        $("coverDlg").close();
        applyCover(job.result.cover);
      }
    } else if (job.kind === "split") {
      await refreshSongs();
      const song = state.songs.find((s) => s.id === job.result.song.id);
      if (song && state.current && state.current.id === song.id) {
        state.current = song;
        renderVersions(song);
        $("btnSplit").hidden = true;
      }
      toast(`"${job.result.song.title}" now has Music only and Vocals only versions. Switch at the top of the player.`, "ok",
            song ? { label: "Play music only", run: () => { if (!state.current || state.current.id !== song.id) loadSong(song, true, { version: "inst" }); else { switchVersion("inst"); requestPlay(); } } } : undefined);
    } else if (job.kind === "voice") {
      const fresh = await api("/api/voices");
      state.voices = fresh.voices;
      state.voice = job.result.voice.name;
      renderVoices(); saveDraft();
      $("voiceDlg").close();
      if (state.previewVoice === state.voice) closeVoicePreview();
      toast(`Voice "${state.voice}" is ready and selected.`, "ok", {
        label: "Listen", run: () => {
          const voice = state.voices.find((v) => v.name === job.result.voice.name);
          if (voice && voice.has_preview) previewVoice(voice);
          else toast("This voice has no saved recording to preview.", "error");
        },
      });
    }
  }

  let pollTimer = null;
  async function poll() {
    clearTimeout(pollTimer);
    try {
      const data = await api("/api/job?since=" + state.logSeq);
      state.logSeq = data.log_seq;
      appendLog(data.log);
      const wasReady = state.engine === "ready";
      state.engine = data.engine.state;
      if (!wasReady && state.engine === "ready") refreshModels();   // the card is known now
      paintGpu(data.gpu);
      const job = data.job;
      state.job = job;
      if (job) {
        if (job.state === "running") paintProgress(job);
        else if (state.booted) await finishJob(job);
        else state.handledJob = job.id;   // finished before this page was opened
      }
      state.booted = true;
      reflectBusy();
    } catch (err) {
      $("enginePill").dataset.state = "error";
      $("enginePill").lastElementChild.textContent = "Not connected";
    }
    pollTimer = setTimeout(poll, busy() ? 700 : 3000);
  }

  function paintGpu(gpu) {
    const pill = $("gpuPill");
    if (!gpu || !gpu.available) { pill.hidden = true; return; }
    if (state.totalGpuGb !== gpu.total_gb) { state.totalGpuGb = gpu.total_gb; lengthUI.schedule(); }
    pill.hidden = false;
    $("gpuName").textContent = gpu.name.replace(/NVIDIA |GeForce /g, "");
    $("gpuText").textContent = `${gpu.used_gb.toFixed(1)} / ${Math.round(gpu.total_gb)} GB`;
    $("gpuBar").style.width = Math.min(100, (gpu.used_gb / gpu.total_gb) * 100) + "%";
  }

  async function startJob(path, opts, lock) {
    if (busy()) return false;
    try {
      const res = await api(path, opts);
      if (!res.job) return false;
      state.startedJobs.add(res.job);
      state.job = { id: res.job, kind: lock, state: "running", elapsed: 0, progress: null };
      reflectBusy();
      poll();
      return true;
    } catch (err) {
      toast(err.message, "error");
      return false;
    }
  }

  async function writeLyrics() {
    if (busy()) {                       // the button doubles as "Stop writing"
      if (state.job.kind === "lyrics") api("/api/cancel", { method: "POST" }).catch(() => {});
      return;
    }
    const idea = $("idea").value.trim();
    if (!idea) { goStep(0); $("idea").focus(); toast("Type what the song should be about first.", "error"); return; }
    const lengthError = lengthUI.validation();
    if (lengthError) { renderLengthSummary(); $("targetLength").focus(); toast(lengthError, "error"); return; }
    const target = getLengthTarget();
    if (target.seconds != null && !await refreshLengthPlan()) {
      toast(lengthUI.error || "Check the song length controls and try again.", "error"); return;
    }
    state.lyricsBefore = $("lyrics").value;
    state.writingKey = lyricsSettingsKey();
    if (!state.structure.some((name) => name !== "intro")) {
      toast("Add at least one section to the song structure in step 2.", "error");
      return;
    }
    await startJob("/api/lyrics", { body: {
      idea, writer: state.writer || "", structure: state.structure, lines: state.lines,
      target_seconds: target.seconds, memory: $("memory").value || "auto",
    } }, "lyrics");
  }

  async function makeSong() {
    const instrumental = Boolean(state.instrumental);
    const lyrics = instrumental ? "" : $("lyrics").value.trim();
    if (!lyrics && !instrumental) {
      toast("There are no lyrics yet — write some in step 2, or let step 1 write them.", "error");
      goStep(1); $("lyrics").focus();
      return;
    }
    const style = $("style").value.trim();
    if (!style) {
      toast(state.styleOwn ? "Describe your style in step 3, or pick one of the presets." : "Pick a style in step 3.", "error");
      goStep(2); $("style").focus();
      return;
    }
    const seedText = $("seed").value.trim();
    if (seedText && !/^\d{1,15}$/.test(seedText)) { toast("The seed has to be a whole number.", "error"); return; }
    // The whole run uses this snapshot, so editing the page between takes changes nothing.
    state.takeRun = { total: state.takes, done: 0, body: {
      lyrics, style, seed: seedText ? Number(seedText) : null, cot: $("cot").value,
      voice: state.voice || "", singer: singerValue(), idea: $("idea").value.trim(),
      title: $("title").value.trim() || (instrumental      // "late night rain" -> "Late Night Rain"
        ? brief($("idea").value, 60).replace(/…$/, "").replace(/(^|\s)(\S)/g, (m, sp, c) => sp + c.toUpperCase()) : ""),
      memory: $("memory").value || "auto", model: state.songModel || "auto", instrumental,
      cover_id: state.cover ? state.cover.id : "", cover_follow: state.cover ? state.cover.follow || "melody" : "melody",
    } };
    if (await startTake()) {
      $("progress").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else state.takeRun = null;
    renderTakes();
  }

  // Take 1 keeps a seed typed in More options; every later take gets a fresh one, so each sounds different.
  async function startTake() {
    const run = state.takeRun;
    const body = { ...run.body, seed: run.done === 0 ? run.body.seed : null };
    const started = await startJob("/api/song", { body }, "song");
    if (started) {
      guide.made = null; renderGuide();
      if (run.total > 1) $("progLabel").textContent = `Take ${run.done + 1} of ${run.total} · Starting…`;
    }
    return started;
  }

  function renderTakes() {
    const n = state.takes;
    document.querySelectorAll("#takesSeg [data-takes]").forEach((b) => {
      const on = Number(b.dataset.takes) === n;
      b.classList.toggle("on", on); b.setAttribute("aria-checked", String(on));
      b.disabled = Boolean(state.takeRun);
    });
    $("btnMake").lastElementChild.textContent = n > 1 ? `Make ${n} takes` : "Make the song";
    $("madeAgain").textContent = n > 1 ? `Make ${n} more takes` : "Make another take";
    $("takesHint").textContent = n > 1
      ? `${n} new performances of the same song, one after another (${n}× the time of one song). Each is saved as it finishes; Cancel stops the rest.`
      : "Each take is a new performance of the same song: same lyrics, style and singer.";
  }

  /* ------------------------------------------------------------ covers */
  function coverLyricsChoice() {
    const hit = document.querySelector('input[name="coverLyrics"]:checked');
    return hit ? hit.value : "transcribe";
  }
  function setCoverLyricsChoice(value) {
    document.querySelectorAll('input[name="coverLyrics"]').forEach((r) => { r.checked = r.value === value; });
  }

  async function openCoverDialog() {
    state.coverFile = null;
    $("coverFile").value = "";
    $("coverDrop").classList.remove("has");
    $("coverDropText").textContent = "Choose a song file, or drop it here";
    $("coverName").value = "";
    $("coverError").hidden = true;
    const pick = $("coverSong");
    pick.replaceChildren(el("option", null, "Pick a song…"));
    pick.firstElementChild.value = "";
    state.songs.filter((s) => !s.instrumental).forEach((s) => {
      const o = el("option", null, s.title); o.value = s.id; pick.append(o);
    });
    $("coverKeepRow").hidden = true;
    setCoverLyricsChoice("transcribe");
    $("coverDlg").showModal();
    try { renderCoverRecent((await api("/api/covers")).covers); } catch (_) { $("coverRecent").hidden = true; }
  }

  function renderCoverRecent(covers) {
    const list = $("coverRecentList");
    list.replaceChildren(...covers.slice(0, 12).map((c) => {
      const li = el("li");
      li.append(el("span", null, `${c.name} · ${c.key || "?"} · ${c.bpm || "?"} BPM`));
      const use = el("button", "btn small", "Use");
      use.type = "button";
      use.onclick = () => { $("coverDlg").close(); applyCover(c); };
      const del = el("button", "icon-btn small");
      del.type = "button"; del.innerHTML = ICON.close;
      del.title = `Forget the analysis of "${c.name}"`; del.setAttribute("aria-label", del.title);
      del.onclick = async () => {
        try { await api("/api/covers/" + c.id, { method: "DELETE" }); li.remove(); if (!list.children.length) $("coverRecent").hidden = true; }
        catch (err) { toast(err.message, "error"); }
      };
      li.append(use, del);
      return li;
    }));
    $("coverRecent").hidden = !covers.length;
  }

  function pickCoverFile(file) {
    if (!file) return;
    state.coverFile = file;
    $("coverSong").value = "";
    $("coverDrop").classList.add("has");
    $("coverDropText").textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
    if (!$("coverName").value.trim()) $("coverName").value = file.name.replace(/\.[^.]+$/, "").slice(0, 80);
    $("coverKeepRow").hidden = true;
    if (coverLyricsChoice() === "song") setCoverLyricsChoice("transcribe");
  }

  function pickCoverSong() {
    const song = state.songs.find((s) => s.id === $("coverSong").value);
    if (!song) { $("coverKeepRow").hidden = true; return; }
    state.coverFile = null;
    $("coverFile").value = "";
    $("coverDrop").classList.remove("has");
    $("coverDropText").textContent = "Choose a song file, or drop it here";
    $("coverName").value = song.title;
    $("coverKeepRow").hidden = !song.lyrics;
    setCoverLyricsChoice(song.lyrics ? "song" : "transcribe");
  }

  async function startCover() {
    const error = (msg) => { $("coverError").textContent = msg; $("coverError").hidden = false; };
    $("coverError").hidden = true;
    if (busy()) return error("Wait for the current job to finish first.");
    const lyrics = coverLyricsChoice(), songId = $("coverSong").value;
    let path, opts;
    if (state.coverFile) {
      const name = $("coverName").value.trim() || state.coverFile.name.replace(/\.[^.]+$/, "");
      path = `/api/covers?name=${encodeURIComponent(name)}&lyrics=${encodeURIComponent(lyrics === "song" ? "transcribe" : lyrics)}`;
      opts = { method: "POST", raw: state.coverFile,
               headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(state.coverFile.name) } };
    } else if (songId) {
      path = `/api/covers/song/${songId}?lyrics=${encodeURIComponent(lyrics)}`;
      opts = { body: {} };
    } else return error("Choose a song file or one of your songs first.");
    if (await startJob(path, opts, "cover")) $("coverProgress").hidden = false;
  }

  // The analysis is ready: its lyrics go to step 2, its sections become the song shape.
  function applyCover(c) {
    if (!c) return;
    state.cover = { id: c.id, name: c.name, key: c.key, bpm: c.bpm, sections: c.sections || [],
                    follow: (state.cover && state.cover.follow) || "melody" };
    if (c.lyrics) setLyrics(c.lyrics);
    const sung = (c.sections || []).filter((s) => s.sung && state.sections.includes(s.tag));
    if (sung.length) {
      state.structure = sung.map((s) => s.tag);
      const counts = sung.map((s) => s.lines).filter(Boolean).sort((a, b) => a - b);
      if (counts.length) state.lines = counts[Math.floor(counts.length / 2)];
      renderShape();
    }
    if (!$("title").value.trim()) $("title").value = `${c.name} (cover)`;
    if (state.instrumental) setInstrumental(false);
    renderCover(); saveDraft();
    if (guide.ready && guide.core) {
      guide.core.ownLyrics = true;
      guide.core.reached = Math.max(guide.core.reached, 1);
      goStep(1);
    }
    const note = c.lyrics_source === "transcribed" ? "Its lyrics were written down by ear, so check them in step 2."
      : c.lyrics_source === "song" ? "Its lyrics are in step 2." : "Step 2 has its sections: write new lyrics, or press Write lyrics.";
    toast(`Covering “${c.name}” (${c.key || "?"}, ${c.bpm || "?"} BPM). ${note} Then pick any sound and singer.`, "ok");
  }

  function renderCover() {
    const c = state.cover;
    $("coverBanner").hidden = !c;
    $("composer").dataset.cover = String(Boolean(c));
    if (!c) return;
    const bits = [c.key, c.bpm ? `${c.bpm} BPM` : "", c.sections && c.sections.length ? `${c.sections.length} sections` : ""].filter(Boolean);
    $("coverText").textContent = `Cover of “${c.name}”${bits.length ? " · " + bits.join(" · ") : ""}. The melody follows the original; the sound and singer are yours.`;
    document.querySelectorAll("#coverFollow [data-follow]").forEach((b) => {
      const on = b.dataset.follow === (c.follow || "melody");
      b.classList.toggle("on", on); b.setAttribute("aria-checked", String(on));
    });
  }

  async function refreshCoverInfo() {
    try {
      const found = (await api("/api/covers")).covers.find((c) => state.cover && c.id === state.cover.id);
      if (!state.cover) return;
      if (!found) { toast(`The analysis of “${state.cover.name}” was deleted, so this can't be a cover. Analyse the song again.`, "error"); state.cover = null; }
      else Object.assign(state.cover, { key: found.key, bpm: found.bpm, sections: found.sections });
      renderCover(); saveDraft();
    } catch (_) { /* the banner keeps what it has */ }
  }

  function renderVocals() {
    $("composer").dataset.instrumental = String(Boolean(state.instrumental));
    $("idea").placeholder = state.instrumental ? "optional: a title for it, e.g. Late Night Rain"
                                               : "a song about a fluffy cute white shih tzu";
    document.querySelectorAll("#vocalsSeg [data-vocals]").forEach((b) => {
      const on = (b.dataset.vocals === "off") === Boolean(state.instrumental);
      b.classList.toggle("on", on); b.setAttribute("aria-checked", String(on));
    });
  }

  // Instrumentals need no lyrics: open step 3 directly (steps 1-2 count as done).
  function skipToSound() {
    if (guide.ready && guide.core.on) {
      guide.core.reached = Math.max(guide.core.reached, 2);
      goStep(2);
    } else {
      $("stepStyle").scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
    }
  }

  function setInstrumental(on) {
    state.instrumental = Boolean(on);
    state.draftDone = false;
    renderVocals(); saveDraft(); renderGuide();
  }

  function setTakes(n) {
    state.takes = Math.min(4, Math.max(1, Number(n) || 1));
    try { localStorage.setItem("yss.takes", String(state.takes)); } catch (_) { /* per-session only */ }
    renderTakes();
  }

  /* ------------------------------------------------------------ library */
  async function refreshSongs() {
    const data = await api("/api/songs");
    state.songs = data.songs;
    queue.setSongs(state.songs.map((song) => song.id));
    $("modeListen").textContent = state.songs.length ? `Listen · ${state.songs.length}` : "Listen";
    renderSongs(); reflectPlayback();
  }

  /* Every song gets its own cover colours, worked out from its id (no image files). */
  function coverOf(song) {
    let hash = 0;
    for (const ch of song.id) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
    const hue = hash % 360;
    const letters = (song.title.match(/[\p{L}\p{N}]+/gu) || ["♪"]).slice(0, 2).map((w) => w[0].toUpperCase()).join("");
    return { c1: `hsl(${hue} 88% 64%)`, c2: `hsl(${(hue + 50) % 360} 72% 44%)`, text: letters };
  }
  function paintCover(node, song, label) {
    const cover = coverOf(song);
    node.style.setProperty("--c1", cover.c1);
    node.style.setProperty("--c2", cover.c2);
    if (label) label.textContent = cover.text;
  }

  function renderSongs() {
    const list = $("songs");
    list.replaceChildren();
    $("libCount").textContent = state.songs.length ? `${state.songs.length} song${state.songs.length === 1 ? "" : "s"}` : "";
    $("libEmpty").hidden = state.songs.length > 0;
    state.songs.forEach((song) => {
      const row = el("li", "song" + (state.current && state.current.id === song.id ? " current" : ""));
      const thumb = el("div", "thumb");
      paintCover(thumb, song, null);
      if (state.current && state.current.id === song.id && !audio.paused) thumb.innerHTML = ICON.bars;
      else thumb.textContent = coverOf(song).text;
      const info = el("div", "info");
      info.append(el("div", "name", song.title));
      const sub = el("div", "sub");
      const bits = [clock(song.duration), styleName(song.style), when(song.created)].filter(Boolean);
      sub.textContent = bits.join(" · ");
      if (song.voice) { sub.append(" · "); sub.append(el("span", "badge", song.voice)); }
      info.append(sub);
      const acts = el("div", "acts");
      if (song.lyrics) {
        const reuse = el("button", "remix", "Remix");
        reuse.type = "button";
        reuse.title = "Start a new song from this one";
        reuse.onclick = (ev) => { ev.stopPropagation(); openRemix(song); };
        acts.append(reuse);
      }
      const del = el("button"); del.title = "Delete"; del.innerHTML = ICON.trash;
      del.onclick = (ev) => {
        ev.stopPropagation();
        if (!del.classList.contains("danger")) {
          del.classList.add("danger"); del.textContent = "Delete?"; acts.classList.add("confirm");
          setTimeout(() => { del.classList.remove("danger"); del.innerHTML = ICON.trash; acts.classList.remove("confirm"); }, 3000);
          return;
        }
        deleteSong(song);
      };
      acts.append(del);
      row.append(thumb, info, acts);
      row.onclick = () => loadSong(song, true);
      row.tabIndex = 0;
      row.setAttribute("aria-label", "Play " + song.title);
      if (state.current && state.current.id === song.id) row.setAttribute("aria-current", "true");
      row.onkeydown = (event) => {
        if (event.target !== row || !["Enter", " "].includes(event.key)) return;
        event.preventDefault(); loadSong(song, true);
      };
      list.append(row);
    });
  }

  async function deleteSong(song) {
    try {
      await api("/api/songs/" + song.id, { method: "DELETE" });
      if (state.current && state.current.id === song.id) {
        audio.pause(); audio.removeAttribute("src"); state.current = null;
        $("player").dataset.empty = "true";
        state.songLoadId = (state.songLoadId || 0) + 1;
        updateMediaMetadata(); reflectPlayback();
      }
      await refreshSongs();
      toast(`Deleted "${song.title}".`);
    } catch (err) { toast(err.message, "error"); }
  }

  /* Remix = start a new song from an old one. Ask what should change, then open that step. */
  const REMIX = {
    take: { step: 4, note: "Loaded. Press Make the song for a fresh take." },
    lyrics: { step: 1, note: "Loaded. Edit the words, then press Next until you reach Make the song." },
    style: { step: 2, note: "Loaded. Pick a different sound, then press Next." },
    voice: { step: 3, note: "Loaded. Pick who sings it, then press Next." },
  };

  function openRemix(song) {
    if (!song || !song.lyrics) return;
    state.remixSong = song;
    $("remixHint").textContent = `Start a new song from “${song.title}”. The original stays in your songs.`;
    $("remixDlg").showModal();
  }

  function reuseSong(song, kind = "take") {
    if (!song) return;
    if (busy() && state.job.kind === "lyrics") { toast("Wait for the lyrics to finish writing first.", "error"); return; }
    const plan = REMIX[kind] || REMIX.take;
    if (song.lyrics) setLyrics(song.lyrics);
    if (song.style) {
      $("style").value = song.style;
      state.styleOwn = !styleName(song.style);
      if (state.styleOwn) state.ownStyle = song.style;
    }
    $("idea").value = song.idea || "";
    $("title").value = song.title || "";
    $("seed").value = "";                       // a remix is a new performance, never a copy
    if (song.cot) $("cot").value = song.cot;
    state.voice = song.voice && state.voices.some((v) => v.name === song.voice) ? song.voice : "";
    state.instrumental = Boolean(song.instrumental); renderVocals();
    state.cover = song.cover && song.cover.id ? { ...song.cover } : null; renderCover();
    if (state.cover) refreshCoverInfo();
    const sung = song.singer || "";
    state.singerOwn = Boolean(sung) && !state.singers.some((p) => p.id === sung);
    state.singer = state.singerOwn ? "" : sung;
    if (state.singerOwn) $("singerCustom").value = sung;
    guide.remix = song.title; guide.made = null;
    if (guide.core) guide.core.jump(plan.step, guideFacts());
    renderStyles(); renderVoices(); saveDraft();
    setMode("create");
    renderGuide();
    if (guide.ready && guide.core.on) showStep(guide.core.current(guideFacts()));
    else window.scrollTo({ top: 0, behavior: "smooth" });
    toast(plan.note);
  }

  function listenTo(song) {
    setMode("listen", true);
    const fresh = song && state.songs.find((entry) => entry.id === song.id);
    if (fresh && (!state.current || state.current.id !== fresh.id)) loadSong(fresh, true);
    else if (state.current && audio.paused) requestPlay();
  }

  /* ------------------------------------------------------------ workout playback */
  let playerPreferences = {};
  try {
    const saved = JSON.parse(localStorage.getItem("yss.player") || "{}");
    if (saved && typeof saved === "object" && !Array.isArray(saved)) playerPreferences = saved;
  } catch (_) { /* fresh or unavailable storage */ }
  const queue = new window.YSSPlaybackQueue(playerPreferences);
  const reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let lastPositionReport = 0;

  function savePlayerPreferences() {
    try {
      localStorage.setItem("yss.player", JSON.stringify({
        mode: state.mode || "create", continuous: queue.continuous,
        shuffle: queue.shuffle, repeat: queue.repeat,
      }));
    } catch (_) { /* playback must work even without persistent storage */ }
  }

  function setMode(mode, focus = false) {
    state.mode = mode === "listen" ? "listen" : "create";
    document.body.dataset.mode = state.mode;
    $("modeCreate").setAttribute("aria-pressed", String(state.mode === "create"));
    $("modeListen").setAttribute("aria-pressed", String(state.mode === "listen"));
    savePlayerPreferences();
    if (focus) {
      $("listeningStage").focus({ preventScroll: true });
      window.scrollTo({ top: 0, behavior: reducedMotion ? "auto" : "smooth" });
    }
    wave.dirty = true;
    reflectPlayback(); drawViz();
  }

  function configureAudioSession(type = "playback") {
    try { if (navigator.audioSession) navigator.audioSession.type = type; } catch (_) { /* optional API */ }
  }

  function mediaSession() {
    try { return navigator.mediaSession || null; } catch (_) { return null; }
  }

  function reportMediaPosition(force = false) {
    const session = mediaSession();
    if (!session) return;
    try { session.playbackState = state.current ? (audio.paused ? "paused" : "playing") : "none"; } catch (_) { /* unsupported state */ }
    const now = performance.now();
    if (!force && now - lastPositionReport < 1000) return;
    const duration = audio.duration;
    if (!session.setPositionState || !Number.isFinite(duration) || duration <= 0) return;
    lastPositionReport = now;
    try {
      session.setPositionState({ duration, playbackRate: audio.playbackRate || 1,
        position: Math.min(duration, Math.max(0, audio.currentTime || 0)) });
    } catch (_) { /* unsupported media position reporting */ }
  }

  function updateMediaMetadata() {
    const session = mediaSession();
    if (!session) return;
    try {
      session.metadata = state.current && window.MediaMetadata ? new MediaMetadata({
        title: state.current.title,
        artist: versionLabel(),
        album: "OS Studio",
        artwork: [{ src: new URL("/static/player-artwork.png?v=os-studio-1", window.location.href).href,
          sizes: "1024x1024", type: "image/png" }],
      }) : null;
    } catch (_) { /* media metadata is optional, audio is not */ }
    reportMediaPosition(true);
  }

  function reflectPlayback() {
    const song = state.current;
    const playing = Boolean(song && !audio.paused);
    [$("btnPlay"), $("miniPlay")].forEach((button) => {
      button.classList.toggle("playing", playing);
      button.setAttribute("aria-label", playing ? "Pause" : "Play");
      button.disabled = !song;
    });
    ["btnPrev", "btnNext", "miniPrev", "miniNext"].forEach((id) => { $(id).disabled = !song || !queue.order.length; });
    $("player").dataset.playing = String(playing);
    $("npState").textContent = playing ? "Now playing" : (song ? "Paused" : "Ready to play");
    $("btnPlayAll").disabled = !state.songs.length;
    $("btnContinuous").setAttribute("aria-pressed", String(queue.continuous));
    $("btnShuffle").setAttribute("aria-pressed", String(queue.shuffle));
    $("btnRepeat").textContent = "Repeat " + queue.repeat;
    $("btnRepeat").setAttribute("aria-label", "Repeat mode: " + queue.repeat);
    $("btnRepeat").setAttribute("aria-pressed", String(queue.repeat !== "off"));
    $("miniPlayer").hidden = !song || (state.mode === "listen" && state.transportInView !== false);
    $("miniTitle").textContent = song ? song.title : "Your song";
    $("miniSub").textContent = song
      ? versionLabel() + (playing ? " · Playing" : " · Paused") : "";
    const at = song ? queue.order.indexOf(song.id) : -1;
    $("queueMeta").textContent = queue.order.length
      ? (at >= 0 ? `${at + 1} of ${queue.order.length}` : `${queue.order.length} songs`) + " · " +
        (song ? (playing ? "Playing" : "Paused") : "Ready to play") : "Your rotation starts with your first song.";
    const next = state.songs.find((entry) => entry.id === queue.peek());
    $("upNext").textContent = !song ? "Tap Play all or choose a song."
      : !queue.continuous ? "Continuous play is off — this song will stop at the end."
      : queue.repeat === "one" ? "Up next: this song again"
      : queue.shuffle ? "Up next: a shuffled pick from your library"
      : next ? "Up next: " + next.title : "End of the rotation";
    reportMediaPosition(true);
  }

  function requestPlay() {
    if (!state.current) return;
    configureAudioSession();
    const attempt = audio.play();
    if (attempt && attempt.catch) attempt.catch((error) => {
      if (error && error.name === "AbortError") return; // a quick track/version change
      reflectPlayback();
      toast(error && error.name === "NotAllowedError"
        ? "Tap Play to start or resume audio on this device."
        : "Playback could not start. Check your connection, then tap Play.", "error");
    });
  }

  function togglePlayback() {
    if (!state.current) { playAll(); return; }
    audio.paused ? requestPlay() : audio.pause();
  }

  function playAll() {
    if (!state.songs.length) return;
    queue.setSongs(state.songs.map((song) => song.id));
    const first = state.songs.find((song) => song.id === queue.order[0]) || state.songs[0];
    loadSong(first, true);
  }

  function nextSong(manual = true) {
    const id = queue.next({ manual });
    const song = state.songs.find((entry) => entry.id === id);
    if (song) loadSong(song, true, { version: state.version, fromQueue: true });
    else { reflectPlayback(); drawViz(); }
  }

  function previousSong() {
    if (!state.current) return;
    if (audio.currentTime > 3) { audio.currentTime = 0; reportMediaPosition(true); return; }
    const id = queue.previous();
    const song = state.songs.find((entry) => entry.id === id);
    if (song) loadSong(song, true, { version: state.version, fromQueue: true });
    else { audio.currentTime = 0; requestPlay(); }
  }

  function seekBy(seconds) {
    if (!state.current || !Number.isFinite(audio.duration)) return;
    audio.currentTime = Math.min(audio.duration, Math.max(0, audio.currentTime + seconds));
    reportMediaPosition(true);
  }

  function updatePlaybackPosition() {
    if (!document.hidden) {
      const duration = Number.isFinite(audio.duration) ? audio.duration : (state.current && state.current.duration) || 0;
      if (duration && !state.seeking) {
        const percent = Math.min(100, Math.max(0, audio.currentTime / duration * 100));
        $("seek").value = Math.round(percent * 10);
        $("seek").style.setProperty("--p", percent + "%");
        $("miniBar").style.width = percent + "%";
      }
      if (!state.seeking) $("tCur").textContent = clock(audio.currentTime);
      syncLyrics();
    }
    reportMediaPosition();
  }

  function wirePlayback() {
    $("modeCreate").onclick = () => setMode("create");
    $("modeListen").onclick = () => setMode("listen");
    $("miniExpand").onclick = () => setMode("listen", true);
    [$("btnPlay"), $("miniPlay")].forEach((button) => { button.onclick = togglePlayback; });
    [$("btnNext"), $("miniNext")].forEach((button) => { button.onclick = () => nextSong(true); });
    [$("btnPrev"), $("miniPrev")].forEach((button) => { button.onclick = previousSong; });
    $("btnPlayAll").onclick = playAll;
    $("btnContinuous").onclick = () => { queue.continuous = !queue.continuous; savePlayerPreferences(); reflectPlayback(); };
    $("btnShuffle").onclick = () => { queue.setShuffle(!queue.shuffle); savePlayerPreferences(); reflectPlayback(); };
    $("btnRepeat").onclick = () => {
      const choices = ["off", "all", "one"];
      queue.setRepeat(choices[(choices.indexOf(queue.repeat) + 1) % choices.length]);
      savePlayerPreferences(); reflectPlayback();
    };
    const session = mediaSession();
    if (session) {
      const actions = {
        play: requestPlay, pause: () => audio.pause(),
        nexttrack: () => nextSong(true), previoustrack: previousSong,
        seekbackward: (event) => seekBy(-(event.seekOffset || 10)),
        seekforward: (event) => seekBy(event.seekOffset || 10),
        seekto: (event) => {
          if (!Number.isFinite(event.seekTime) || !Number.isFinite(audio.duration)) return;
          audio.currentTime = Math.min(audio.duration, Math.max(0, event.seekTime));
          reportMediaPosition(true);
        },
        stop: () => { audio.pause(); audio.currentTime = 0; reportMediaPosition(true); },
      };
      Object.entries(actions).forEach(([action, handler]) => {
        try { session.setActionHandler(action, handler); } catch (_) { /* per-action browser support varies */ }
      });
    }
    // One sound at a time across every open copy of the app (other tabs, Pinokio views, windows).
    const pageId = Math.random().toString(36).slice(2);
    let otherPages = null;
    try {
      otherPages = new BroadcastChannel("os-studio-playback");
      otherPages.onmessage = (event) => {
        if (event.data && event.data.playing && event.data.page !== pageId) {
          audio.pause(); $("voiceAudio").pause(); $("recPlay").pause();
        }
      };
    } catch (_) { /* older browser: each page still plays one sound at a time */ }
    const claimSpeakers = () => { try { otherPages && otherPages.postMessage({ playing: true, page: pageId }); } catch (_) {} };
    $("voiceAudio").addEventListener("play", claimSpeakers);
    $("recPlay").addEventListener("play", claimSpeakers);
    audio.addEventListener("play", () => {
      claimSpeakers();
      $("voiceAudio").pause(); $("recPlay").pause();
      reflectPlayback(); startTick(); renderSongs();
    });
    audio.addEventListener("pause", () => { reflectPlayback(); startTick(); renderSongs(); });   // bars settle, then the timer stops
    audio.addEventListener("ended", () => nextSong(false));
    audio.addEventListener("loadedmetadata", () => {
      if (Number.isFinite(audio.duration)) {
        $("tDur").textContent = clock(audio.duration);
        const pending = state.pendingSeek;
        state.pendingSeek = null;
        if (pending && state.current && pending.id === state.current.id && pending.version === state.version)
          audio.currentTime = Math.min(Math.max(0, audio.duration - 0.05), Math.max(0, pending.time));
      }
      reportMediaPosition(true);
    });
    audio.addEventListener("timeupdate", updatePlaybackPosition);
    audio.addEventListener("seeked", () => { updatePlaybackPosition(); reportMediaPosition(true); });
    audio.addEventListener("ratechange", () => reportMediaPosition(true));
    audio.addEventListener("durationchange", () => reportMediaPosition(true));
    audio.addEventListener("error", () => {
      if (audio.getAttribute("src")) {
        audio.pause(); reflectPlayback();
        toast("That audio file could not be played. Check your connection or choose another song.", "error");
      }
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) { updatePlaybackPosition(); wave.dirty = true; drawViz(); }
      // Never suspend/pause audio just because the display is hidden or locked.
    });
    state.transportInView = false;
    if (window.IntersectionObserver) {
      const observer = new IntersectionObserver((entries) => {
        state.transportInView = entries[0].isIntersecting && entries[0].intersectionRatio >= 0.5;
        reflectPlayback();
      }, { threshold: [0, 0.5, 1] });
      observer.observe(document.querySelector(".transport"));
    }
    configureAudioSession();
    setMode(playerPreferences.mode);
    reflectPlayback();
  }

  /* ------------------------------------------------------------ player */
  function parseLrc(text) {
    const lines = [];
    const stamp = (m, s) => Number(m) * 60 + Number(s);
    text.split(/\r?\n/).forEach((raw) => {
      const m = raw.match(/^\[(\d+):(\d+(?:\.\d+)?)\](.*)$/);
      if (!m) return;
      const t = stamp(m[1], m[2]);
      const parts = m[3].split(/<(\d+):(\d+(?:\.\d+)?)>/);
      const words = [];
      let end = null;
      if (parts.length > 1) {
        for (let i = 1; i < parts.length; i += 3) {
          const wt = stamp(parts[i], parts[i + 1]);
          const word = parts[i + 2] || "";
          if (word.trim()) words.push({ t: wt, text: word });
          else end = wt;
        }
      }
      const plain = (words.length ? words.map((w) => w.text).join("") : m[3]).trim();
      if (!plain) return;
      const section = plain.match(/^(?:\d+x\s*)?\[([^\]]+)\]$/);
      lines.push({ t, end, words, text: plain, section: section ? section[1] : null });
    });
    return lines;
  }

  function renderLyricsView(song) {
    const view = $("lyricsView");
    view.replaceChildren();
    state.lineNodes = [];
    state.activeLine = -1;
    if (state.lrc && state.lrc.length) {
      state.lrc.forEach((line, i) => {
        if (line.section) {
          view.append(el("div", "section", line.section));
          state.lineNodes.push(null);
          return;
        }
        const node = el("div", "line" + (line.words.length ? " words" : ""));
        if (line.words.length) {
          line.words.forEach((w) => { const span = el("span", "w", w.text); span.dataset.t = w.t; node.append(span); });
        } else node.textContent = line.text;
        node.onclick = () => { audio.currentTime = line.t + 0.01; if (audio.paused) requestPlay(); };
        view.append(node);
        state.lineNodes.push(node);
        void i;
      });
    } else {
      view.append(el("div", "plain", song.lyrics || "No timed lyrics were saved for this song."));
    }
    view.scrollTop = 0;
  }

  function syncLyrics() {
    if (!state.lrc || !state.lrc.length) return;
    const t = audio.currentTime;
    let idx = -1;
    for (let i = 0; i < state.lrc.length; i++) {
      if (state.lrc[i].section) continue;
      if (state.lrc[i].t <= t) idx = i; else break;
    }
    if (idx !== state.activeLine) {
      state.lineNodes.forEach((node, i) => {
        if (!node) return;
        node.classList.toggle("on", i === idx);
        node.classList.toggle("past", i < idx);
      });
      state.activeLine = idx;
      const node = idx >= 0 ? state.lineNodes[idx] : null;
      if (node) {
        const view = $("lyricsView");
        view.scrollTop = node.offsetTop - view.offsetTop - view.clientHeight / 2 + node.clientHeight / 2;
      }
    }
    const node = idx >= 0 ? state.lineNodes[idx] : null;
    if (node) node.querySelectorAll(".w").forEach((w) => w.classList.toggle("sung", Number(w.dataset.t) <= t));
  }

  function mediaUrl(song, version, download) {
    const params = new URLSearchParams();
    if (version) params.set("v", version);
    if (download) params.set("dl", "1");
    const q = params.toString();
    return "/media/" + song.id + (q ? "?" + q : "");
  }

  async function loadSong(song, autoplay, options = {}) {
    const loadId = state.songLoadId = (state.songLoadId || 0) + 1;
    state.current = song;
    state.version = songVersions(song).some((v) => v.v === options.version) ? options.version : "";
    queue.select(song.id);
    if (queue.shuffle && !options.fromQueue) queue.setShuffle(true);
    state.pendingSeek = null;
    $("player").dataset.empty = "false";
    $("npTitle").textContent = song.title;
    const bits = [song.cover ? `Cover of ${song.cover.name}` : "", styleName(song.style), song.instrumental ? "Instrumental" : song.voice ? "sung in " + song.voice : ((song.singer ? (state.singers.find((p) => p.id === song.singer) || { name: song.singer }).name + " " : "") + "AI singer"),
                  clock(song.duration), when(song.created)].filter(Boolean);
    $("npSub").textContent = bits.join(" · ");
    renderVersions(song);
    $("btnSplit").hidden = song.instrumental || (song.has_inst && song.has_vocals);
    $("dlMp3").href = mediaUrl(song, state.version, true);
    $("dlLrc").href = "/api/songs/" + song.id + "/lrc?dl=1";
    $("dlLrc").hidden = !song.has_lrc;
    $("btnReuse").hidden = !song.lyrics;
    audio.src = mediaUrl(song, state.version);
    paintCover($("npCover"), song, $("npCoverText"));
    paintCover($("miniCover"), song, $("miniCover"));
    loadViz(song, state.version);
    $("tCur").textContent = "0:00";
    $("tDur").textContent = clock(song.duration);
    state.lrc = null;
    renderLyricsView(song);
    updateMediaMetadata(); reflectPlayback(); renderSongs(); drawViz();
    // Keep play() in the original tap/ended handler; never wait for a lyrics fetch.
    if (autoplay) requestPlay();
    if (song.has_lrc) {
      try {
        const resp = await fetch("/api/songs/" + song.id + "/lrc");
        const text = resp.ok ? await resp.text() : null;
        if (loadId !== state.songLoadId) return;
        if (text != null) state.lrc = labelSections(parseLrc(text), song.lyrics);
      } catch (_) { /* audio continues even if timed lyrics are unavailable */ }
    }
    if (loadId !== state.songLoadId) return;
    renderLyricsView(song); syncLyrics();
  }

  /* The mixes a song has: itself, the AI singer before voice cloning, and the split-out halves. */
  function songVersions(song) {
    if (!song) return [];
    const list = [{ v: "", label: song.instrumental ? "Instrumental" : song.voice || (song.has_inst ? "Full song" : "AI singer") }];
    if (song.has_ai) list.push({ v: "ai", label: "AI singer" });
    if (song.has_inst) list.push({ v: "inst", label: "Music only" });
    if (song.has_vocals) list.push({ v: "vocals", label: "Vocals only" });
    return list;
  }
  const versionLabel = () => {
    const hit = songVersions(state.current).find((v) => v.v === state.version);
    return hit ? hit.label : "AI singer";
  };

  function renderVersions(song) {
    const seg = $("versionSeg"), list = songVersions(song);
    seg.replaceChildren(...list.map(({ v, label }) => {
      const b = el("button", v === state.version ? "on" : "", label);
      b.type = "button"; b.dataset.v = v;
      b.onclick = () => switchVersion(v);
      return b;
    }));
    seg.hidden = list.length < 2;
  }

  function switchVersion(version) {
    const song = state.current;
    if (!song || !songVersions(song).some((v) => v.v === version) || version === state.version) return;
    const at = audio.currentTime, playing = !audio.paused;
    state.version = version;
    state.pendingSeek = { id: song.id, version, time: at };
    $("versionSeg").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === version));
    $("dlMp3").href = mediaUrl(song, version, true);
    audio.src = mediaUrl(song, version);
    loadViz(song, version);
    updateMediaMetadata(); reflectPlayback();
    if (playing) requestPlay();
  }

  /* Visuals. Both pictures are drawn from numbers the server measured in the song file
     (/api/songs/<id>/viz), looked up at the current playback time. Nothing here touches Web
     Audio, so it behaves the same on a phone, in a browser and inside an embedded app window.
     A timer drives it, not animation frames: those can be frozen in embedded and background views. */
  const viz = { data: null, key: "", timer: 0, levels: [], settled: true };
  const wave = { dirty: true, mark: -1 };

  async function loadViz(song, version) {
    const key = song.id + "|" + version;
    viz.key = key; viz.data = null; wave.dirty = true;
    drawViz();
    try {
      const resp = await fetch("/api/songs/" + song.id + "/viz" + (version ? "?v=" + version : ""));
      if (!resp.ok) return;
      const data = await resp.json();
      if (viz.key !== key) return;                          // another song was picked meanwhile
      const raw = atob(data.spectrum || "");
      const spectrum = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) spectrum[i] = raw.charCodeAt(i);
      viz.data = { fps: data.fps, bands: data.bands, frames: Math.floor(spectrum.length / data.bands), spectrum, peaks: data.peaks || [] };
    } catch (_) { return; /* the slider still shows progress */ }
    wave.dirty = true;
    drawViz();
    if (!audio.paused) startTick();
  }

  function fitCanvas(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (width < 40 || height < 10) return null;             // not on screen
    const w = Math.round(width * ratio), h = Math.round(height * ratio);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; wave.dirty = true; }
    const g = canvas.getContext("2d");
    g.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { g, width, height };
  }

  function brandFill(g, width) {
    const grad = g.createLinearGradient(0, 0, width, 0);
    grad.addColorStop(0, "#febd49"); grad.addColorStop(1, "#9addd8");
    return grad;
  }

  function playbackFraction() {
    if (state.seeking) return Number($("seek").value) / 1000;
    const duration = Number.isFinite(audio.duration) ? audio.duration : (state.current && state.current.duration) || 0;
    return duration ? Math.min(1, Math.max(0, audio.currentTime / duration)) : 0;
  }

  /* Live bars: the measured loudness of each pitch range right now, bass in the middle. */
  function drawSpectrum() {
    const canvas = $("viz");
    const fit = fitCanvas(canvas);
    const data = viz.data, playing = !audio.paused && Boolean(state.current);
    canvas.dataset.mode = data ? "measured" : "idle";
    const bands = data ? data.bands : 24;
    if (viz.levels.length !== bands) viz.levels = new Array(bands).fill(0);
    let frame = 0, blend = 0;
    if (data && playing && data.frames) {
      const at = Math.min(data.frames - 1, Math.max(0, audio.currentTime * data.fps));
      frame = Math.floor(at); blend = at - frame;
    }
    let moving = false;
    for (let b = 0; b < bands; b++) {
      let target = 0;
      if (data && playing && data.frames) {
        const now = data.spectrum[frame * bands + b];
        const next = data.spectrum[Math.min(data.frames - 1, frame + 1) * bands + b];
        target = (now + (next - now) * blend) / 255;
      }
      const level = viz.levels[b];
      viz.levels[b] = reducedMotion ? target : level + (target - level) * (target > level ? 0.65 : 0.22);   // quick up, soft down
      if (Math.abs(viz.levels[b] - target) > 0.01) moving = true;
    }
    viz.settled = !moving;
    if (!fit) return;
    const { g, width, height } = fit;
    g.clearRect(0, 0, width, height);
    const count = bands * 2, gap = width < 420 ? 2 : 3;
    const bar = (width - gap * (count - 1)) / count;
    g.fillStyle = brandFill(g, width);
    g.globalAlpha = playing ? 0.95 : 0.4;
    for (let i = 0; i < count; i++) {
      const band = i < bands ? bands - 1 - i : i - bands;    // mirrored
      const h = Math.max(3, viz.levels[band] * (height - 4));
      g.beginPath();
      g.roundRect(i * (bar + gap), height - h, bar, h, Math.max(0, Math.min(bar / 2, 3)));
      g.fill();
    }
    g.globalAlpha = 1;
  }

  /* Whole-song outline that fills in as the song plays; tap or drag it to jump. */
  function drawWave() {
    const canvas = $("wave");
    const before = wave.dirty;
    const fit = fitCanvas(canvas);
    if (!fit) return;
    const { g, width, height } = fit;
    const peaks = viz.data && viz.data.peaks.length ? viz.data.peaks : null;
    const count = Math.max(30, Math.min(peaks ? peaks.length : 120, Math.floor(width / 4)));
    const mark = Math.round(playbackFraction() * count);
    if (!wave.dirty && !before && mark === wave.mark) return;   // nothing moved
    wave.dirty = false; wave.mark = mark;
    canvas.dataset.mode = peaks ? "measured" : "idle";
    g.clearRect(0, 0, width, height);
    const gap = 2, bar = (width - gap * (count - 1)) / count;
    const played = brandFill(g, width);
    for (let i = 0; i < count; i++) {
      let level = 0.1;
      if (peaks) {
        const from = Math.floor(i * peaks.length / count), to = Math.max(from + 1, Math.floor((i + 1) * peaks.length / count));
        level = 0;
        for (let k = from; k < to; k++) level = Math.max(level, peaks[k] / 255);
      }
      const h = Math.max(3, level * (height - 6));
      g.fillStyle = i < mark ? played : "rgba(255, 255, 255, .2)";
      g.beginPath();
      g.roundRect(i * (bar + gap), (height - h) / 2, bar, h, Math.max(0, Math.min(bar / 2, 2)));
      g.fill();
    }
  }

  function drawViz() { drawSpectrum(); drawWave(); }

  function stopTick() { clearInterval(viz.timer); viz.timer = 0; }
  function startTick() {
    stopTick();
    viz.timer = setInterval(tick, reducedMotion ? 250 : 33);
    tick();
  }
  function tick() {
    if (document.hidden) return;
    drawViz(); updatePlaybackPosition();
    if (audio.paused && viz.settled) stopTick();
  }

  /* ------------------------------------------------------------ song structure */
  const SHAPES = [
    { name: "Short", s: ["verse", "chorus", "verse", "chorus"] },
    { name: "Standard", s: ["verse", "chorus", "verse", "chorus", "bridge", "outro"] },
    { name: "Long", s: ["intro", "verse", "pre-chorus", "chorus", "verse", "pre-chorus", "chorus", "bridge", "chorus", "outro"] },
    { name: "Verses only", s: ["verse", "verse", "verse"] },
  ];
  const MAX_SECTIONS = 16;
  const capital = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  function shapeStats() {
    const per = Number(state.lines) || 4;
    let lines = 0;
    state.structure.forEach((s) => {
      if (s === "intro") return;                       // instrumental
      lines += s === "pre-chorus" ? Math.max(2, Math.floor(per / 2)) : per;
    });
    const seconds = lines * (state.secondsPerLine || 3.9) + (state.structure.includes("intro") ? 8 : 0);
    return { lines, seconds };
  }

  function renderShape() {
    const presets = $("shapePresets");
    presets.replaceChildren();
    const current = state.structure.join(",");
    SHAPES.forEach((shape) => {
      const chip = el("button", "chip" + (shape.s.join(",") === current ? " on" : ""), shape.name);
      if (shape.name === "Standard") chip.append(el("span", "shape-recommended", "Recommended"));
      chip.onclick = () => { state.structure = shape.s.slice(); renderShape(); saveDraft(); };
      presets.append(chip);
    });
    if (!SHAPES.some((shape) => shape.s.join(",") === current)) presets.append(el("span", "chip on", "Your own"));

    $("shapeOverview").textContent = state.structure.length
      ? state.structure.map(capital).join(" → ")
      : "No sections yet — choose a shape or customize below.";
    renderWriteButtons();

    const seq = $("shapeSeq");
    seq.replaceChildren();
    const totals = {}, seen = {};
    state.structure.forEach((s) => { totals[s] = (totals[s] || 0) + 1; });
    state.structure.forEach((s, i) => {
      seen[s] = (seen[s] || 0) + 1;
      const pill = el("span", "sec");
      pill.dataset.s = s;
      pill.draggable = true;
      pill.append(el("span", null, capital(s)));
      if (totals[s] > 1 && s !== "chorus") pill.append(el("span", "n", String(seen[s])));
      const remove = el("button");
      remove.type = "button";
      remove.title = "Remove this " + s;
      remove.setAttribute("aria-label", remove.title);
      remove.innerHTML = ICON.close;
      remove.onclick = () => { state.structure.splice(i, 1); renderShape(); saveDraft(); };
      pill.append(remove);
      pill.addEventListener("dragstart", (ev) => {
        state.dragFrom = i; pill.classList.add("dragging");
        ev.dataTransfer.effectAllowed = "move"; ev.dataTransfer.setData("text/plain", String(i));
      });
      pill.addEventListener("dragend", () => { state.dragFrom = null; renderShape(); });
      pill.addEventListener("dragover", (ev) => { ev.preventDefault(); pill.classList.add("over"); });
      pill.addEventListener("dragleave", () => pill.classList.remove("over"));
      pill.addEventListener("drop", (ev) => {
        ev.preventDefault();
        const from = state.dragFrom;
        if (from == null || from === i) return;
        const [moved] = state.structure.splice(from, 1);
        state.structure.splice(from < i ? i - 1 : i, 0, moved);
        state.dragFrom = null;
        renderShape(); saveDraft();
      });
      seq.append(pill);
    });
    if (!state.structure.length) seq.append(el("span", "empty", "No sections yet — add some below, or pick a length above."));

    const add = $("shapeAdd");
    add.replaceChildren();
    const full = state.structure.length >= MAX_SECTIONS;
    (state.sections || []).forEach((s) => {
      const btn = el("button", "tag", "+ " + s);
      btn.type = "button";
      btn.disabled = full || (s === "intro" && state.structure.includes("intro"));
      btn.onclick = () => {
        if (s === "intro") state.structure.unshift(s); else state.structure.push(s);
        renderShape(); saveDraft();
      };
      add.append(btn);
    });

    $("shapeLines").value = String(state.lines);
    const { lines, seconds } = shapeStats();
    $("shapeMeta").textContent = lines ? `about ${clock(seconds)} · ${lines} lines` : "";
    $("shapeHint").textContent = seconds > 330
      ? "That is a long song — it stops at 6 minutes (3 on the Low memory setting)."
      : (full ? `That is the most sections a song can have (${MAX_SECTIONS}).`
              : "Drag a section to move it. The lyrics writer follows this order.");
    $("shapeHint").classList.toggle("warn", seconds > 330);
    lengthUI.schedule();
  }

  /* Section headings for the player. The timed-lyrics file only carries the
     sung lines, so the headings come from the song's own lyrics. */
  const SECTION_RE = /^\s*(?:\d+\s*[x×]\s*)?\[\s*([^\]]+?)\s*\]\s*$/i;
  const lineKey = (s) => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "");

  function labelSections(lrc, lyrics) {
    const sung = lrc.filter((line) => !line.section);
    const source = [];
    let section = null, block = -1;
    (lyrics || "").split(/\r?\n/).forEach((raw) => {
      const tag = raw.match(SECTION_RE);
      if (tag) { section = tag[1].toLowerCase().replace(/\s*\d+$/, ""); block += 1; return; }
      const key = lineKey(raw);
      if (!key) return;
      if (block < 0) { block = 0; section = "verse"; }
      source.push({ key, section, block });
    });
    if (!source.length || block < 1) return lrc;        // nothing to label with

    const out = [];
    let next = 0, lastBlock = -1, matched = 0;
    sung.forEach((line) => {
      const key = lineKey(line.text);
      let hit = -1;
      for (let k = next; k < Math.min(source.length, next + 6) && key; k++) {
        const other = source[k].key;
        if (other === key || (key.length > 5 && (other.startsWith(key) || key.startsWith(other)))) { hit = k; break; }
      }
      if (hit >= 0) {
        matched += 1;
        if (source[hit].block !== lastBlock) {
          lastBlock = source[hit].block;
          out.push({ t: line.t, end: null, words: [], text: `[${source[hit].section}]`, section: source[hit].section });
        }
        next = hit + 1;
      }
      out.push(line);
    });
    return matched >= sung.length * 0.5 ? out : lrc;
  }

  /* ------------------------------------------------------------ record a voice */
  const rec = { stream: null, recorder: null, chunks: [], ctx: null, started: 0, timer: 0, on: false, keep: true };
  const REC_MIN = 10, REC_MAX = 180;

  function voiceError(msg) {
    $("voiceError").textContent = msg;
    $("voiceError").hidden = !msg;
  }

  function recordingAction() {
    return { talking: "Talk normally", singing: "Sing naturally", rapping: "Rap naturally" }[$("voiceRecordingType").value] || "Talk normally";
  }

  function updateRecordingGuidance() {
    const type = $("voiceRecordingType").value || "talking";
    $("voiceClipHint").textContent = `Use a recording of one person ${type} — 30 seconds to 3 minutes, no music in the background. Training takes a few minutes and only happens once per voice.`;
    $("recScript").hidden = !rec.on || type !== "talking";
    if (rec.on) {
      $("recHint").textContent = `Recording… ${recordingAction().toLowerCase()}. Stop any time after 30 seconds.`;
    } else if (!$("recPlay").hidden) {
      $("recHint").textContent = `${capital(type)} recording ready. Listen back, then name the voice below. 30 seconds or more gives a closer match.`;
    } else {
      $("recHint").textContent = `Uses your microphone. ${recordingAction()} for 30 seconds or more.`;
    }
  }

  function recReset() {
    if (rec.on) { rec.keep = false; stopRecording(); }
    $("recBtn").classList.remove("on");
    $("recBtn").lastElementChild.textContent = "Record my voice";
    $("recTime").hidden = $("recMeter").hidden = $("recScript").hidden = true;
    $("recPlay").hidden = true;
    $("recPlay").removeAttribute("src");
    updateRecordingGuidance();
  }

  async function startRecording() {
    audio.pause(); $("voiceAudio").pause(); $("recPlay").pause();
    configureAudioSession("play-and-record");
    voiceError("");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
      return voiceError("This window cannot use the microphone. Record with the Windows Sound Recorder and choose the file instead.");
    }
    try {
      rec.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: true, autoGainControl: true },
      });
    } catch (err) {
      const name = err && err.name;
      if (name === "NotAllowedError" || name === "SecurityError") {
        return voiceError("The microphone is blocked. Click the lock or microphone icon next to the address, allow the microphone, then try again. Also check Windows Settings > Privacy > Microphone.");
      }
      if (name === "NotFoundError" || name === "OverconstrainedError") {
        return voiceError("No microphone was found. Plug one in, or choose a recording file instead.");
      }
      return voiceError("The microphone could not be started: " + (err && err.message ? err.message : name));
    }
    rec.chunks = [];
    rec.keep = true;
    rec.recorder = new MediaRecorder(rec.stream);
    rec.recorder.ondataavailable = (ev) => { if (ev.data && ev.data.size) rec.chunks.push(ev.data); };
    rec.recorder.onstop = finishRecording;
    rec.recorder.start(500);
    rec.on = true;
    rec.started = performance.now();

    rec.ctx = new AudioContext();
    rec.ctx.resume().catch(() => {});
    const analyser = rec.ctx.createAnalyser();
    analyser.fftSize = 1024;
    rec.ctx.createMediaStreamSource(rec.stream).connect(analyser);
    const mute = rec.ctx.createGain();                    // keeps the meter running without playing the mic back
    mute.gain.value = 0;
    analyser.connect(mute).connect(rec.ctx.destination);
    const buf = new Float32Array(analyser.fftSize);
    const bar = $("recMeter").firstElementChild;
    const level = () => {
      analyser.getFloatTimeDomainData(buf);
      let peak = 0;
      for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i]));
      bar.style.width = Math.min(100, Math.round(peak * 140)) + "%";
    };

    $("recBtn").classList.add("on");
    $("recBtn").lastElementChild.textContent = "Stop recording";
    $("recTime").hidden = $("recMeter").hidden = false;
    $("recPlay").hidden = true;
    updateRecordingGuidance();
    const tick = () => {
      const secs = (performance.now() - rec.started) / 1000;
      $("recTime").textContent = clock(secs);
      level();
      if (secs >= REC_MAX) stopRecording();
    };
    tick();
    rec.timer = setInterval(tick, 100);                   // a timer, not animation frames: keeps going in a background window
  }

  function stopRecording() {
    if (!rec.on) return;
    rec.on = false;
    clearInterval(rec.timer);
    try { if (rec.recorder && rec.recorder.state !== "inactive") rec.recorder.stop(); } catch (_) { /* already stopped */ }
    if (rec.stream) rec.stream.getTracks().forEach((track) => track.stop());
    if (rec.ctx) rec.ctx.close().catch(() => {});
    configureAudioSession();
    $("recBtn").classList.remove("on");
    $("recBtn").lastElementChild.textContent = "Record again";
    $("recTime").hidden = $("recMeter").hidden = $("recScript").hidden = true;
  }

  function encodeWav(samples, rate) {
    const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
    const text = (at, s) => { for (let i = 0; i < s.length; i++) view.setUint8(at + i, s.charCodeAt(i)); };
    text(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE");
    text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    text(36, "data"); view.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
      const v = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
    }
    return view.buffer;
  }

  async function finishRecording() {
    if (!rec.keep) return;                                // dialog was closed mid-recording
    const blob = new Blob(rec.chunks, { type: rec.recorder.mimeType || "audio/webm" });
    let decoded;
    try {
      const ctx = new AudioContext();
      decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
      ctx.close().catch(() => {});
    } catch (err) {
      updateRecordingGuidance();
      return voiceError("That recording could not be read. Please try again.");
    }
    const mono = new Float32Array(decoded.length);        // mix down to one channel
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const data = decoded.getChannelData(c);
      for (let i = 0; i < data.length; i++) mono[i] += data[i] / decoded.numberOfChannels;
    }
    let peak = 0;
    for (let i = 0; i < mono.length; i++) peak = Math.max(peak, Math.abs(mono[i]));
    const secs = decoded.duration;
    if (secs < REC_MIN) {
      updateRecordingGuidance();
      return voiceError(`That was only ${Math.round(secs)} seconds. Record at least ${REC_MIN} — 30 or more works best.`);
    }
    if (peak < 0.01) {
      updateRecordingGuidance();
      return voiceError("The recording is silent. Check that the right microphone is selected in Windows and that it is not muted.");
    }
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");
    const file = new File([encodeWav(mono, decoded.sampleRate)], `recording_${stamp}.wav`, { type: "audio/wav" });
    pickClip(file, true);
    $("recPlay").src = URL.createObjectURL(file);
    $("recPlay").hidden = false;
    $("recHint").textContent = `${capital($("voiceRecordingType").value || "talking")} recording · ${clock(secs)}` + (secs < 30 ? " — it will work, but 30 seconds or more gives a closer match." : ". Listen back, then name the voice below.");
  }

  /* ------------------------------------------------------------ delete a voice */
  function askDeleteVoice(name) {
    state.voiceToDelete = name;
    $("delVoiceError").hidden = true;
    $("delVoiceOk").disabled = false;
    $("delVoiceText").textContent =
      `"${name}" will be removed from this app. Songs you already made with it stay in your library. ` +
      "The voice goes to the Recycle Bin, so you can still restore it from there.";
    $("delVoiceDlg").showModal();
  }

  async function deleteVoice() {
    const name = state.voiceToDelete;
    if (!name) return;
    $("delVoiceOk").disabled = true;
    try {
      if (state.previewVoice === name) closeVoicePreview();
      const data = await api("/api/voices/" + encodeURIComponent(name), { method: "DELETE" });
      state.voices = data.voices;
      if (state.voice === name) state.voice = "";
      renderVoices(); saveDraft();
      $("delVoiceDlg").close();
      toast(data.how === "recycled" ? `"${name}" was moved to the Recycle Bin.` : `"${name}" was deleted.`);
    } catch (err) {
      $("delVoiceError").textContent = err.message;
      $("delVoiceError").hidden = false;
      $("delVoiceOk").disabled = false;
    }
  }

  function refreshModels() {
    return api("/api/state").then((data) => {
      state.writers = data.writers || state.writers;
      state.songModels = data.song_models || state.songModels;
      state.autoSongModel = data.auto_song_model || state.autoSongModel;
      renderWriters(); renderSongModels();
    }).catch(() => {});
  }

  /* ------------------------------------------------------------ song model */
  /* Each engine has its own style prompts (YuE2 and ACE-Step want different wording).
     Switching engines swaps the preset chips and the style library; a picked preset
     follows the switch by name, a style you typed yourself is left alone. */
  function engineKind() {
    const id = state.songModel === "auto" ? state.autoSongModel : state.songModel;
    const m = state.songModels.find((x) => x.id === id);
    return m && state.styleSets[m.kind] ? m.kind : "yue2";
  }

  function applyEngineStyles() {
    const want = engineKind();
    if (!state.styleSets.yue2 || want === state.styleEngine) return;
    const current = $("style").value.trim();
    const hit = Object.values(state.styleSets).flatMap((set) => [...set.presets, ...set.library])
      .find((p) => p.style === current);
    const set = state.styleSets[want];
    state.presets = set.presets; state.styleLibrary = set.library; state.styleEngine = want;
    const match = hit && [...set.presets, ...set.library].find((p) => p.name === hit.name);
    if (match) state.styleOwn = false;
    if (match && match.style !== current) {
      $("style").value = match.style;
      toast(`Style switched to the ${want === "ace15" ? "ACE-Step" : "YuE2"} version of “${match.name}”.`);
    }
    renderStyles(); saveDraft();
  }

  function renderSongModels() {
    const select = $("songModel");
    select.replaceChildren();
    const auto = state.songModels.find((m) => m.id === state.autoSongModel);
    const first = el("option", null, "Auto — picks for your graphics card" + (auto ? ` (${auto.name})` : ""));
    first.value = "auto";
    select.append(first);
    state.songModels.forEach((m) => {
      const option = el("option", null, `${m.name} — ${m.note} (${m.vram_gb} GB)` + (m.installed ? "" : " · download"));
      option.value = m.id;
      select.append(option);
    });
    if (state.songModel !== "auto" && !state.songModels.some((m) => m.id === state.songModel)) state.songModel = "auto";
    select.value = state.songModel;
    applyEngineStyles();
    const picked = state.songModels.find((m) => m.id === state.songModel);
    const download = picked && !picked.installed
      ? `${picked.name} is not on this PC yet — the first song downloads it (${picked.download_gb || 4} GB).` : "";
    $("songModelHint").textContent = picked && picked.kind === "ace15"
      ? "Experimental: ACE-Step songs don't come out as good as YuE2 — vocals sound lower quality. " +
        "Use it to try things out; for your best songs, stay on Auto (YuE2). Melody planning doesn't apply, " +
        "and the song length comes from your lyrics." + (download ? " " + download : "")
      : download;
  }

  /* ------------------------------------------------------------ voice dialog */
  function openVoiceDialog() {
    $("voiceError").hidden = true;
    $("voiceName").value = "";
    $("voiceRecordingType").value = "talking";
    $("clip").value = "";
    $("drop").classList.remove("has");
    $("dropText").textContent = "Choose a recording, or drop it here";
    state.clip = null;
    recReset();
    $("voiceDlg").showModal();
  }

  function pickClip(file, recorded) {
    if (!file) return;
    state.clip = file;
    voiceError("");
    $("drop").classList.toggle("has", !recorded);
    $("dropText").textContent = recorded ? "Choose a recording, or drop it here"
      : `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
    if (recorded) { $("voiceName").focus(); return; }
    recReset();                                           // a chosen file replaces a recording
    if (!$("voiceName").value.trim()) {
      $("voiceName").value = file.name.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9 _-]+/g, "_").slice(0, 40);
    }
  }

  async function trainVoice(replace) {
    const name = $("voiceName").value.trim();
    const showError = (msg) => { $("voiceError").textContent = msg; $("voiceError").hidden = false; };
    $("voiceError").hidden = true;
    if (rec.on) return showError("Stop the recording first.");
    if (!state.clip) return showError("Record your voice or choose a recording first.");
    if (!name) return showError("Give the voice a name.");
    if (busy()) return showError("Wait for the current job to finish first.");
    try {
      const res = await api(`/api/voices?name=${encodeURIComponent(name)}&replace=${replace ? 1 : 0}&recording_type=${encodeURIComponent($("voiceRecordingType").value)}`, {
        method: "POST", raw: state.clip,
        headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(state.clip.name) },
      });
      state.job = { id: res.job, kind: "voice", state: "running", elapsed: 0, progress: null };
      reflectBusy();
      poll();
    } catch (err) {
      if (err.status === 409 && /already exists/.test(err.message)) {
        showError(`${err.message} Click "Learn this voice" again to replace it.`);
        state.replaceArmed = name;
      } else showError(err.message);
    }
  }

  /* ------------------------------------------------------------ log drawer */
  function appendLog(lines) {
    if (!lines || !lines.length) return;
    const pre = $("log");
    const stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
    pre.append(lines.map((l) => l.text).join("\n") + "\n");
    if (pre.textContent.length > 120000) pre.textContent = pre.textContent.slice(-80000);
    if (stick) pre.scrollTop = pre.scrollHeight;
  }
  function openDrawer(open) {
    $("drawer").classList.toggle("open", open);
    $("drawer").setAttribute("aria-hidden", String(!open));
    if (open) $("log").scrollTop = $("log").scrollHeight;
  }

  /* ------------------------------------------------------------ wire up */
  function wire() {
    lengthUI.wire();
    FIELDS.forEach((f) => $(f).addEventListener("input", () => { state.draftDone = false; saveDraft(); }));
    $("lyrics").addEventListener("input", lyricsMeta);
    $("style").addEventListener("input", styleTyped);
    $("singerCustom").addEventListener("input", () => { saveDraft(); renderSingers(); });
    $("btnBrowseStyles").onclick = openStyleLibrary;
    $("btnDeleteStyle").onclick = () => { const p = findStyle($("style").value.trim()); if (p) deleteStyle(p); };
    $("btnSaveStyle").onclick = saveStyle;
    $("styleSaveName").addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); saveStyle(); } });
    $("btnStyleAI").onclick = makeStyleAI;
    $("styleIdea").addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); makeStyleAI(); } });
    $("btnRestoreStyles").onclick = restoreStyles;
    $("styleClose").onclick = () => $("styleDlg").close();
    $("styleSearch").addEventListener("input", renderStyleLibrary);
    $("styleCategory").addEventListener("change", renderStyleLibrary);
    $("voicePreviewClose").onclick = closeVoicePreview;
    $("voiceAudio").addEventListener("play", () => { audio.pause(); $("recPlay").pause(); });
    $("voiceAudio").addEventListener("error", () => {
      if ($("voiceAudio").getAttribute("src")) toast("That saved voice recording could not be played.", "error");
    });
    $("idea").addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); ideaWrite(); }
    });
    $("idea").addEventListener("input", renderGuide);
    $("title").addEventListener("input", renderGuide);
    $("btnLyrics").onclick = ideaWrite;
    // guided steps
    $("guideToggle").onclick = () => {
      guide.core.on = !guide.core.on;
      try { localStorage.setItem("yss.guide", guide.core.on ? "on" : "off"); } catch (_) { /* still works this visit */ }
      renderGuide();
      if (guide.core.on) showStep(guide.core.current(guideFacts()));
    };
    document.querySelectorAll("[data-guide='next']").forEach((b) => { b.onclick = guideNext; });
    document.querySelectorAll("[data-guide='back']").forEach((b) => {
      b.onclick = () => { const at = guide.core.back(guideFacts()); saveDraft(); renderGuide(); showStep(at); };
    });
    document.querySelectorAll("[data-go]").forEach((b) => {
      b.onclick = (ev) => { ev.stopPropagation(); goStep(Number(b.dataset.go)); };
    });
    document.querySelectorAll(".step").forEach((card) => {
      card.querySelector(".step-head").onclick = () => {
        if (guide.core.on && ["done", "ahead"].includes(card.dataset.state)) goStep(Number(card.dataset.step));
      };
    });
    $("btnOwnLyrics").onclick = () => {
      guide.core.ownLyrics = true;
      if (guideNext()) $("lyrics").focus();
    };
    $("btnVoiceInvite").onclick = openVoiceDialog;
    $("emptyCreate").onclick = () => setMode("create");
    $("remixStop").onclick = startFresh;
    $("resumeGo").onclick = resumeLast;
    $("resumeDrop").onclick = () => {
      try { localStorage.removeItem("yss.lastDraft"); } catch (_) { /* nothing to drop */ }
      renderResume();
    };
    $("madeNew").onclick = startFresh;
    $("madeListen").onclick = () => listenTo(guide.made);
    $("madeAgain").onclick = () => { $("seed").value = ""; saveDraft(); makeSong(); };
    $("remixClose").onclick = () => $("remixDlg").close();
    document.querySelectorAll("[data-remix]").forEach((b) => {
      b.onclick = () => { $("remixDlg").close(); reuseSong(state.remixSong, b.dataset.remix); };
    });
    $("btnLyrics2").onclick = writeLyrics;
    ["idea", "targetLength", "lengthMinutes", "lengthSeconds", "shapeLines"].forEach((id) => {
      $(id).addEventListener("input", renderWriteButtons);
      $(id).addEventListener("change", renderWriteButtons);
    });
    $("shapeLines").addEventListener("change", () => { state.lines = Number($("shapeLines").value); renderShape(); saveDraft(); });
    $("recBtn").onclick = () => (rec.on ? stopRecording() : startRecording());
    $("voiceDlg").addEventListener("close", recReset);
    $("writer").addEventListener("change", () => { state.writer = $("writer").value; renderWriters(); saveDraft(); });
    $("songModel").addEventListener("change", () => { state.songModel = $("songModel").value; renderSongModels(); saveDraft(); });
    $("delVoiceCancel").onclick = () => $("delVoiceDlg").close();
    $("delVoiceOk").onclick = deleteVoice;
    $("btnMake").onclick = makeSong;
    document.querySelectorAll("#takesSeg [data-takes]").forEach((b) => { b.onclick = () => setTakes(b.dataset.takes); });
    document.querySelectorAll("[data-vocals]").forEach((b) => { b.onclick = () => setInstrumental(b.dataset.vocals === "off"); });
    $("btnInstrumental").onclick = () => { setInstrumental(true); skipToSound(); };
    $("btnCover").onclick = openCoverDialog;
    $("coverStop").onclick = () => { state.cover = null; renderCover(); saveDraft(); toast("Not covering any more. The lyrics stay as they are."); };
    document.querySelectorAll("#coverFollow [data-follow]").forEach((b) => {
      b.onclick = () => { if (state.cover) { state.cover.follow = b.dataset.follow; renderCover(); saveDraft(); } };
    });
    $("coverFile").addEventListener("change", () => pickCoverFile($("coverFile").files[0]));
    const coverDrop = $("coverDrop");
    ["dragenter", "dragover"].forEach((n) => coverDrop.addEventListener(n, (ev) => { ev.preventDefault(); coverDrop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((n) => coverDrop.addEventListener(n, (ev) => { ev.preventDefault(); coverDrop.classList.remove("over"); }));
    coverDrop.addEventListener("drop", (ev) => pickCoverFile(ev.dataTransfer.files[0]));
    $("coverSong").addEventListener("change", pickCoverSong);
    $("coverGo").onclick = startCover;
    $("coverCancel").onclick = () => {
      if (busy() && state.job.kind === "cover") api("/api/cancel", { method: "POST" }).catch(() => {});
      $("coverDlg").close();
    };
    renderCover();
    document.querySelectorAll("[data-skip-sound]").forEach((b) => { b.onclick = skipToSound; });
    $("btnSplit").onclick = async () => {
      const song = state.current;
      if (!song || busy()) return;
      if (await startJob(`/api/songs/${song.id}/split`, { body: {} }, "split"))
        toast("Separating the music and the vocals. This takes about a minute the first time, then seconds.");
    };
    renderVocals();
    try { state.takes = Math.min(4, Math.max(1, Number(localStorage.getItem("yss.takes")) || 1)); } catch (_) { /* default 1 */ }
    renderTakes();
    $("btnCancel").onclick = () => {
      $("btnCancel").disabled = true;
      $("btnCancel").textContent = "Cancelling…";     // model loads finish before they can stop
      state.takeRun = null;                           // and no further takes start
      api("/api/cancel", { method: "POST" }).catch(() => {});
    };
    document.querySelectorAll("[data-tag]").forEach((b) => { b.onclick = () => insertTag(b.dataset.tag); });
    $("btnClear").onclick = () => { if ($("lyrics").value) setLyrics(""); };
    $("btnUndo").onclick = () => {
      if (state.undo == null) return;
      const now = $("lyrics").value; $("lyrics").value = state.undo; state.undo = now;
      lyricsMeta(); saveDraft();
    };
    $("btnSeedClear").onclick = () => { $("seed").value = ""; saveDraft(); };
    $("btnFolder").onclick = () => api("/api/open-folder", { method: "POST" }).catch((e) => toast(e.message, "error"));
    $("btnShow").onclick = () => state.current &&
      api("/api/open-folder?sid=" + state.current.id, { method: "POST" }).catch((e) => toast(e.message, "error"));
    $("btnReuse").onclick = () => openRemix(state.current);
    $("btnLog").onclick = () => openDrawer(!$("drawer").classList.contains("open"));
    $("drawerClose").onclick = () => openDrawer(false);

    // player
    wirePlayback();
    const seek = $("seek");
    const songLength = () => audio.duration || (state.current && state.current.duration) || 0;
    const previewSeek = () => {
      state.seeking = true;
      seek.style.setProperty("--p", seek.value / 10 + "%");
      $("tCur").textContent = clock((seek.value / 1000) * songLength());
      drawViz();
    };
    const commitSeek = () => {
      if (songLength()) audio.currentTime = (seek.value / 1000) * songLength();
      state.seeking = false;
      updatePlaybackPosition(); drawViz();
    };
    seek.addEventListener("input", previewSeek);
    seek.addEventListener("change", commitSeek);
    // the waveform is a big seek bar: tap or drag anywhere on it
    const waveCanvas = $("wave");
    let dragging = false;
    const pointSeek = (ev) => {
      const box = waveCanvas.getBoundingClientRect();
      if (!box.width || !state.current) return;
      seek.value = Math.round(Math.min(1, Math.max(0, (ev.clientX - box.left) / box.width)) * 1000);
      previewSeek();
    };
    waveCanvas.addEventListener("pointerdown", (ev) => {
      if (!state.current) return;
      dragging = true;
      try { waveCanvas.setPointerCapture(ev.pointerId); } catch (_) { /* a plain tap still works */ }
      pointSeek(ev);
    });
    waveCanvas.addEventListener("pointermove", (ev) => { if (dragging) pointSeek(ev); });
    const endDrag = (commit) => () => {
      if (!dragging) return;
      dragging = false;
      if (commit) commitSeek(); else { state.seeking = false; updatePlaybackPosition(); drawViz(); }
    };
    waveCanvas.addEventListener("pointerup", endDrag(true));
    waveCanvas.addEventListener("pointercancel", endDrag(false));
    $("vol").addEventListener("input", () => {
      audio.volume = $("vol").value / 100;
      $("vol").style.setProperty("--p", $("vol").value + "%");
      localStorage.setItem("yss.vol", $("vol").value);
    });
    document.addEventListener("keydown", (ev) => {
      if (ev.code !== "Space" || !state.current) return;
      if (/^(TEXTAREA|INPUT|SELECT|BUTTON)$/.test(document.activeElement.tagName)) return;
      ev.preventDefault();
      togglePlayback();
    });
    window.addEventListener("resize", () => { wave.dirty = true; drawViz(); });

    // voice dialog
    $("voiceRecordingType").addEventListener("change", updateRecordingGuidance);
    $("clip").addEventListener("change", () => pickClip($("clip").files[0]));
    const drop = $("drop");
    ["dragenter", "dragover"].forEach((n) => drop.addEventListener(n, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((n) => drop.addEventListener(n, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (ev) => pickClip(ev.dataTransfer.files[0]));
    $("voiceTrain").onclick = () => {
      const again = state.replaceArmed && state.replaceArmed === $("voiceName").value.trim();
      state.replaceArmed = null;
      trainVoice(again);
    };
    $("voiceCancel").onclick = () => {
      if (busy() && state.job.kind === "voice") api("/api/cancel", { method: "POST" }).catch(() => {});
      $("voiceDlg").close();
    };
  }

  async function boot() {
    setTimeout(checkForUpdate, 4000);   // after the page is up; never in its way, even if startup trips
    renderIdeas();
    freshStart();
    loadDraft();
    renderVocals();
    wire();
    const vol = localStorage.getItem("yss.vol");
    if (vol != null) { $("vol").value = vol; audio.volume = vol / 100; $("vol").style.setProperty("--p", vol + "%"); }
    drawViz();
    try {
      const data = await api("/api/state");
      state.presets = data.presets;
      state.styleLibrary = data.style_library || [];
      state.styleSets = { yue2: { presets: state.presets, library: state.styleLibrary, hidden: 0 } };
      state.styleEngine = "yue2";
      setStyleSets(data.engine_styles);
      state.voices = data.voices;
      state.singers = data.singers || [];
      if (!state.singerOwn && state.singer && !state.singers.some((p) => p.id === state.singer)) state.singer = "";
      state.writers = data.writers || [];
      state.songModels = data.song_models || [];
      state.autoSongModel = data.auto_song_model || "";
      state.sections = data.sections || [];
      state.secondsPerLine = data.seconds_per_line || 3.9;
      state.structure = state.structure.filter((name) => state.sections.includes(name));
      if (!state.structure.length) state.structure = (data.default_structure || []).slice();
      if (!state.lines) state.lines = data.default_lines || 4;
      state.defaultWriter = data.default_writer || "";
      state.engine = data.engine.state;
      applyEngineStyles();                         // the chosen engine's style presets
      if (!$("style").value.trim() && state.presets.length) $("style").value = state.presets[0].style;
      const typed = $("style").value.trim();
      if (typed && !styleName(typed)) { state.styleOwn = true; state.ownStyle = state.ownStyle || $("style").value; }
      const missing = data.models.filter((m) => !m.ok && m.required);
      if (missing.length) {
        const banner = $("banner");
        banner.hidden = false;
        banner.append("The song model isn't downloaded yet. It downloads automatically with your first song (one time, several GB), or use “Download missing song models” in Pinokio. Expected at ");
        missing.forEach((m, i) => { if (i) banner.append(", "); banner.append(el("code", null, m.path)); });
        banner.append(".");
      }
      if (data.engine.state === "error") toast("The engine failed to start: " + data.engine.error, "error", { label: "Details", run: () => openDrawer(true) });
    } catch (err) {
      toast(err.message, "error");
    }
    if (guide.core) {
      if (guide.fresh) guide.core.adopt(guideFacts());
      let saved = null;
      try { saved = localStorage.getItem("yss.guide"); } catch (_) { /* default: guided */ }
      guide.core.on = saved !== "off";
      guide.ready = true;
      renderResume();
    }
    renderWriters(); renderSongModels(); renderShape(); renderStyles(); renderVoices(); lyricsMeta();
    try { await refreshSongs(); } catch (_) { /* shown by poll */ }
    poll();
  }

  // A newer OS Studio on GitHub? Pinokio's Update only shows while the app is stopped, so say so.
  async function checkForUpdate() {
    let info;
    try { info = await api("/api/update"); } catch (_) { return; }
    if (!info || !info.available) return;
    try { if (localStorage.getItem("yss.updateLater") === info.latest) return; } catch (_) { /* show it */ }
    const named = info.release ? ` (${info.release.name})` : "";
    $("updateText").textContent = `A new version of OS Studio is available${named}. To get it, stop OS Studio in Pinokio and press Update. Your songs, voices and styles are kept.`;
    $("updateWhatsNew").href = info.release ? info.release.url : info.url;
    $("updateLater").onclick = () => {
      $("updateBanner").hidden = true;
      try { localStorage.setItem("yss.updateLater", info.latest); } catch (_) { /* hidden for this visit */ }
    };
    $("updateBanner").hidden = false;
  }

  boot();
})();
