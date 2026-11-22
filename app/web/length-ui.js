/* Target duration guides the next lyrics draft; it never edits existing words. */
((root, factory) => {
  const LengthUI = factory();
  if (typeof module === "object" && module.exports) module.exports = LengthUI;
  else root.YSSLengthUI = LengthUI;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  "use strict";
  return class LengthUI {
    constructor({ state, $, api, clock, shapeStats, saveDraft, setTimer = setTimeout, clearTimer = clearTimeout }) {
      Object.assign(this, { state, $, api, clock, shapeStats, saveDraft });
      // Browser timer functions require the global receiver, not this controller.
      this.setTimer = (handler, delay) => setTimer.call(globalThis, handler, delay);
      this.clearTimer = (timer) => clearTimer.call(globalThis, timer);
      this.sequence = 0;
      this.timer = 0;
      this.plan = null;
      this.key = "";
      this.error = "";
    }
    target() {
      const mode = this.$("targetLength").value || "auto";
      if (mode === "auto") return { seconds: null, error: "" };
      if (["60", "120", "180"].includes(mode)) return { seconds: Number(mode), error: "" };
      const fail = { seconds: null, error: "Enter a target between 0:30 and 6:00, with seconds from 0 to 59." };
      if (mode !== "custom") return fail;
      const minutes = this.$("lengthMinutes").value.trim(), seconds = this.$("lengthSeconds").value.trim();
      if (!/^\d{1,2}$/.test(minutes) || !/^\d{1,2}$/.test(seconds)) return fail;
      const total = Number(minutes) * 60 + Number(seconds);
      if (Number(seconds) > 59 || total < 30 || total > 360) return fail;
      return { seconds: total, error: "" };
    }
    limit() {
      const memory = this.$("memory").value || "auto";
      return memory === "low" || (memory === "auto" && Number.isFinite(this.state.totalGpuGb) && this.state.totalGpuGb < 14)
        ? 180 : 360;
    }
    validation() {
      const target = this.target();
      if (target.error) return target.error;
      if (target.seconds != null && target.seconds > this.limit())
        return `This memory setting allows targets up to ${this.clock(this.limit())}. Choose a shorter target or change Graphics memory in More options.`;
      return "";
    }
    body() {
      return { structure: this.state.structure.slice(), lines: this.state.lines,
        target_seconds: this.target().seconds, memory: this.$("memory").value || "auto" };
    }
    currentKey() { return JSON.stringify([this.body(), this.limit()]); }
    summary() {
      const target = this.target(), validation = this.validation();
      const custom = this.$("targetLength").value === "custom";
      this.$("lengthCustom").hidden = !custom;
      this.$("shapeLines").disabled = (this.$("targetLength").value || "auto") !== "auto";
      ["lengthMinutes", "lengthSeconds"].forEach((id) => this.$(id).setAttribute("aria-invalid", String(custom && Boolean(target.error))));
      const active = this.plan && this.key === this.currentKey() ? this.plan : null;
      const automatic = this.shapeStats();
      const estimate = active ? active.estimated_seconds : automatic.seconds;
      const totalLines = active ? active.total_lines : automatic.lines;
      this.$("shapeMeta").textContent = totalLines ? `about ${this.clock(estimate)} · ${totalLines} lines` : "";
      let message = validation || (this.key === this.currentKey() ? this.error : "");
      let warn = Boolean(message);
      if (!message) {
        message = target.seconds == null
          ? "Auto uses your song shape and lines per section."
          : `Target ${this.clock(target.seconds)}. ` + (active
            ? `The next draft plans about ${this.clock(estimate)} (${totalLines} lyric lines).`
            : "Checking how many lyric lines fit your shape…");
        if (active && target.seconds != null && Math.abs(estimate - target.seconds) > 15) {
          message += " This shape cannot closely match that target; try adding/removing sections.";
          warn = true;
        }
        const text = this.$("lyrics").value || "";
        const sung = text.split(/\r?\n/).filter((line) => line.trim() && !/^\s*(?:\d+\s*[x×]\s*)?\[.*\]\s*$/i.test(line)).length;
        if (sung) {
          const seconds = sung * (this.state.secondsPerLine || 3.9) + (/^\s*\[intro\]\s*$/im.test(text) ? 8 : 0);
          message += ` Your current lyrics estimate about ${this.clock(seconds)}; your words stay unchanged.`;
          if (seconds > this.limit()) { message += ` They may exceed this memory setting's ${this.clock(this.limit())} ceiling.`; warn = true; }
          else if (target.seconds != null && Math.abs(seconds - target.seconds) > 20) { message += " They may not fit the target."; warn = true; }
        }
        message += " Length is approximate; tempo and instrumental parts vary. Targets guide the next Write lyrics, not an audio cut-off.";
      }
      if (this.$("lengthHelp").textContent !== message) this.$("lengthHelp").textContent = message;
      this.$("lengthHelp").classList.toggle("warn", warn);
    }
    schedule() {
      this.sequence += 1; // invalidate responses as soon as the user changes a control
      this.clearTimer(this.timer);
      this.$("lengthHelp").setAttribute("aria-busy", "false");
      this.error = "";
      this.summary();
      if (this.target().seconds == null || this.validation()) return;
      this.timer = this.setTimer(() => this.refresh(), 180);
    }
    async refresh() {
      this.clearTimer(this.timer);
      const sequence = ++this.sequence, key = this.currentKey();
      this.error = "";
      if (this.validation() || this.target().seconds == null) { this.summary(); return null; }
      this.$("lengthHelp").setAttribute("aria-busy", "true");
      try {
        const plan = await this.api("/api/lyrics-plan", { body: this.body() });
        if (sequence !== this.sequence || key !== this.currentKey()) return null;
        this.plan = plan; this.key = key;
        this.summary();
        return plan;
      } catch (error) {
        if (sequence !== this.sequence || key !== this.currentKey()) return null;
        this.plan = null; this.key = key;
        this.error = error.status === 404
          ? "Length planning is not available yet. Restart OS Studio after this update, then try again."
          : "Length planning: " + error.message;
        this.summary();
        return null;
      } finally {
        if (sequence === this.sequence) this.$("lengthHelp").setAttribute("aria-busy", "false");
      }
    }
    wire() {
      const changed = () => { this.schedule(); this.saveDraft(); };
      this.$("targetLength").addEventListener("change", changed);
      ["lengthMinutes", "lengthSeconds"].forEach((id) => this.$(id).addEventListener("input", changed));
      this.$("memory").addEventListener("change", changed);
    }
  };
});
