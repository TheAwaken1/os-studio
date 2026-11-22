/* Pure step logic for the guided composer: no DOM, audio, or network dependency. */
((root, factory) => {
  const Guide = factory();
  if (typeof module === "object" && module.exports) module.exports = Guide;
  else root.YSSGuide = Guide;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  "use strict";
  const STEPS = ["idea", "lyrics", "style", "voice", "make"];
  const LAST = STEPS.length - 1;
  const whole = (value, fallback) => (Number.isInteger(value) ? Math.min(LAST, Math.max(0, value)) : fallback);

  return class Guide {
    static get STEPS() { return STEPS.slice(); }

    constructor(saved = {}) {
      saved = saved && typeof saved === "object" ? saved : {};
      this.on = saved.on !== false;
      this.reached = whole(saved.reached, 0);
      this.at = Math.min(this.reached, whole(saved.at, this.reached));
      this.ownLyrics = Boolean(saved.ownLyrics);
    }

    /* What each step still needs before the next one opens; "" when satisfied. */
    missing(index, facts = {}) {
      if (index === 0 && !facts.idea && !facts.lyrics && !this.ownLyrics) return "Say what the song is about first.";
      if (index === 1 && !facts.lyrics) return "The song needs lyrics first.";
      if (index === 2 && !facts.style) return "Pick a sound first.";
      return "";
    }

    /* The furthest step the user may open right now. */
    limit(facts = {}) {
      let open = 0;
      while (open < LAST && open < this.reached && !this.missing(open, facts)) open += 1;
      return open;
    }

    status(index, facts = {}) {
      if (!this.on) return "open";
      const limit = this.limit(facts);
      const at = Math.min(this.at, limit);
      if (index === at) return "current";
      if (index > limit) return "locked";
      return index < at ? "done" : "ahead";
    }

    current(facts = {}) { return Math.min(this.at, this.limit(facts)); }

    go(index, facts = {}) {
      if (!Number.isInteger(index) || index < 0 || index > LAST) return false;
      if (index > this.limit(facts)) return false;
      this.at = index;
      return true;
    }

    /* Continue from the current step. Returns "" on success, or what is missing. */
    advance(facts = {}) {
      const at = this.current(facts);
      const missing = this.missing(at, facts);
      if (missing) return missing;
      if (at < LAST) {
        this.reached = Math.max(this.reached, at + 1);
        this.at = at + 1;
      }
      return "";
    }

    back(facts = {}) {
      const at = this.current(facts);
      if (at > 0) this.at = at - 1;
      return this.at;
    }

    /* Open everything a loaded song already satisfies and land on one step. */
    jump(index, facts = {}) {
      this.reached = LAST;
      this.at = Math.min(whole(index, LAST), this.limit(facts));
      return this.at;
    }

    restart() {
      this.reached = 0; this.at = 0; this.ownLyrics = false;
    }

    /* A draft saved before the guide existed: resume where its content leaves off. */
    adopt(facts = {}) {
      if (facts.lyrics) { this.reached = LAST; this.at = facts.style ? LAST : 2; }
      else if (facts.idea) { this.reached = 1; this.at = 0; }
    }

    toJSON() {
      return { on: this.on, reached: this.reached, at: this.at, ownLyrics: this.ownLyrics };
    }
  };
});
