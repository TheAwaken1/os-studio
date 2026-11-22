/* Pure playback queue: no DOM, audio, or network dependency. */
((root, factory) => {
  const Queue = factory();
  if (typeof module === "object" && module.exports) module.exports = Queue;
  else root.YSSPlaybackQueue = Queue;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  "use strict";
  return class PlaybackQueue {
    constructor({ continuous = true, shuffle = false, repeat = "all", random = Math.random } = {}) {
      this.continuous = continuous !== false;
      this.shuffle = Boolean(shuffle);
      this.repeat = ["off", "all", "one"].includes(repeat) ? repeat : "all";
      this.random = random;
      this.ids = [];
      this.order = [];
      this.current = null;
    }
    _shuffled(ids) {
      const out = ids.slice();
      for (let i = out.length - 1; i > 0; i--) {
        const j = Math.min(i, Math.max(0, Math.floor(this.random() * (i + 1))));
        [out[i], out[j]] = [out[j], out[i]];
      }
      return out;
    }
    setSongs(ids) {
      this.ids = [...new Set(ids.filter((id) => typeof id === "string" && id))];
      if (this.current && !this.ids.includes(this.current)) this.current = null;
      if (this.shuffle) {
        const retained = this.order.filter((id) => this.ids.includes(id));
        this.order = retained.concat(this._shuffled(this.ids.filter((id) => !retained.includes(id))));
      } else this.order = this.ids.slice();
    }
    select(id) {
      if (!this.ids.includes(id)) return null;
      this.current = id;
      return id;
    }
    setShuffle(enabled) {
      this.shuffle = Boolean(enabled);
      this.order = this.shuffle
        ? (this.current ? [this.current] : []).concat(this._shuffled(this.ids.filter((id) => id !== this.current)))
        : this.ids.slice();
    }
    setRepeat(value) {
      if (["off", "all", "one"].includes(value)) this.repeat = value;
    }
    peek() {
      if (!this.continuous || !this.order.length || !this.current) return null;
      if (this.repeat === "one") return this.current;
      const at = this.order.indexOf(this.current);
      if (at + 1 < this.order.length) return this.order[at + 1];
      if (this.repeat === "all" && !this.shuffle) return this.order[0];
      return null;
    }
    next({ manual = false } = {}) {
      if (!this.order.length || (!manual && !this.continuous)) return null;
      if (!this.current) return this.select(this.order[0]);
      if (!manual && this.repeat === "one") return this.current;
      const at = this.order.indexOf(this.current);
      if (at + 1 < this.order.length) return this.select(this.order[at + 1]);
      if (this.repeat === "off") return null;
      if (this.shuffle && this.order.length > 1) {
        this.order = this._shuffled(this.ids);
        if (this.order[0] === this.current) [this.order[0], this.order[1]] = [this.order[1], this.order[0]];
      }
      return this.select(this.order[0]);
    }
    previous() {
      if (!this.order.length || !this.current) return null;
      const at = this.order.indexOf(this.current);
      if (at > 0) return this.select(this.order[at - 1]);
      return this.repeat !== "off" ? this.select(this.order[this.order.length - 1]) : null;
    }
  };
});
