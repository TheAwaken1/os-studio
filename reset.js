// Reset dependencies ONLY. Never remove songs, configuration, voices (app/voices) or models.
module.exports = {
  run: [
    { method: "script.stop", params: { uri: "start.js" } },
    { method: "fs.rm", params: { path: "app/.installed.json" } },
    { method: "fs.rm", params: { path: "app/env" } }
  ]
}
