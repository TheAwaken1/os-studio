// Mirrors prototype/system/examples/mochi/update.js: pull the latest launcher + app code,
// then re-run install so new dependencies and missing models are picked up.
// Songs, voices, models and app/config.json are git-ignored and never touched.
module.exports = {
  run: [
    { method: "shell.run", params: { message: "git pull" } },
    { method: "script.start", params: { uri: "install.js" } }
  ]
}
