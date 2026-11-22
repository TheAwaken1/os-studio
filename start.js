// Based on prototype/system/examples/mochi/start.js; capture group per Gepeto guide.
module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: { PYTHONUTF8: "1", PYTHONUNBUFFERED: "1" },
        message: ["python -X utf8 app.py --no-browser --port {{port}}"],
        on: [{ event: "/(http:\\/\\/[0-9.:]+)/", done: true }]
      }
    },
    { method: "local.set", params: { url: "{{input.event[1]}}" } },
    { method: "process.wait", params: { url: "{{local.url}}/api/ping", interval: 1, title: "Starting OS Studio" } }
  ]
}
