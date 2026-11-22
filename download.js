module.exports = {
  run: [{
    method: "shell.run",
    params: {
      path: "app", venv: "env", env: { PYTHONUTF8: "1" },
      message: ["python ../tools/download_models.py"]
    }
  }]
}
