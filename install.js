// Optional local cache/tool overrides avoid re-downloading packages already cached on this PC.
// With no settings.local.json, use Pinokio's managed uv and cache normally.
const fs = require('fs');
const path = require('path');
const settingsPath = path.join(__dirname, 'settings.local.json');
const settings = fs.existsSync(settingsPath) ? JSON.parse(fs.readFileSync(settingsPath, 'utf8')) : {};
const uv = settings.uv_executable ? `"${settings.uv_executable}"` : 'uv';
// The venv's python lives in Scripts\ on Windows and bin/ on Linux.
const envPython = process.platform === 'win32' ? 'env/Scripts/python.exe' : 'env/bin/python';
const installEnv = { PYTHONUTF8: '1', UV_LINK_MODE: 'copy', ...(settings.uv_cache_dir ? { UV_CACHE_DIR: settings.uv_cache_dir } : {}) };
module.exports = {
  requires: { bundle: "ai" },
  run: [
    {
      when: "{{!which('ffmpeg')}}",
      method: "shell.run",
      params: { message: ["conda install -c conda-forge ffmpeg -y"] }
    },
    {
      when: `{{!exists('app/${envPython}')}}`,
      method: "shell.run",
      params: { message: ["uv venv --python 3.10 app/env"] }
    },
    // CUDA 13 needs NVIDIA driver 580+. Check before the multi-GB PyTorch download;
    // break (PINOKIO.md shell.run `on` handlers) shows the reason and stops the install.
    {
      method: "shell.run",
      params: {
        path: "app", venv: "env",
        message: ["python ../tools/check_driver.py"],
        on: [
          { event: "/OS Studio can't install:.*/", break: true },
          { event: "/NVIDIA driver check passed/", kill: true }
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        path: "app", venv: "env",
        env: installEnv,
        message: [
          `${uv} pip install --python ${envPython} -r requirements.txt --index-strategy unsafe-best-match`,
          `${uv} pip install --python ${envPython} whisperx==3.8.6 --no-deps`,
          "python ../tools/configure.py",
          "python ../tools/verify_install.py"
        ]
      }
    },
    // Model download step mirrors prototype/system/examples/mochi/install.js (download after deps).
    // Reuses YuE2 weights already in another Pinokio app; downloads only what is missing.
    {
      method: "shell.run",
      params: {
        path: "app", venv: "env",
        env: { PYTHONUTF8: "1" },
        message: ["python ../tools/download_models.py"]
      }
    }
  ]
}
