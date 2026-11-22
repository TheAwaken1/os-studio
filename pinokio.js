// Menu structure follows prototype/system/examples/mochi/pinokio.js.
module.exports = {
  version: "8.0",
  title: "OS Studio",
  description: "Write lyrics, make songs, and sing them in your own voice. Windows / Linux, NVIDIA.",
  icon: "icon.svg",
  menu: async (kernel, info) => {
    const envPython = process.platform === "win32" ? "app/env/Scripts/python.exe" : "app/env/bin/python";
    const installed = info.exists(envPython) && info.exists("app/.installed.json");
    for (const [script, text] of [["install.js", "Installing"], ["update.js", "Updating"], ["reset.js", "Resetting dependencies"], ["download.js", "Downloading models"]]) {
      if (info.running(script)) return [{ default: true, icon: "fa-solid fa-terminal", text, href: script }];
    }
    if (!installed) return [{ default: true, icon: "fa-solid fa-plug", text: "Install", href: "install.js" }];
    if (info.running("start.js")) {
      const local = info.local("start.js");
      return [
        ...(local && local.url ? [{ default: true, icon: "fa-solid fa-music", text: "Open OS Studio", href: local.url }] : []),
        { default: !(local && local.url), icon: "fa-solid fa-terminal", text: "Terminal", href: "start.js" }
      ];
    }
    return [
      { default: true, icon: "fa-solid fa-play", text: "Start", href: "start.js" },
      { icon: "fa-solid fa-rotate", text: "Update", href: "update.js", confirm: "Download the latest version from GitHub? Songs, models, voices and configuration are kept." },
      { icon: "fa-solid fa-wrench", text: "Repair dependencies", href: "install.js" },
      { icon: "fa-solid fa-download", text: "Download missing song models", href: "download.js", confirm: "Download the YuE2 song model if it is missing (about 7.8 GB, or 4 GB on cards under 12 GB)? Existing weights are reused." },
      { icon: "fa-solid fa-circle-info", text: "Read me", href: "README.md" },
      { icon: "fa-solid fa-trash-can", text: "Reset dependencies only", href: "reset.js", confirm: "Remove this app's Python environment? Songs, shared models, trained voices and configuration will NOT be removed. Click Install afterward." }
    ];
  }
}
