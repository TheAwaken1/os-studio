"""Tell the user when GitHub has a newer OS Studio than this install.

Pinokio installs the app with `git clone`, so the install knows its own commit and
where it came from. Once per start (cached for 6 hours) this asks GitHub whether the
branch has moved on. The request carries only the public repo name and commit id;
nothing about the user. No git, no GitHub remote, offline, or `"update_check": false`
in config.json -> no notice, never an error.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_SECONDS = 6 * 3600
USER_AGENT = "OS-Studio-update-check"
_lock = threading.Lock()
_cache: dict = {"at": 0.0, "result": None}


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def read_head(git_dir: str) -> tuple[str | None, str | None]:
    """(commit sha, branch name) of the checkout, read straight from .git (git may not be on PATH)."""
    head = (_read(os.path.join(git_dir, "HEAD")) or "").strip()
    if re.fullmatch(r"[0-9a-f]{40}", head):
        return head, None                                   # detached checkout
    match = re.fullmatch(r"ref: (refs/heads/(.+))", head)
    if not match:
        return None, None
    ref, branch = match.group(1), match.group(2)
    sha = (_read(os.path.join(git_dir, *ref.split("/"))) or "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        sha = None
        for line in (_read(os.path.join(git_dir, "packed-refs")) or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref and re.fullmatch(r"[0-9a-f]{40}", parts[0]):
                sha = parts[0]
                break
    return sha, branch


def github_repo(git_dir: str) -> str | None:
    """'owner/name' of the origin remote when it is on github.com."""
    config = _read(os.path.join(git_dir, "config")) or ""
    section = re.search(r'\[remote "origin"\](.*?)(?:\n\[|\Z)', config, re.S)
    url = re.search(r"^\s*url\s*=\s*(\S+)", section.group(1), re.M) if section else None
    if not url:
        return None
    match = re.search(r"github\.com[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", url.group(1))
    return f"{match.group(1)}/{match.group(2)}" if match else None


def _fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        return json.load(resp)


def check(repo_dir: str = REPO_DIR, fetch=_fetch_json) -> dict:
    none = {"available": False}
    git_dir = os.path.join(repo_dir, ".git")
    if not os.path.isdir(git_dir):
        return none
    sha, branch = read_head(git_dir)
    repo = github_repo(git_dir)
    if not (sha and repo):
        return none
    branch = branch or "main"
    try:
        diff = fetch(f"https://api.github.com/repos/{repo}/compare/{sha}...{branch}")
    except (urllib.error.URLError, OSError, ValueError):
        return none                                         # offline, rate-limited, or a local-only commit
    behind = int(diff.get("ahead_by") or 0)                 # how far GitHub's branch is ahead of us
    if diff.get("status") not in ("ahead", "diverged") or behind <= 0:
        return {**none, "current": sha[:7]}
    result = {"available": True, "current": sha[:7], "latest": (diff.get("commits") or [{}])[-1].get("sha", "")[:7],
              "behind": behind, "url": f"https://github.com/{repo}/compare/{sha[:7]}...{branch}", "release": None}
    try:
        release = fetch(f"https://api.github.com/repos/{repo}/releases/latest")
        if release.get("tag_name"):
            result["release"] = {"tag": release["tag_name"], "name": release.get("name") or release["tag_name"],
                                 "url": release.get("html_url") or result["url"]}
    except (urllib.error.URLError, OSError, ValueError):
        pass                                                # no releases yet: the compare link still works
    return result


def cached_check() -> dict:
    with _lock:
        if _cache["result"] is None or time.time() - _cache["at"] > CACHE_SECONDS:
            _cache["result"] = check()
            _cache["at"] = time.time()
        return _cache["result"]
