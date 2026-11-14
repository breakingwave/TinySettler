#!/usr/bin/env python3
"""Manage this Mac's dedicated working-file session; never migrate other sessions."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "dev/sync.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "pause", "resume", "flush", "create"])
    args = parser.parse_args()
    mutagen = shutil.which("mutagen") or "/opt/homebrew/bin/mutagen"
    sessions = json.loads(subprocess.check_output(
        [mutagen, "sync", "list", "--template", "{{ json . }}"], text=True)) or []
    ours = [s for s in sessions if s.get("name") == CONFIG["name"]]
    if len(ours) > 1:
        raise SystemExit("Duplicate TinySettler sessions; inspect manually.")
    if ours:
        s = ours[0]
        assert Path(s["alpha"]["path"]).samefile(ROOT)
        assert s["beta"]["host"] == "rex-coding-sync"
        assert s["beta"]["path"] == "/home/rexdev/projects/TinySettler"
    if args.action == "create":
        if ours:
            raise SystemExit("Session already exists; use status or resume.")
        assert Path(CONFIG["alpha"]).samefile(ROOT)
        remote = CONFIG["beta"].split(":", 1)[1]
        for s in sessions:
            for e in (s["alpha"], s["beta"]):
                path = e.get("path", "").rstrip("/")
                target = str(ROOT) if e["protocol"] == "local" else remote
                if e["protocol"] == "local":
                    path, target = path.casefold(), target.casefold()
                relevant = e["protocol"] == "local" or e.get("host") in (
                    "rex-coding-sync", "hostinger", "2.25.188.2", "187.124.118.69")
                if relevant and path and (path == target or target.startswith(path + "/") or path.startswith(target + "/")):
                    raise SystemExit("Overlapping session exists (even if paused); inspect manually.")
        command = [mutagen, "sync", "create", "--name", CONFIG["name"],
                   "--no-global-configuration", "--mode", CONFIG["mode"], "--ignore-vcs"]
        for pattern in CONFIG["ignores"]:
            command += ["--ignore", pattern]
        subprocess.run(command + [str(ROOT), CONFIG["beta"]], check=True)
    else:
        if not ours:
            raise SystemExit("TinySettler session does not exist.")
        action = ["list", "--long"] if args.action == "status" else [args.action]
        subprocess.run([mutagen, "sync", *action, ours[0]["identifier"]], check=True)


if __name__ == "__main__":
    main()
