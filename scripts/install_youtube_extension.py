"""Register the extension's local model host for this Windows user, without a web server.

Load extension/youtube as an unpacked extension in Chrome or Edge after this command.
Registration is restricted to this extension's stable public-key-derived ID.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "com.factchecker.youtube"


def extension_id(key: str) -> str:
    digest = hashlib.sha256(base64.b64decode(key, validate=True)).hexdigest()[:32]
    return "".join(chr(ord("a") + int(char, 16)) for char in digest)


def install() -> tuple[str, Path]:
    if sys.platform != "win32":
        raise SystemExit("This host installer currently supports Windows.")
    import winreg

    extension = ROOT / "extension/youtube"
    identity = extension_id(json.loads((extension / "manifest.json").read_text(encoding="utf-8"))["key"])
    python = ROOT / ".venv/Scripts/python.exe"
    if not python.is_file():
        raise SystemExit("Install requirements.txt into .venv before registering the local host.")
    destination = ROOT / "runs/youtube-extension"
    destination.mkdir(parents=True, exist_ok=True)
    launcher = destination / "host.cmd"
    # Escape percent expansion in batch paths; delayed expansion is explicitly off.
    workspace, executable = str(ROOT).replace("%", "%%"), str(python).replace("%", "%%")
    launcher.write_text(f'@echo off\nsetlocal DisableDelayedExpansion\nset PYTHONDONTWRITEBYTECODE=1\nset PYTHONIOENCODING=utf-8\ncd /d "{workspace}"\n"{executable}" -m scripts.youtube_host\n', encoding="utf-8")
    manifest = destination / "host.json"
    manifest.write_text(json.dumps({"name": HOST, "description": "Local caption claim verifier", "path": str(launcher),
                                   "type": "stdio", "allowed_origins": [f"chrome-extension://{identity}/"]}, indent=2), encoding="utf-8")
    for browser in (r"Google\Chrome", r"Microsoft\Edge"):
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"Software\{browser}\NativeMessagingHosts\{HOST}") as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, str(manifest))
    return identity, extension


def main() -> None:
    identity, extension = install()
    print(f"Local model host registered for Chrome and Edge. Extension ID: {identity}")
    print(f"Open chrome://extensions or edge://extensions, enable Developer mode, and Load unpacked: {extension}")


if __name__ == "__main__":
    main()
