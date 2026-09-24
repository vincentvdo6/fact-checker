"""Register the extension's local model host for this Windows user, without a web server.

Load extension/youtube as an unpacked extension in Chrome or Edge after this command.
Registration is restricted to this extension's stable public-key-derived ID.

`--record [DIR]` writes FACT_CHECKER_RECORD_CLICKS into the host launcher so every check is
recorded under DIR (default runs/clicks, absolute) without touching the Windows environment or
restarting the browser: the browser spawns the launcher afresh after the extension is reloaded.
`--no-record` removes it.
"""

from __future__ import annotations

import argparse
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


def launcher_text(workspace: Path, executable: Path, record: Path | None) -> str:
    """The batch launcher the browser spawns; a record directory is baked in as an environment line."""
    # Escape percent expansion in batch paths; delayed expansion is explicitly off.
    lines = ["@echo off", "setlocal DisableDelayedExpansion", "set PYTHONDONTWRITEBYTECODE=1", "set PYTHONIOENCODING=utf-8",
             'set "FACT_CHECKER_RECORD_CLICKS="']
    if record is not None:
        lines.append(f'set "FACT_CHECKER_RECORD_CLICKS={str(record).replace("%", "%%")}"')
    lines += [f'cd /d "{str(workspace).replace("%", "%%")}" || exit /b 1',
              f'"{str(executable).replace("%", "%%")}" -m scripts.youtube_host']
    return "\n".join(lines) + "\n"


def install(record: Path | None = None) -> tuple[str, Path]:
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
    if record is not None:
        record.mkdir(parents=True, exist_ok=True)
    launcher = destination / "host.cmd"
    launcher.write_text(launcher_text(ROOT, python, record), encoding="utf-8")
    manifest = destination / "host.json"
    manifest.write_text(json.dumps({"name": HOST, "description": "Local caption claim verifier", "path": str(launcher),
                                   "type": "stdio", "allowed_origins": [f"chrome-extension://{identity}/"]}, indent=2), encoding="utf-8")
    for browser in (r"Google\Chrome", r"Microsoft\Edge"):
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"Software\{browser}\NativeMessagingHosts\{HOST}") as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, str(manifest))
    return identity, extension


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", nargs="?", const="runs/clicks", default=None, metavar="DIR",
                        help="record every check under DIR (default runs/clicks); the path is made absolute")
    parser.add_argument("--no-record", action="store_true", help="stop recording (the default when neither flag is given)")
    args = parser.parse_args()
    record = None if args.no_record or args.record is None else (ROOT / args.record).resolve()
    identity, extension = install(record)
    print(f"Local model host registered for Chrome and Edge. Extension ID: {identity}")
    print(f"Open chrome://extensions or edge://extensions, enable Developer mode, and Load unpacked: {extension}")
    if record is not None:
        print(f"Every check will be recorded under {record}; reload the extension so the browser spawns the new launcher.")


if __name__ == "__main__":
    main()
