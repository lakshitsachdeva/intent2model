"""
Download and start the drift engine binary when no backend is running.
Open source — no tokens, no auth. Downloads from public GitHub Releases.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

GITHUB_REPO = "lakshitsachdeva/intent2model"  # Engine binaries (same repo)
ENGINE_TAG = "v0.2.6"  # Pinned — direct URL, no API, no rate limits
ENGINE_PORT = os.environ.get("DRIFT_ENGINE_PORT", "8000")
HEALTH_URL = f"http://127.0.0.1:{ENGINE_PORT}/health"


def _get_platform_key() -> Tuple[str, str]:
    """Return (plat, arch) e.g. ('macos', 'arm64')."""
    p = platform.system().lower()
    a = platform.machine().lower()
    plat = "macos" if p == "darwin" else "windows" if p == "windows" else "linux"
    arch = "arm64" if a in ("arm64", "aarch64") else "x64"
    if p == "windows" and "amd64" in a:
        arch = "x64"
    return plat, arch


def _get_engine_dir() -> Optional[Path]:
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if not home:
        return None
    return Path(home) / ".drift" / "bin"


def _get_engine_path() -> Optional[Path]:
    d = _get_engine_dir()
    if not d:
        return None
    plat, arch = _get_platform_key()
    ext = ".exe" if platform.system() == "Windows" else ""
    return d / f"drift-engine-{plat}-{arch}{ext}"


def _engine_running() -> bool:
    if not requests:
        return False
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _get_asset_url(asset_name: str) -> str:
    """Direct download URL. No API — avoids rate limits."""
    base = os.environ.get("DRIFT_ENGINE_BASE_URL")
    if base:
        return f"{base.rstrip('/')}/{asset_name}"
    return f"https://github.com/{GITHUB_REPO}/releases/download/{ENGINE_TAG}/{asset_name}"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Drift/1.0"}

    def try_curl():
        result = subprocess.run(
            ["curl", "-fsSL", "-o", str(dest), url],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError("curl failed")

    def try_requests():
        r = requests.get(url, headers=headers, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)

    def try_urllib():
        import urllib.request
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(dest, "wb") as f:
                f.write(resp.read())

    for attempt in [try_curl, try_requests, try_urllib]:
        try:
            attempt()
            return
        except Exception:
            continue
    raise RuntimeError(f"Download failed. Try: curl -L -o {dest} {url}")


def ensure_engine() -> bool:
    """Download (if needed), start engine, wait for health."""
    if _engine_running():
        return True

    if not requests:
        print("drift: 'requests' required. pip install requests", file=sys.stderr)
        return False

    bin_path = _get_engine_path()
    bin_dir = _get_engine_dir()
    if not bin_path or not bin_dir:
        print("drift: Could not resolve ~/.drift/bin. Set HOME or USERPROFILE.", file=sys.stderr)
        return False

    bin_dir.mkdir(parents=True, exist_ok=True)

    if not bin_path.exists():
        plat, arch = _get_platform_key()
        ext = ".exe" if platform.system() == "Windows" else ""
        asset = f"drift-engine-{plat}-{arch}{ext}"
        print(f"drift: Downloading engine ({asset})...", file=sys.stderr)
        try:
            url = _get_asset_url(asset)
            _download_file(url, bin_path)
        except Exception as e:
            print(f"drift: Download failed: {e}", file=sys.stderr)
            print(f"drift: Run: mkdir -p ~/.drift/bin && curl -L -o ~/.drift/bin/{asset} <url>", file=sys.stderr)
            return False
        if platform.system() != "Windows":
            bin_path.chmod(0o755)
        if platform.system() == "Darwin":
            try:
                subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(bin_path)], check=False, capture_output=True)
            except Exception:
                pass

    if platform.system() == "Darwin" and bin_path:
        try:
            bin_path.chmod(0o755)
            subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(bin_path)], check=False, capture_output=True)
            # Ad-hoc sign so macOS Gatekeeper doesn't kill the binary (requires non-UPX build)
            subprocess.run(["codesign", "-s", "-", "--force", str(bin_path)], check=False, capture_output=True)
        except Exception:
            pass

    # macOS: use wrapper script via bash to avoid spawn -88
    # Windows: use batch file so engine starts reliably (inherits PATH for gemini etc.)
    if platform.system() == "Darwin":
        wrapper = bin_dir / "run-engine.sh"
        port = ENGINE_PORT
        bin_name = bin_path.name
        script = f'#!/bin/bash\ncd "$(dirname "$0")"\nexport DRIFT_ENGINE_PORT="{port}"\nexec ./{bin_name}\n'
        if not wrapper.exists() or wrapper.read_text() != script:
            wrapper.write_text(script)
            wrapper.chmod(0o755)
        launch_cmd = ["/bin/bash", str(wrapper)]
    elif platform.system() == "Windows":
        bat = bin_dir / "run-engine.bat"
        port = ENGINE_PORT
        bin_name = bin_path.name
        script = f'@echo off\ncd /d "%~dp0"\nset DRIFT_ENGINE_PORT={port}\nstart /b "" {bin_name}\n'
        if not bat.exists() or bat.read_text() != script:
            bat.write_text(script)
        launch_cmd = ["cmd", "/c", str(bat)]
    else:
        launch_cmd = [str(bin_path)]

    env = {**os.environ, "DRIFT_ENGINE_PORT": ENGINE_PORT}
    stderr_file = bin_dir / ".engine-stderr.log"
    proc = None
    try:
        with open(stderr_file, "w") as errf:
            proc = subprocess.Popen(
                launch_cmd,
                cwd=str(bin_dir),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                start_new_session=True,
            )
        try:
            proc.wait(timeout=3)
            if proc.returncode and proc.returncode != 0:
                err = stderr_file.read_text().strip() if stderr_file.exists() else ""
                print(f"drift: Engine exited with code {proc.returncode}", file=sys.stderr)
                if err:
                    print(f"drift: {err[-400:]}", file=sys.stderr)
                return False
        except subprocess.TimeoutExpired:
            pass  # Engine still running
    except Exception as e:
        print(f"drift: Engine spawn failed: {e}", file=sys.stderr)
        return False

    import time
    print("drift: Starting engine (first run may take 30s)...", file=sys.stderr)
    for i in range(120):  # 60s — PyInstaller binary can take 30+ s to unpack
        if _engine_running():
            return True
        time.sleep(0.5)
    err = stderr_file.read_text().strip() if stderr_file.exists() else ""
    if err:
        print(f"drift: Engine log: {err[-400:]}", file=sys.stderr)
    return False
