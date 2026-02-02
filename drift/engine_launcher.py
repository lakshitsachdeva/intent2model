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

GITHUB_REPO = "lakshitsachdeva/drift"  # Engine binaries (v0.2.0+ also in intent2model)
ENGINE_TAG = "v0.2.0"  # Pinned — direct URL, no API, no rate limits
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
        except Exception:
            pass

    env = {**os.environ, "DRIFT_ENGINE_PORT": ENGINE_PORT}
    proc = subprocess.Popen(
        [str(bin_path)],
        cwd=str(bin_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    proc.wait()
    del proc

    for _ in range(60):
        if _engine_running():
            return True
        import time
        time.sleep(0.5)
    return False
