"""
Download and start the drift engine binary when no backend is running.
Makes pipx install drift-ml work standalone (no npm needed).
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

ENGINE_TAG = "v0.1.4"
GITHUB_REPO = "lakshitsachdeva/drift"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{ENGINE_TAG}"
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


def _get_asset_download_url(asset_name: str) -> str:
    """Resolve GitHub release asset to download URL."""
    # Direct URL works for public repos, no API rate limits
    direct_url = f"https://github.com/{GITHUB_REPO}/releases/download/{ENGINE_TAG}/{asset_name}"
    r = requests.head(direct_url, timeout=10, allow_redirects=True)
    if r.status_code == 200:
        return r.url  # follow redirects to final URL

    # Fallback: API (needed for private repos or if direct fails)
    token = os.environ.get("DRIFT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    headers = {
        "User-Agent": "Drift-Engine-Launcher/1.0",
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(GITHUB_API, headers=headers, timeout=15)
    if r.status_code == 404:
        raise RuntimeError(
            f"Release {ENGINE_TAG} not found. "
            "If the repo is private, set DRIFT_GITHUB_TOKEN with repo read access."
        )
    r.raise_for_status()
    data = r.json()
    for a in data.get("assets", []):
        if a.get("name") == asset_name:
            url = a.get("browser_download_url") or a.get("url")
            if url:
                return url
    raise RuntimeError(f"Asset {asset_name} not found in release {ENGINE_TAG}")


def _download_file(url: str, dest: Path) -> None:
    token = os.environ.get("DRIFT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    headers = {"User-Agent": "Drift-Engine-Launcher/1.0"}
    # API URLs need Accept: application/octet-stream; browser_download_url works with default
    if "api.github.com" in url:
        headers["Accept"] = "application/octet-stream"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)


def ensure_engine() -> bool:
    """
    If engine not running: download (if needed), start it, wait for health.
    Returns True if engine is ready, False on failure.
    """
    if _engine_running():
        return True

    if not requests:
        print("drift: 'requests' required for engine download. pip install requests", file=sys.stderr)
        return False

    bin_path = _get_engine_path()
    bin_dir = _get_engine_dir()
    if not bin_path or not bin_dir:
        print("drift: Could not resolve engine dir (~/.drift/bin). Set HOME or USERPROFILE.", file=sys.stderr)
        return False

    bin_dir.mkdir(parents=True, exist_ok=True)

    if not bin_path.exists():
        plat, arch = _get_platform_key()
        ext = ".exe" if platform.system() == "Windows" else ""
        asset = f"drift-engine-{plat}-{arch}{ext}"
        print(f"drift: Downloading engine ({asset})...", file=sys.stderr)
        try:
            url = _get_asset_download_url(asset)
            _download_file(url, bin_path)
        except Exception as e:
            print(f"drift: Download failed: {e}", file=sys.stderr)
            return False
        if platform.system() != "Windows":
            bin_path.chmod(0o755)
        if platform.system() == "Darwin":
            try:
                subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(bin_path)], check=False, capture_output=True)
            except Exception:
                pass

    # Start engine
    env = {**os.environ, "DRIFT_ENGINE_PORT": ENGINE_PORT}
    proc = subprocess.Popen(
        [str(bin_path)],
        cwd=str(bin_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    proc.wait()  # wait briefly to avoid race
    del proc

    # Poll for health
    for _ in range(60):
        if _engine_running():
            return True
        import time
        time.sleep(0.5)
    return False
