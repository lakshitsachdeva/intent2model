#!/usr/bin/env node
//
// drift — terminal-first AutoML. Engine auto-starts locally.
// No tokens. No auth. No backend setup.
//

const { spawnSync, spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const https = require("https");
const http = require("http");

const isWindows = process.platform === "win32";
const ENGINE_PORT = process.env.DRIFT_ENGINE_PORT || "8000";
const GITHUB_REPO = "lakshitsachdeva/drift";  // Engine binaries
const ENGINE_TAG = "v0.2.0";  // Pinned — direct URL, no API, no rate limits
const ENGINE_BASE_URL = `https://github.com/${GITHUB_REPO}/releases/download/${ENGINE_TAG}`;
const HEALTH_URL = `http://127.0.0.1:${ENGINE_PORT}/health`;
const HEALTH_TIMEOUT_MS = 2000;
const HEALTH_POLL_MS = 500;
const HEALTH_POLL_MAX = 60;
const isMac = process.platform === "darwin";

function getPlatformKey() {
  const p = process.platform;
  const a = process.arch;
  const plat = p === "darwin" ? "macos" : p === "win32" ? "windows" : "linux";
  const arch = a === "arm64" || a === "aarch64" ? "arm64" : "x64";
  return { plat, arch };
}

function getEngineDir() {
  const home = process.env.HOME || process.env.USERPROFILE || "";
  if (!home) return null;
  return path.join(home, ".drift", "bin");
}

function getEnginePath() {
  const dir = getEngineDir();
  if (!dir) return null;
  const { plat, arch } = getPlatformKey();
  const ext = isWindows ? ".exe" : "";
  return path.join(dir, `drift-engine-${plat}-${arch}${ext}`);
}

function fetchOk(url) {
  return new Promise((resolve) => {
    const client = url.startsWith("https") ? https : http;
    const req = client.get(url, { timeout: HEALTH_TIMEOUT_MS }, (res) => {
      const redirect = res.statusCode >= 301 && res.statusCode <= 302 && res.headers.location;
      if (redirect) {
        fetchOk(redirect).then(resolve).catch(() => resolve(false));
        return;
      }
      resolve(res.statusCode === 200);
    });
    req.on("error", () => resolve(false));
    req.on("timeout", () => { req.destroy(); resolve(false); });
  });
}

function getAssetUrl(assetName) {
  const baseUrl = process.env.DRIFT_ENGINE_BASE_URL;
  if (baseUrl) {
    return `${baseUrl.replace(/\/$/, "")}/${assetName}`;
  }
  return `${ENGINE_BASE_URL}/${assetName}`;
}

function downloadWithCurl(url, destPath) {
  return new Promise((resolve, reject) => {
    const result = spawnSync("curl", ["-fsSL", "-o", destPath, url], {
      stdio: "pipe",
      timeout: 120000,
    });
    if (result.status !== 0) {
      reject(new Error("Download failed"));
      return;
    }
    resolve();
  });
}

function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith("https") ? https : http;
    const req = client.get(url, { headers: { "User-Agent": "Drift/1.0" } }, (res) => {
      const redirect = res.statusCode >= 301 && res.statusCode <= 302 && res.headers.location;
      if (redirect) {
        downloadFile(redirect, destPath).then(resolve).catch(reject);
        return;
      }
      if (res.statusCode !== 200) {
        reject(new Error(`Download failed: ${res.statusCode}`));
        return;
      }
      const dir = path.dirname(destPath);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      const file = fs.createWriteStream(destPath);
      res.pipe(file);
      file.on("finish", () => { file.close(); resolve(); });
      file.on("error", reject);
    });
    req.on("error", reject);
  });
}

function engineRunning() {
  return fetchOk(HEALTH_URL);
}

function waitForEngine() {
  return new Promise((resolve) => {
    let n = 0;
    const t = setInterval(() => {
      n++;
      fetchOk(HEALTH_URL).then((ok) => {
        if (ok) { clearInterval(t); resolve(true); }
        else if (n >= HEALTH_POLL_MAX) { clearInterval(t); resolve(false); }
      });
    }, HEALTH_POLL_MS);
  });
}

async function ensureEngine() {
  const binPath = getEnginePath();
  const binDir = getEngineDir();
  if (!binPath || !binDir) {
    console.error("drift: Could not resolve engine directory (~/.drift/bin). Set HOME or USERPROFILE.");
    return false;
  }
  if (!fs.existsSync(binDir)) {
    fs.mkdirSync(binDir, { recursive: true });
  }
  if (!fs.existsSync(binPath)) {
    const { plat, arch } = getPlatformKey();
    const ext = isWindows ? ".exe" : "";
    const asset = `drift-engine-${plat}-${arch}${ext}`;
    const url = getAssetUrl(asset);
    process.stderr.write(`drift: Downloading engine (${asset})...\n`);
    try {
      await downloadWithCurl(url, binPath).catch(() => downloadFile(url, binPath));
    } catch (e) {
      console.error("drift: Download failed.", e.message);
      return false;
    }
    if (!isWindows) {
      try { fs.chmodSync(binPath, 0o755); } catch (_) {}
    }
    if (isMac) {
      try {
        spawnSync("xattr", ["-dr", "com.apple.quarantine", binPath], { stdio: "pipe" });
      } catch (_) {}
    }
  }
  if (isMac && binPath) {
    try {
      fs.chmodSync(binPath, 0o755);
      spawnSync("xattr", ["-dr", "com.apple.quarantine", binPath], { stdio: "pipe" });
    } catch (_) {}
  }
  const child = spawn(binPath, [], {
    detached: true,
    stdio: "ignore",
    cwd: binDir,
    env: { ...process.env, DRIFT_ENGINE_PORT: ENGINE_PORT },
  });
  child.unref();
  return waitForEngine();
}

function findPythonDrift() {
  const home = process.env.HOME || process.env.USERPROFILE || "";
  if (!home) return null;
  const localBin = path.join(home, ".local", "bin");
  if (isWindows) {
    const exe = path.join(localBin, "drift.exe");
    if (fs.existsSync(exe)) return exe;
    return null;
  }
  const unix = path.join(localBin, "drift");
  if (fs.existsSync(unix)) return unix;
  return null;
}

async function main() {
  const userBackend = process.env.DRIFT_BACKEND_URL;
  if (!userBackend) {
    const already = await engineRunning();
    if (!already) {
      const started = await ensureEngine().catch((e) => {
        console.error("drift:", e.message);
        return false;
      });
      if (!started) {
        console.error("Failed to start drift engine.");
        process.exit(1);
      }
    }
  }

  const driftPath = findPythonDrift();
  if (!driftPath) {
    console.error(`
drift is not installed. Install it with:

  pipx install drift-ml
`);
    process.exit(1);
  }

  const backendUrl = userBackend || `http://127.0.0.1:${ENGINE_PORT}`;
  const env = { ...process.env, DRIFT_BACKEND_URL: backendUrl };

  const result = spawnSync(driftPath, process.argv.slice(2), {
    stdio: "inherit",
    env,
  });

  process.exit(result.status === null ? (result.signal ? 128 + 9 : 1) : result.status);
}

main().catch((e) => {
  console.error("drift:", e.message);
  process.exit(1);
});
