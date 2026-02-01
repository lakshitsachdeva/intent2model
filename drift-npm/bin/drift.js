#!/usr/bin/env node

const { spawnSync, spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const https = require("https");
const http = require("http");

const ENGINE_PORT = process.env.DRIFT_ENGINE_PORT || "8000";
const ENGINE_BASE = process.env.DRIFT_ENGINE_BASE_URL || "https://github.com/lakshitsachdeva/drift/releases/latest/download";
const HEALTH_URL = `http://127.0.0.1:${ENGINE_PORT}/health`;
const HEALTH_TIMEOUT_MS = 2000;
const HEALTH_POLL_MS = 500;
const HEALTH_POLL_MAX = 60; // 30s

function which(cmd) {
  const { status, stdout } = spawnSync("which", [cmd], { encoding: "utf8" });
  return status === 0 ? (stdout || "").trim() : null;
}

function getPlatformKey() {
  const p = process.platform;
  const a = process.arch;
  const plat = p === "darwin" ? "macos" : p === "win32" ? "windows" : "linux";
  const arch = a === "arm64" || a === "aarch64" ? "arm64" : "x64";
  return { plat, arch };
}

function getEngineDir() {
  const home = process.env.HOME || process.env.USERPROFILE || process.env.HOMEPATH;
  if (!home) return null;
  return path.join(home, ".drift", "bin");
}

function getEnginePath() {
  const dir = getEngineDir();
  if (!dir) return null;
  const { plat, arch } = getPlatformKey();
  const ext = process.platform === "win32" ? ".exe" : "";
  return path.join(dir, `drift-engine-${plat}-${arch}${ext}`);
}

function fetchOk(url) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith("https") ? https : http;
    const req = client.get(url, { timeout: HEALTH_TIMEOUT_MS }, (res) => {
      const redirect = res.statusCode >= 301 && res.statusCode <= 302 && res.headers.location;
      if (redirect) {
        fetchOk(redirect).then(resolve).catch(reject);
        return;
      }
      resolve(res.statusCode === 200);
    });
    req.on("error", () => resolve(false));
    req.on("timeout", () => { req.destroy(); resolve(false); });
  });
}

function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith("https") ? https : http;
    const req = client.get(url, (res) => {
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
  if (!fs.existsSync(binPath)) {
    const { plat, arch } = getPlatformKey();
    const ext = process.platform === "win32" ? ".exe" : "";
    const asset = `drift-engine-${plat}-${arch}${ext}`;
    const url = `${ENGINE_BASE}/${asset}`;
    process.stderr.write(`drift: Downloading engine (${asset})...\n`);
    try {
      await downloadFile(url, binPath);
    } catch (e) {
      console.error("drift: Download failed.", e.message);
      console.error("drift: Run the engine manually or set DRIFT_ENGINE_BASE_URL.");
      return false;
    }
    if (process.platform !== "win32") {
      try { fs.chmodSync(binPath, 0o755); } catch (_) {}
    }
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

async function main() {
  const userBackend = process.env.DRIFT_BACKEND_URL;
  if (userBackend) {
    // User runs engine elsewhere; skip download/start and connect to their URL.
  } else if (await engineRunning()) {
    // Engine already running locally.
  } else {
    const engineStarted = await ensureEngine().catch(() => false);
    if (!engineStarted) {
      console.error("drift: Engine did not start. Run the engine manually or set DRIFT_BACKEND_URL to a running engine URL.");
      process.exit(1);
    }
  }

  const python = which("python3") || which("python");
  if (!python) {
    console.error("drift: Python is required for the chat CLI. Install Python 3 and ensure `python` or `python3` is on your PATH.");
    process.exit(1);
  }

  const pip = spawnSync(python, ["-m", "pip", "install", "--quiet", "--upgrade", "drift"], {
    encoding: "utf8",
    stdio: ["inherit", "pipe", "inherit"],
  });
  if (pip.status !== 0) {
    console.error("drift: Failed to install or upgrade the drift Python package.");
    process.exit(pip.status === null ? 1 : pip.status);
  }

  const backendUrl = userBackend || `http://127.0.0.1:${ENGINE_PORT}`;
  const run = spawnSync(python, ["-m", "drift"], {
    encoding: "utf8",
    stdio: "inherit",
    env: { ...process.env, DRIFT_BACKEND_URL: backendUrl },
  });
  process.exit(run.status === null ? (run.signal ? 128 + 9 : 1) : run.status);
}

main().catch((e) => {
  console.error("drift:", e.message);
  process.exit(1);
});
