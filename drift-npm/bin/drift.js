#!/usr/bin/env node

const { spawnSync, spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const https = require("https");
const http = require("http");

const isWindows = process.platform === "win32";
const ENGINE_PORT = process.env.DRIFT_ENGINE_PORT || "8000";
const ENGINE_BASE_URL = process.env.DRIFT_ENGINE_BASE_URL || "https://github.com/lakshitsachdeva/intent2model/releases/latest/download";
const HEALTH_URL = `http://127.0.0.1:${ENGINE_PORT}/health`;
const HEALTH_TIMEOUT_MS = 2000;
const HEALTH_POLL_MS = 500;
const HEALTH_POLL_MAX = 60;

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
    const ext = isWindows ? ".exe" : "";
    const asset = `drift-engine-${plat}-${arch}${ext}`;
    const url = `${ENGINE_BASE_URL}/${asset}`;
    process.stderr.write(`drift: Downloading engine (${asset})...\n`);
    try {
      await downloadFile(url, binPath);
    } catch (e) {
      console.error("drift: Download failed.", e.message);
      console.error("drift: Set DRIFT_ENGINE_BASE_URL or run the engine manually.");
      return false;
    }
    if (!isWindows) {
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

// --- Python drift discovery (unchanged) ---
function getPipxBinPaths() {
  const home = process.env.HOME || process.env.USERPROFILE || "";
  if (!home) return [];
  const localBin = path.join(home, ".local", "bin");
  if (isWindows) {
    return [
      path.join(localBin, "drift.exe"),
      path.join(localBin, "drift"),
    ];
  }
  return [
    path.join(localBin, "drift"),
    "/usr/local/bin/drift",
  ];
}

function isLikelyUsOrNpmWrapper(filePath, content) {
  if (!content || typeof content !== "string") return false;
  const s = content.slice(0, 200);
  if (s.includes("#!/usr/bin/env node")) return true;
  if (s.includes("node") && (s.startsWith("@") || s.includes("cmd.exe"))) return true;
  return false;
}

function findPythonDrift() {
  for (const p of getPipxBinPaths()) {
    if (fs.existsSync(p)) {
      try {
        const content = fs.readFileSync(p, "utf8").slice(0, 200);
        if (!isLikelyUsOrNpmWrapper(p, content)) return p;
      } catch (_) {
        return p;
      }
    }
  }
  const pathEnv = process.env.PATH || "";
  const dirs = pathEnv.split(path.delimiter);
  for (const dir of dirs) {
    const base = path.join(dir, "drift");
    const candidates = isWindows ? [base + ".exe", base + ".cmd", base] : [base];
    for (const candidate of candidates) {
      if (!fs.existsSync(candidate)) continue;
      const ext = path.extname(candidate).toLowerCase();
      if (isWindows && (ext === ".cmd" || ext === ".bat" || ext === ".ps1")) continue;
      try {
        const content = fs.readFileSync(candidate, "utf8").slice(0, 200);
        if (isLikelyUsOrNpmWrapper(candidate, content)) continue;
        return candidate;
      } catch (_) {
        return candidate;
      }
    }
  }
  return null;
}

async function main() {
  const userBackend = process.env.DRIFT_BACKEND_URL;
  if (!userBackend) {
    const already = await engineRunning();
    if (!already) {
      const started = await ensureEngine().catch(() => false);
      if (!started) {
        console.error("drift: Engine did not start. Set DRIFT_BACKEND_URL to a running engine URL.");
        process.exit(1);
      }
    }
  }

  const driftPath = findPythonDrift();
  if (!driftPath) {
    console.error(`
drift is not installed.

Install the Python CLI first:

  pip install --user pipx
  pipx ensurepath
  pipx install drift-ml

Then run:

  drift
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
