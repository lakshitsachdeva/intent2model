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
const GITHUB_REPO = "lakshitsachdeva/intent2model";  // Engine binaries (same repo)
const ENGINE_TAG = "v0.2.11";  // Pinned — direct URL, no API, no rate limits
const ENGINE_BASE_URL = `https://github.com/${GITHUB_REPO}/releases/download/${ENGINE_TAG}`;
const HEALTH_URL = `http://127.0.0.1:${ENGINE_PORT}/health`;
const HEALTH_TIMEOUT_MS = 2000;
const HEALTH_POLL_MS = 500;
const HEALTH_POLL_MAX = 120;  // 60s — PyInstaller binary can take 30+ s to unpack on first run
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
  // Re-download if engine tag changed (e.g. after npm upgrade)
  const versionFile = path.join(binDir, ".engine-tag");
  if (fs.existsSync(binPath)) {
    try {
      const stored = fs.existsSync(versionFile) ? fs.readFileSync(versionFile, "utf8").trim() : "";
      if (stored !== ENGINE_TAG) {
        fs.unlinkSync(binPath);
      }
    } catch (_) {}
  }
  if (!fs.existsSync(binPath)) {
    const { plat, arch } = getPlatformKey();
    const ext = isWindows ? ".exe" : "";
    const asset = `drift-engine-${plat}-${arch}${ext}`;
    const url = getAssetUrl(asset);
    process.stderr.write(`drift: Downloading engine (${asset})...\n`);
    try {
      await downloadWithCurl(url, binPath).catch(() => downloadFile(url, binPath));
      try {
        fs.writeFileSync(versionFile, ENGINE_TAG);
      } catch (_) {}
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
      // Ad-hoc sign so macOS Gatekeeper doesn't kill the binary
      spawnSync("codesign", ["-s", "-", "--force", binPath], { stdio: "pipe" });
    } catch (_) {}
  }

  const absPath = path.resolve(binPath);
  const env = { ...process.env, DRIFT_ENGINE_PORT: ENGINE_PORT };

  // Windows + npm/pipx: Engine inherits limited PATH. Prepend npm/pipx bins so Gemini CLI is found.
  if (isWindows) {
    const home = process.env.USERPROFILE || process.env.HOME || "";
    const appdata = process.env.APPDATA || "";
    const localappdata = process.env.LOCALAPPDATA || "";
    const pf = process.env.ProgramFiles || "";
    const pf86 = process.env["ProgramFiles(x86)"] || "";
    const extraPaths = [
      path.join(localappdata, "npm"),
      path.join(appdata, "npm"),
      path.join(pf, "nodejs"),
      path.join(pf86, "nodejs"),
      path.join(home, ".local", "bin"),
    ].filter((p) => p && fs.existsSync(p));
    if (extraPaths.length) {
      const sep = path.delimiter;
      env.PATH = extraPaths.join(sep) + sep + (env.PATH || "");
    }
  }

  // macOS: spawn a wrapper script instead of the binary directly (avoids -88)
  // Windows: use batch file so engine starts reliably (inherits PATH for gemini etc.)
  const binName = path.basename(binPath);
  let toSpawn = absPath;
  let spawnArgs = [];

  if (isMac) {
    const wrapperPath = path.join(binDir, "run-engine.sh");
    const wrapperScript = `#!/bin/bash
cd "$(dirname "$0")"
export DRIFT_ENGINE_PORT="${ENGINE_PORT}"
exec ./${binName}
`;
    if (!fs.existsSync(wrapperPath) || fs.readFileSync(wrapperPath, "utf8") !== wrapperScript) {
      fs.writeFileSync(wrapperPath, wrapperScript);
      fs.chmodSync(wrapperPath, 0o755);
    }
    toSpawn = "/bin/bash";
    spawnArgs = [wrapperPath];
  } else if (isWindows) {
    const batPath = path.join(binDir, "run-engine.bat");
    const batScript = `@echo off
cd /d "%~dp0"
set DRIFT_ENGINE_PORT=${ENGINE_PORT}
start /b "" ${binName}
`;
    if (!fs.existsSync(batPath) || fs.readFileSync(batPath, "utf8") !== batScript) {
      fs.writeFileSync(batPath, batScript);
    }
    toSpawn = "cmd";
    spawnArgs = ["/c", batPath];
  }

  process.stderr.write("drift: Starting engine (first run may take 30s)...\n");

  const stderrLog = path.join(binDir, ".engine-stderr.log");
  let stderrFd;
  try {
    stderrFd = fs.openSync(stderrLog, "w");
  } catch (_) {
    stderrFd = "ignore";
  }
  const stdio = stderrFd === "ignore" ? "ignore" : ["ignore", "ignore", stderrFd];

  function trySpawn(cmd, args) {
    return new Promise((resolve, reject) => {
      let child;
      try {
        child = spawn(cmd, args || [], {
          detached: true,
          stdio,
          cwd: binDir,
          env,
        });
      } catch (e) {
        reject(e);
        return;
      }
      child.on("error", (err) => reject(err));
      child.unref();
      waitForEngine().then(resolve);
    });
  }

  try {
    return await trySpawn(toSpawn, spawnArgs);
  } catch (e) {
    const isSpawn88 = (e.errno === -88 || (e.message && String(e.message).includes("-88")));
    if (isMac && isSpawn88) {
      process.stderr.write("drift: Spawn failed (-88). Trying nohup fallback...\n");
      try {
        const cmd = (isMac && wrapperPath) ? `"${wrapperPath}"` : `"${absPath}"`;
        spawnSync("/bin/sh", ["-c", `nohup ${cmd} > /dev/null 2>&1 &`], {
          cwd: binDir,
          env: { ...process.env, DRIFT_ENGINE_PORT: ENGINE_PORT },
          stdio: "pipe",
        });
        return await waitForEngine();
      } catch (e2) {
        console.error("drift: Engine failed. Run manually in another terminal:");
        console.error("  ~/.drift/bin/drift-engine-macos-arm64");
        console.error("Then run drift again.");
        throw e2;
      }
    }
    throw e;
  }
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
        const binDir = getEngineDir();
        if (binDir) {
          const stderrLog = path.join(binDir, ".engine-stderr.log");
          if (fs.existsSync(stderrLog)) {
            const err = fs.readFileSync(stderrLog, "utf8").trim();
            if (err) console.error("drift: Engine log:", err.slice(-500));
          }
          const enginePath = getEnginePath();
          if (isWindows && enginePath) {
            console.error("drift: Engine failed on Windows. Try running the backend manually:");
            console.error("  cd backend");
            console.error("  python -m uvicorn main:app --host 0.0.0.0 --port 8000");
            console.error("  (Create .env in project root with GEMINI_API_KEY=... for LLM)");
            console.error("Then: set DRIFT_BACKEND_URL=http://localhost:8000");
            console.error("Or run the engine manually: cd %USERPROFILE%\\.drift\\bin && drift-engine-windows-x64.exe");
          }
        }
        process.exit(1);
      }
    }
  }

  const driftPath = findPythonDrift();
  if (!driftPath) {
    console.error("drift: Python CLI not found. Install with:");
    if (isWindows) {
      console.error("  pip install pipx");
      console.error("  pipx ensurepath");
      console.error("  (restart PowerShell)");
    }
    console.error("  pipx install drift-ml");
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
