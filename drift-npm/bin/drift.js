#!/usr/bin/env node

const { spawnSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const isWindows = process.platform === "win32";

// Known pipx install locations (Unix and Windows)
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
  // Node launcher
  if (s.includes("#!/usr/bin/env node")) return true;
  // Windows npm .cmd wrapper (invokes node)
  if (s.includes("node") && (s.startsWith("@") || s.includes("cmd.exe"))) return true;
  return false;
}

// Find Python-based drift (not this Node script or npm's drift.cmd)
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

function main() {
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

  const result = spawnSync(driftPath, process.argv.slice(2), {
    stdio: "inherit",
    env: process.env,
  });

  process.exit(result.status === null ? (result.signal ? 128 + 9 : 1) : result.status);
}

main();
