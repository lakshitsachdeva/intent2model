#!/usr/bin/env node

const { spawnSync } = require("child_process");
const path = require("path");
const fs = require("fs");

// Known pipx install locations
function getPipxBinPaths() {
  const home = process.env.HOME || process.env.USERPROFILE || "";
  return [
    path.join(home, ".local", "bin", "drift"),
    "/usr/local/bin/drift",
  ];
}

// Find Python-based drift (not this Node script)
function findPythonDrift() {
  // First check pipx standard locations
  for (const p of getPipxBinPaths()) {
    if (fs.existsSync(p)) {
      // Verify it's not a Node script (i.e., not us)
      try {
        const content = fs.readFileSync(p, "utf8").slice(0, 100);
        if (!content.includes("#!/usr/bin/env node")) {
          return p;
        }
      } catch (_) {}
    }
  }
  
  // Fallback: search PATH but skip Node scripts
  const pathEnv = process.env.PATH || "";
  const dirs = pathEnv.split(path.delimiter);
  
  for (const dir of dirs) {
    const candidate = path.join(dir, "drift");
    if (fs.existsSync(candidate)) {
      try {
        const content = fs.readFileSync(candidate, "utf8").slice(0, 100);
        // Skip if it's a Node script (that's us or another Node launcher)
        if (content.includes("#!/usr/bin/env node")) {
          continue;
        }
        return candidate;
      } catch (_) {
        // Binary file or unreadable - might be the real one
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

Install it with:

  pipx install drift-ml

Then run:

  drift
`);
    process.exit(1);
  }

  // Run the Python drift and forward stdin/stdout
  const result = spawnSync(driftPath, process.argv.slice(2), {
    stdio: "inherit",
    env: process.env,
  });

  process.exit(result.status === null ? (result.signal ? 128 + 9 : 1) : result.status);
}

main();
