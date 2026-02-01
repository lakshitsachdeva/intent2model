#!/usr/bin/env node

const { spawnSync } = require("child_process");

function which(cmd) {
  const isWindows = process.platform === "win32";
  const whichCmd = isWindows ? "where" : "which";
  const { status, stdout } = spawnSync(whichCmd, [cmd], { encoding: "utf8" });
  return status === 0 ? (stdout || "").trim().split("\n")[0] : null;
}

function main() {
  // Check if drift binary exists on PATH
  const driftPath = which("drift");
  
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

  // Run drift and forward stdin/stdout
  const result = spawnSync(driftPath, process.argv.slice(2), {
    stdio: "inherit",
    env: process.env,
  });

  process.exit(result.status === null ? (result.signal ? 128 + 9 : 1) : result.status);
}

main();
