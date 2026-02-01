# PyInstaller spec for drift engine binary.
# Run from project root: pyinstaller engine_build/drift-engine.spec
# Output: dist/drift-engine (Unix) or dist/drift-engine.exe (Windows)
# Paths are absolute so the spec works regardless of cwd or spec location.

import sys
import os

# Repo root = cwd when running: pyinstaller engine_build/drift-engine.spec (from repo root)
# __file__ is not defined when PyInstaller exec()s the spec
REPO_ROOT = os.path.abspath(os.getcwd())
SCRIPT_PATH = os.path.join(REPO_ROOT, "backend", "run_engine.py")
BACKEND_PATH = os.path.join(REPO_ROOT, "backend")

# Platform-specific output name for release (drift-engine-macos-arm64, etc.)
platform_map = {"darwin": "macos", "linux": "linux", "win32": "windows"}
arch_map = {"x86_64": "x64", "aarch64": "arm64", "arm64": "arm64", "AMD64": "x64"}
plat = platform_map.get(sys.platform, sys.platform)
if sys.platform == "win32":
    arch = arch_map.get(os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"), "x64")
else:
    arch = arch_map.get(os.uname().machine, "x64")
exe_suffix = ".exe" if sys.platform == "win32" else ""
out_name = f"drift-engine-{plat}-{arch}{exe_suffix}"

block_cipher = None

a = Analysis(
    [SCRIPT_PATH],
    pathex=[REPO_ROOT, BACKEND_PATH],
    binaries=[],
    datas=[],
    hiddenimports=[
        "main",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "fastapi",
        "starlette",
        "pydantic",
        "pandas",
        "numpy",
        "sklearn",
        "sklearn.ensemble",
        "sklearn.linear_model",
        "sklearn.model_selection",
        "sklearn.preprocessing",
        "sklearn.metrics",
        "dotenv",
        "ml",
        "ml.profiler",
        "ml.trainer",
        "ml.evaluator",
        "ml.pipeline_builder",
        "agents",
        "agents.automl_agent",
        "agents.llm_interface",
        "agents.error_analyzer",
        "agents.recovery_agent",
        "agents.intent_detector",
        "agents.planner_agent",
        "agents.execution_planner",
        "agents.autonomous_executor",
        "agents.explainer_agent",
        "agents.reasoning_agent",
        "schemas",
        "schemas.pipeline_schema",
        "schemas.session_schema",
        "schemas.run_state_schema",
        "utils",
        "utils.logging",
        "utils.artifact_generator",
        "utils.api_key_manager",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=out_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
