from __future__ import annotations

import os
import argparse
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"


def _frontend_command() -> list[str]:
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm:
        return [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173", "--strictPort"]

    # Some Windows installations have Node but no npm on PATH. Vite is already
    # installed by `npm ci`, so invoking its CLI directly is sufficient.
    node = shutil.which("node") or shutil.which("node.exe")
    vite = FRONTEND_DIR / "node_modules" / "vite" / "bin" / "vite.js"
    if node and vite.is_file():
        return [node, str(vite), "--host", "127.0.0.1", "--port", "5173", "--strictPort"]

    raise RuntimeError(
        "Node.js and npm are required. Install Node.js, then run `npm ci` in frontend/."
    )


def _wait_for_api(process: subprocess.Popen, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("FastAPI exited during startup; inspect the server output above.")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError("FastAPI did not become ready within 45 seconds.")


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EEG dashboard and FastAPI service.")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run Vite separately for hot reload. The default serves the bundled React build from FastAPI.",
    )
    args = parser.parse_args()
    if not FRONTEND_DIR.is_dir():
        raise RuntimeError(f"React frontend not found: {FRONTEND_DIR}")
    if not args.dev and not (FRONTEND_DIR / "dist" / "index.html").is_file():
        raise RuntimeError(
            "The bundled React build is missing. Run `npm ci` and `npm run build` in frontend/, "
            "or start development mode with `python run_project.py --dev`."
        )

    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=1) as response:
            if response.status == 200:
                print("The project dashboard is already running at http://127.0.0.1:8000")
                return
    except (urllib.error.URLError, TimeoutError):
        pass

    env = os.environ.copy()
    project_temp = ROOT / "tmp_uploads"
    project_temp.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(project_temp)
    env["TMP"] = str(project_temp)
    env["TMPDIR"] = str(project_temp)
    env["PIP_CACHE_DIR"] = str(ROOT / "artifacts" / "pip-cache")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    backend: subprocess.Popen | None = None
    frontend: subprocess.Popen | None = None
    try:
        backend = subprocess.Popen(
            [sys.executable, str(ROOT / "api_server.py")],
            cwd=ROOT,
            env=env,
        )
        _wait_for_api(backend)
        if args.dev:
            frontend = subprocess.Popen(
                _frontend_command(),
                cwd=FRONTEND_DIR,
                env=env,
            )
            print("React development dashboard: http://127.0.0.1:5173")
        else:
            print("React dashboard: http://127.0.0.1:8000")
        print("FastAPI API: http://127.0.0.1:8000")
        print("Press Ctrl+C to stop both services.")
        while True:
            if backend.poll() is not None:
                raise RuntimeError("FastAPI stopped unexpectedly.")
            if frontend is not None and frontend.poll() is not None:
                raise RuntimeError("React/Vite stopped unexpectedly.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping React and FastAPI…")
    finally:
        _stop(frontend)
        _stop(backend)


if __name__ == "__main__":
    main()
