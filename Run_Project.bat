@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"
if not exist "%PROJECT_ROOT%tmp_uploads" mkdir "%PROJECT_ROOT%tmp_uploads"
set "TEMP=%PROJECT_ROOT%tmp_uploads"
set "TMP=%PROJECT_ROOT%tmp_uploads"
set "TMPDIR=%PROJECT_ROOT%tmp_uploads"
if exist "%PROJECT_ROOT%.venv-gpu\Scripts\python.exe" (
  "%PROJECT_ROOT%.venv-gpu\Scripts\python.exe" "%PROJECT_ROOT%run_project.py"
) else if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
  "%PROJECT_ROOT%.venv\Scripts\python.exe" "%PROJECT_ROOT%run_project.py"
) else (
  python "%PROJECT_ROOT%run_project.py"
)
pause
