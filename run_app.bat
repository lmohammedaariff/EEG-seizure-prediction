@echo off
setlocal
set "PROJECT=E:\seizure_prediction"
cd /d "%PROJECT%"
if not exist ".venv\Scripts\python.exe" (
  echo Project Python environment is missing: %PROJECT%\.venv
  pause
  exit /b 1
)
if not exist ".tmp" mkdir ".tmp"
set "TEMP=%PROJECT%\.tmp"
set "TMP=%PROJECT%\.tmp"
set "STREAMLIT_CONFIG_DIR=%PROJECT%\.streamlit"
set "MPLCONFIGDIR=%PROJECT%\.mplconfig"
set "MNE_DATA=%PROJECT%\.mne-data"
set "XDG_CACHE_HOME=%PROJECT%\.cache"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
echo Starting Seizure Prediction app from %PROJECT%
echo Keep this window open while using the app. Press Ctrl+C here to stop it.
set "PORT=8501"
for /L %%P in (8501,1,8510) do (
  netstat -ano | findstr /R /C:":%%P .*LISTENING" >nul
  if errorlevel 1 (
    set "PORT=%%P"
    goto port_found
  )
)
:port_found
echo Streamlit will open http://127.0.0.1:%PORT% after it is ready.
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.headless false --server.address 127.0.0.1 --server.port %PORT%
if errorlevel 1 echo Streamlit stopped with an error. Read the message above, then try again.
pause
