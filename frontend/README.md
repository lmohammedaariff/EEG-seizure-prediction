# React frontend

This Vite + React app is the user interface for the project. Its `/api` requests
are proxied to the FastAPI service at `http://127.0.0.1:8000` during development.

From the project root, install Node dependencies once with `npm ci` in this
folder, then run both services with `python run_project.py`. Open
`http://127.0.0.1:5173` in your browser. Keep the terminal running while using
the dashboard; press Ctrl+C there to stop both services.

To build the frontend assets separately, run `npm run build` in this folder.
