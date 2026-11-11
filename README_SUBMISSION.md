# Minor M.Tech project submission: Code and Output

Student: Mohammed Aariff L  
Roll No: AM.SC.P2ARI25014  
Project Guide: Dr. Anjali T  
Project: Patient-Independent Multi-Horizon EEG Seizure Prediction

## Contents

- `src/`, `scripts/`, `tests/`: Python source, experiment scripts, and project tests.
- `frontend/`: React source and built interface (`dist/`).
- `api_server.py`, `run_project.py`, `Run_Project.bat`: FastAPI and dashboard launch files.
- `artifacts/results-multihorizon-16case/`: trained fold checkpoints, inference checkpoint, CHB-MIT results, person tables, uncertainty intervals, ablations, and baselines.
- `artifacts/results-siena-fullcohort/`: external transfer evaluation outputs.
- `docs/`: architecture, research gap, and evaluation integrity documents.
- `OUTPUT_SUMMARY.md`: results summary. `RUN_STATUS.txt`: live health check at package time.
- `deliverables/`: abstract and architecture document in Word/PDF formats.

## Run the dashboard

1. Extract the ZIP to a writable folder.
2. Use Python 3.11 and install dependencies with `python -m pip install -e .`.
3. Double-click `Run_Project.bat` and keep its terminal open.
4. Open `http://127.0.0.1:8000/`.

The React production build is included, so Node.js is not needed to run the dashboard. Trained checkpoints and stored results are included. Raw EEG (about 42 GB) is not bundled; EDF analysis requires a local EDF recording.

## Research scope

This is a retrospective research system. Siena transfer used one frozen CHB-MIT fold and shows a sensitivity/false-alarm trade-off; it is not clinical validation.
