@echo off
cd /d "%~dp0"
echo Installing requirements (first time only)...
py -m pip install -r requirements.txt
echo Starting HealthGuard...
start "" http://localhost:5000
py app.py
pause
