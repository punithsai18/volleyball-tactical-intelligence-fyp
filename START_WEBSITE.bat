@echo off
title VolleyTactix AI - Tactical Intelligence Web App
echo =========================================================================
echo               VOLLEYTACTIX AI - TACTICAL INTELLIGENCE
echo =========================================================================
echo.
echo Starting FastAPI Web Server with 13 Tactical Models...
echo Access the dashboard in your browser at: http://127.0.0.1:8000
echo.

start "" "http://127.0.0.1:8000"
"C:\Users\punit\AppData\Local\Programs\Python\Python311\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
