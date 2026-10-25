@echo off
title Stopping VolleyTactix AI Web Server
echo Stopping Uvicorn web server running on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a
)
echo Done!
pause
