@echo off
echo =======================================================
echo     Starting Karate 3D Avatar Studio Server...
echo =======================================================

:: Start the Python HTTP server in the background
start /b python -m http.server 8080 --bind 127.0.0.1

echo.
echo Waiting for server to initialize...
timeout /t 2 /nobreak >nul

:: Open Chrome or default browser to the correct localhost URL
echo Opening Avatar Studio in your browser...
start http://127.0.0.1:8080/avatar_studio.html

echo.
echo NOTE: Do not close this window while you are using the studio.
echo Close this terminal window when you are finished.
pause
