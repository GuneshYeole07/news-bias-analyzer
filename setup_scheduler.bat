@echo off
REM ======================================================
REM  News Bias Analyzer — Schedule Daily Refresh
REM  Run this once as Administrator to register a daily task
REM ======================================================

set TASK_NAME=NewsBiasAnalyzer_DailyRefresh
set SCRIPT_DIR=%~dp0
set PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe
set SCRIPT=%SCRIPT_DIR%daily_refresh.py
set HOUR=08
set MINUTE=00

echo.
echo ===================================================
echo  News Bias Analyzer — Daily Scheduler Setup
echo ===================================================
echo.
echo Task Name : %TASK_NAME%
echo Python    : %PYTHON%
echo Script    : %SCRIPT%
echo Schedule  : Every day at %HOUR%:%MINUTE%
echo.

REM Delete existing task if present
schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

REM Create new scheduled task
schtasks /Create ^
    /TN "%TASK_NAME%" ^
    /TR "\"%PYTHON%\" \"%SCRIPT%\"" ^
    /SC DAILY ^
    /ST %HOUR%:%MINUTE% ^
    /F

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Scheduled task created successfully!
    echo    The data pipeline will run daily at %HOUR%:%MINUTE%
    echo.
    echo    To modify the time, edit HOUR and MINUTE in this file
    echo    To remove: schtasks /Delete /TN "%TASK_NAME%" /F
) else (
    echo.
    echo ❌ Failed. Try running this script as Administrator.
)

pause
