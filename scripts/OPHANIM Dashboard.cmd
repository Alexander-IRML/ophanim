@echo off
setlocal
title OPHANIM
echo Starting OPHANIM...
echo The application will open in your browser when it is ready.
echo.
for /f "usebackq delims=" %%P in (`wsl.exe wslpath -a -u "%~dp0.."`) do set "OPHANIM_PROJECT=%%P"
if not defined OPHANIM_PROJECT (
  echo ERROR: Could not translate the OPHANIM folder into a WSL path.
  pause
  exit /b 1
)
wsl.exe --cd "%OPHANIM_PROJECT%" --exec "%OPHANIM_PROJECT%/scripts/launch_desktop.sh" %*
set "OPHANIM_EXIT=%ERRORLEVEL%"
if "%OPHANIM_EXIT%"=="0" exit /b 0
echo.
echo ERROR: OPHANIM could not start. Exit code %OPHANIM_EXIT%.
pause
exit /b %OPHANIM_EXIT%
