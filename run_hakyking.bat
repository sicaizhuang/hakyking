@echo off
cd /d "%~dp0"
if exist "%~dp0tools\rubberband" set "PATH=%~dp0tools\rubberband\rubberband-4.0.0-gpl-executable-windows;%PATH%"
if not defined HAKYKING_PITCH_ENGINE set "HAKYKING_PITCH_ENGINE=parselmouth_psola"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m hakyking.main
) else (
    python -m hakyking.main
)
