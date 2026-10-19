@echo off
rem One-time setup: virtual environment + dependencies + desktop shortcut.
cd /d "%~dp0"
where py >nul 2>nul || (echo Python not found. Run: winget install Python.Python.3.11 & pause & exit /b 1)
if not exist ".venv" (
  py -3.11 -m venv .venv || py -3 -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo Install failed. & pause & exit /b 1)
if not exist config.yaml copy config.example.yaml config.yaml >nul
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\clipforge.lnk'); $s.TargetPath='%~dp0clipforge.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.Save()"
echo.
echo Done. A "clipforge" shortcut is on your desktop.
pause
