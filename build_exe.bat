@echo off
echo ========================================================
echo  MoCap2MMD Standalone Executable Builder
echo ========================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo [1/3] Ensuring PyInstaller is installed...
python -m pip install --quiet pyinstaller

echo [2/3] Building MoCap2MMD.exe with PyInstaller...
python -m PyInstaller MoCap2MMD.spec --noconfirm --clean

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed! Check the log messages above.
    pause
    exit /b %errorlevel%
)

echo.
echo [3/3] Build succeeded!
echo Output executable: dist\MoCap2MMD.exe
echo ========================================================
pause
