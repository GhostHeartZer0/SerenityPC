@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo   SerenityPC Application Builder (Main)
echo ========================================================
echo   [1] Fast Onedir Folder (Recommended - Instant HDD startup)
echo   [2] Single-File Standalone EXE
echo ========================================================
set "CHOICE=1"
set /p CHOICE="Select build target [1/2] (default: 1): "
if "%CHOICE%"=="2" (
    set "BUILD_MODE=onefile"
    echo Building single-file standalone executable...
) else (
    set "BUILD_MODE=onedir"
    echo Building onedir application folder...
)

set "APP_NAME=SerenityPC v1.8.0"

echo [1/2] Checking dependencies...
call .venv\Scripts\python.exe -m pip install pyinstaller

echo [2/2] Building %APP_NAME% (%BUILD_MODE%)...
call .venv\Scripts\python.exe -m PyInstaller SerenityPC.spec --clean -y

if "%BUILD_MODE%"=="onedir" (
    set "DIST_TARGET=dist\%APP_NAME%"
    if exist "%DIST_TARGET%" (
        echo [3/3] Assembling standalone distribution layout...
        if not exist "%DIST_TARGET%\Media" xcopy /E /I /Y "Media" "%DIST_TARGET%\Media" >nul 2>&1
        if not exist "%DIST_TARGET%\Users" mkdir "%DIST_TARGET%\Users" >nul 2>&1
        if not exist "%DIST_TARGET%\Models" mkdir "%DIST_TARGET%\Models" >nul 2>&1
        if not exist "%DIST_TARGET%\Logs" mkdir "%DIST_TARGET%\Logs" >nul 2>&1
        if exist "Models\For More Models....txt" copy /Y "Models\For More Models....txt" "%DIST_TARGET%\Models\" >nul 2>&1
    )
)

echo Build finished. Output in: dist\
pause
