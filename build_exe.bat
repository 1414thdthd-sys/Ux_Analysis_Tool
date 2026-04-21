@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo Building UX_Analysis_Tool.exe
echo ==========================================

echo.
echo Cleaning previous build output...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
for %%f in (*.spec) do del /q "%%f"

echo.
echo Running PyInstaller...
python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name UX_Analysis_Tool ^
    gui_runner.py

if errorlevel 1 (
    echo.
    echo Build failed.
    pause
    exit /b 1
)

echo.
echo Build finished successfully.
echo Output:
echo dist\UX_Analysis_Tool.exe
echo.
echo IMPORTANT:
echo Keep this EXE inside your project folder structure.
echo It will use:
echo   spython\
echo   data\
echo   results\
echo from the project root.
echo.
pause
endlocal