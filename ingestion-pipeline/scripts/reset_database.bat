@echo off
REM Reset Database Script for Windows

echo === Regulatory Chat Bot Database Reset ===
echo.

REM Check if .env file exists
if not exist .env (
    echo Error: .env file not found!
    echo Please create a .env file with your database credentials.
    exit /b 1
)

REM Load environment variables from .env
for /f "tokens=*" %%a in ('type .env ^| findstr /v "^#"') do (
    set %%a
)

REM Confirm reset
echo Warning: This will reset the entire database!
set /p confirm="Are you sure you want to continue? (yes/no): "

if /i not "%confirm%"=="yes" (
    echo Database reset cancelled.
    exit /b 0
)

REM Run the reset
echo.
echo Starting database reset...
python pipeline reset --group all --force

if %errorlevel% equ 0 (
    echo.
    echo Database reset completed successfully!
) else (
    echo.
    echo Database reset failed!
    exit /b 1
)