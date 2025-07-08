@echo off
REM Monitor Pipeline Script for Windows

echo === Regulatory Pipeline Monitor ===
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

REM Run monitoring
python pipeline monitor --once

if %errorlevel% neq 0 (
    echo.
    echo Pipeline monitoring failed!
    exit /b 1
)