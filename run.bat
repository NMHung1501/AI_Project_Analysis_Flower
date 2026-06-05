@echo off
:: ═══════════════════════════════════════════════════════════
::  IrisANN — Run Script
::  Khởi động Flask server
:: ═══════════════════════════════════════════════════════════
title IrisANN Server

echo.
echo  🌸  IrisANN — Khởi động server...
echo  Địa chỉ: http://127.0.0.1:5000
echo.

:: Kích hoạt môi trường ảo
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [!] Chưa có môi trường ảo. Chạy setup.bat trước!
    pause
    exit /b 1
)

:: Khởi động Flask
python app.py

pause
