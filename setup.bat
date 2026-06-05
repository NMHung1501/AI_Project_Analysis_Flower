@echo off
:: ═══════════════════════════════════════════════════════════
::  IrisANN — Setup Script
::  Tự động tạo môi trường ảo Python 3.11 và cài thư viện
:: ═══════════════════════════════════════════════════════════
title IrisANN Setup

echo.
echo  ┌────────────────────────────────────────────┐
echo  │   🌸  IrisANN — Environment Setup          │
echo  │   Tao moi truong ao Python 3.11            │
echo  └────────────────────────────────────────────┘
echo.

:: ── Kiểm tra uv (trình quản lý venv + Python) ──
where uv >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Tìm thấy uv — Sử dụng uv để tạo môi trường
    goto :USE_UV
)

:: ── Kiểm tra Python 3.11 ──
py -3.11 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Tìm thấy Python 3.11 — Tạo venv...
    goto :USE_PY311
)

:: ── Không tìm thấy → cài uv qua pip ──
echo [!] Không tìm thấy Python 3.11. Đang cài uv...
pip install uv --user
if %ERRORLEVEL% NEQ 0 (
    echo [✗] Không cài được uv. Vui lòng tải Python 3.11 từ:
    echo     https://www.python.org/downloads/release/python-3119/
    pause
    exit /b 1
)

:USE_UV
echo.
echo [1/3] Tạo môi trường ảo với Python 3.11...
uv venv .venv --python 3.11
if %ERRORLEVEL% NEQ 0 (
    echo [!] uv không tìm thấy Python 3.11 — đang tải xuống...
    uv python install 3.11
    uv venv .venv --python 3.11
)

echo.
echo [2/3] Kích hoạt môi trường và cài thư viện...
call .venv\Scripts\activate.bat
uv pip install -r requirements.txt
goto :DONE

:USE_PY311
echo.
echo [1/3] Tạo môi trường ảo...
py -3.11 -m venv .venv

echo.
echo [2/3] Kích hoạt môi trường và cài thư viện...
call .venv\Scripts\activate.bat
pip install -r requirements.txt

:DONE
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [✗] Cài đặt thất bại. Kiểm tra kết nối mạng và thử lại.
    pause
    exit /b 1
)

echo.
echo  ┌────────────────────────────────────────────┐
echo  │  ✅  Cài đặt thành công!                   │
echo  │                                            │
echo  │  Để chạy ứng dụng:                        │
echo  │  > run.bat                                 │
echo  │  Sau đó mở: http://127.0.0.1:5000         │
echo  └────────────────────────────────────────────┘
echo.
pause
