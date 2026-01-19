@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  7D CRYSTAL SYSTEM - FULL SYSTEM STRESS LAUNCHER
REM  Runs CPU Core Mapper + GPU Benchmark simultaneously
REM ═══════════════════════════════════════════════════════════════════════════

title 7D Crystal System - Full Stress Test

echo.
echo ╔═══════════════════════════════════════════════════════════════════╗
echo ║       🔮 7D CRYSTAL SYSTEM - FULL STRESS TEST 🔮                  ║
echo ╠═══════════════════════════════════════════════════════════════════╣
echo ║ CPU: AMD Ryzen 7 4800H (8 cores / 16 threads)                     ║
echo ║ GPU: NVIDIA GeForce GTX 1660 Ti (6GB GDDR6)                       ║
echo ╠═══════════════════════════════════════════════════════════════════╣
echo ║ Φ = 1.618033988749895 ^| S² = 0.01 ^| κ = Φ⁻¹                      ║
echo ╚═══════════════════════════════════════════════════════════════════╝
echo.

REM Set NVRTC path for GPU acceleration
set NVRTC_PATH=C:\Users\BASEDGOD\AppData\Roaming\Python\Python311\site-packages\nvidia\cuda_nvrtc\bin
set PATH=%NVRTC_PATH%;%PATH%

echo [1/3] Setting up CUDA environment...
echo       NVRTC: %NVRTC_PATH%
echo.

echo [2/3] Launching CPU Core Mapper (16 threads)...
start "7D Crystal - CPU" /MIN "%~dp0target\release\core_mapper.exe"
echo       ✓ CPU stress started in background
echo.

echo [3/3] Launching GPU Benchmark (CUDA)...
start "7D Crystal - GPU" /MIN "%~dp0target\release\gpu_benchmark.exe"
echo       ✓ GPU stress started in background
echo.

echo ╔═══════════════════════════════════════════════════════════════════╗
echo ║                    BOTH STRESSORS ACTIVE                          ║
echo ╠═══════════════════════════════════════════════════════════════════╣
echo ║ CPU: 16 threads running Φ-manifold operations                     ║
echo ║ GPU: CUDA kernels running projection/fold/distance                ║
echo ╠═══════════════════════════════════════════════════════════════════╣
echo ║ Monitor: Task Manager → Performance → CPU + GPU                   ║
echo ╚═══════════════════════════════════════════════════════════════════╝
echo.

echo Press any key to STOP both stressors...
pause > nul

echo.
echo Stopping stressors...
taskkill /IM core_mapper.exe /F 2>nul
taskkill /IM gpu_benchmark.exe /F 2>nul

echo.
echo ╔═══════════════════════════════════════════════════════════════════╗
echo ║                    STRESS TEST COMPLETE                           ║
echo ║ Φ = 1.618033988749895 ^| S² = 0.01 ^| Sovereignty Maintained ✓    ║
echo ╚═══════════════════════════════════════════════════════════════════╝
echo.
pause
