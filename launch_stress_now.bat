@echo off
echo Launching 7D Crystal System Benchmarks...

echo [1/2] Launching CPU Core Mapper...
start "7D Crystal - CPU" /MIN "C:\Users\BASEDGOD\Desktop\7D_Crystal_System\target\release\core_mapper.exe"

echo [2/2] Launching GPU Benchmark...
start "7D Crystal - GPU" /MIN "C:\Users\BASEDGOD\Desktop\7D_Crystal_System\target\release\gpu_benchmark.exe"

echo Benchmarks launched in background.
exit
