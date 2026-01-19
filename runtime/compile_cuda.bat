@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\BASEDGOD\Desktop\7D_Crystal_System\runtime\src"
nvcc --version
nvcc -ptx -arch=sm_75 -O3 --use_fast_math -o crystal7d_precompiled.ptx cuda_kernels.cu
if exist crystal7d_precompiled.ptx (
    echo SUCCESS: PTX compiled to crystal7d_precompiled.ptx
    dir crystal7d_precompiled.ptx
) else (
    echo FAILED: PTX compilation failed
)
pause
