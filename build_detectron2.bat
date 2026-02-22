@echo off
echo ============================================
echo  Building Detectron2 for Windows
echo  Python 3.11 + CUDA 12.4 + PyTorch 2.6
echo ============================================

REM Set up Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if %ERRORLEVEL% NEQ 0 (echo [ERROR] vcvarsall failed && pause && exit /b 1)

REM Critical flags for MSVC + Python build
set DISTUTILS_USE_SDK=1
set MSSdk=1

REM Add git and CUDA to PATH
set PATH=C:\Program Files\Git\bin;C:\Program Files\Git\cmd;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%

REM CUDA flags
set TORCH_CUDA_ARCH_LIST=8.9
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4

echo [OK] Environment set up. Building...

REM Download detectron2 source
set SRC_DIR=%TEMP%\detectron2_build
if exist %SRC_DIR% rmdir /s /q %SRC_DIR%
mkdir %SRC_DIR%

git clone https://github.com/facebookresearch/detectron2.git %SRC_DIR% --depth 1 --branch v0.6
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] v0.6 tag not found, using main branch...
    git clone https://github.com/facebookresearch/detectron2.git %SRC_DIR% --depth 1
)

echo [OK] Source cloned. Installing...
C:\Python311\python.exe -m pip install %SRC_DIR% --no-build-isolation

echo Done. Exit code: %ERRORLEVEL%
