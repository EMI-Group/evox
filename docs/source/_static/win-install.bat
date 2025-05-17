@echo off
setlocal enabledelayedexpansion

REM Check NVIDIA driver version
set "use_cpu=N"
set "install_triton=Y"
where nvidia-smi > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] NVIDIA driver not found. Please install the latest NVIDIA driver first from https://www.nvidia.com/drivers/.
    echo Do you want install torch cpu version instead? ^(Y/N, default N^)
    set /p use_cpu=">> "
    if "!use_cpu!"=="" set "use_cpu=N"
    if /i "!use_cpu!"=="Y" (
        goto Continue
    ) else (
        exit /b 1
    )
)

REM If using GPU, ask if the user want to install triton-windows for torch.compile support
if /i "!use_cpu!"=="N" (
    echo Do you want to install triton-windows for torch.compile support? ^(Y/N, default Y^)
    set /p install_triton=">> "
    if "!install_triton!"=="" set "install_triton=Y"
    if /i "!install_triton!"=="Y" (
        echo [INFO] Will install triton-windows
    ) else (
        echo [INFO] Skip triton-windows installation
    )
)

for /f "tokens=1* delims=," %%A in ('nvidia-smi --query-gpu=driver_version --format="csv,noheader"') do (
    echo [INFO] Detect installed NVIDIA driver version: %%A. Make sure the latest NVIDIA driver is installed from https://www.nvidia.com/drivers/
)

:Continue
@REM setlocal DisableDelayedExpansion
@REM echo %use_cpu%
REM check winget is installed
where winget > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] winget is not installed. This usually happens on LTSC and Server OS. Please manually install winget from https://github.com/microsoft/winget-cli, then reopen this script.
    pause
    exit /b 1
)

REM Check if VS Code is installed
where code > nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] VSCode is not found. Downloading Visual Studio Code installer...
    winget install -e --id Microsoft.VisualStudioCode
) else (
    echo [INFO] VSCode is already installed. Skip
)

REM Checking if Git is installed
where git > nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Git is not found. Downloading Git installer...
    winget install -e --id Git.Git
) else (
    echo [INFO] Git is already installed. Skip
)



REM Install Miniforge3 (example using curl)
@REM where conda > nul 2>&1
echo [INFO] Downloading Miniforge...
if not exist "%UserProfile%\miniforge3" (
    curl -L -o Miniforge3-Windows-x86_64.exe https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe
    if exist Miniforge3-Windows-x86_64.exe (
        echo [INFO] Installing Miniforge on %UserProfile%\miniforge3
        start /wait Miniforge3-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\miniforge3
    )
) else (
    echo [INFO] Miniforge is already installed. Skip
)


REM Temp activate conda

call %UserProfile%\miniforge3\Scripts\activate.bat %UserProfile%\miniforge3
call mamba info
echo [INFO] Installing EvoX packages...
call conda env list | findstr "evox-env" >nul
if %ERRORLEVEL%==0 (
    echo Environment evox-env already exists. Removing...
    call conda env remove -n evox-env -y
)
call mamba create -n evox-env python=3.12 -y
call mamba activate evox-env
pip install numpy jupyterlab nbformat
if /i "!use_cpu!"=="Y" (
    pip install torch
) else (
    pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu124
    pip show triton > nul 2>&1
    REM Check if install_triton is Y and triton-windows is not installed
    if /i "!install_triton!"=="Y" if %errorlevel% neq 0 (
        echo [INFO] Installing triton-windows
        echo [INFO] Downloading MSVC and Windows SDK
        curl -L -o vs_buildtools.exe https://aka.ms/vs/17/release/vs_BuildTools.exe
        if exist vs_buildtools.exe (
            echo [INFO] Installing MSVC and Windows SDK
            start /wait vs_buildtools.exe --quiet --wait --norestart --nocache --installPath %UserProfile%\vs_buildtools
        )
        echo [INFO] Downloading vcredist
        curl -L -o vcredist.exe https://aka.ms/vs/17/release/vc_redist.x64.exe
        if exist vcredist.exe (
            echo [INFO] Installing vcredist
            start /wait vcredist.exe /install /quiet /norestart
        )
        echo [INFO] Installing triton-windows pip package
        pip install -U "triton-windows<3.4"

        echo [INFO] Check if Windows file path length limit (260) exists
        echo [INFO] Attempting to modify registry to enable long path support.
        powershell -Command "Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -Command New-ItemProperty -Path \"HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem\" -Name \"LongPathsEnabled\" -Value 1 -PropertyType DWORD -Force' -Verb RunAs"
        echo [INFO] Long Path Support should now be enabled. Restart required.
    )
)
echo [INFO] Installing EvoX packages...
pip install "evox[vis]>=1.0.0" torchvision
REM Download some demo
mkdir %UserProfile%\evox-demo
git clone --depth 1 https://github.com/EMI-Group/evox.git %UserProfile%\evox-demo\evox
xcopy %UserProfile%\evox-demo\evox\docs\source\example\ %UserProfile%\evox-demo /E /I /Y
start code %UserProfile%\evox-demo

echo Reboot is highly recommended to apply the changes.

pause
:: ...existing code...
