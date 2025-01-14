@echo off
setlocal enabledelayedexpansion

REM Check NVIDIA driver version
set "use_cpu=N"
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
call conda env list | findstr "evox-zoo" >nul
if %ERRORLEVEL%==0 (
    echo Environment evox-zoo already exists. Removing...
    call conda env remove -n evox-zoo -y
)
call mamba create -n evox-zoo python=3.10 -y
call mamba activate evox-zoo
pip install numpy jupyterlab
if /i "!use_cpu!"=="Y" (
    pip install torch
) else (
    pip install torch --index-url https://download.pytorch.org/whl/cu124
)
pip install "evox>=1.0.0a1"
REM Download some demo
mkdir %UserProfile%\evox-demo
@REM curl -L -o %UserProfile%\evox-demo\custom_algo_prob.ipynb https://raw.githubusercontent.com/EMI-Group/evox/refs/heads/evoxtorch-main/docs/source/example/custom_algo_prob.ipynb
git clone --depth 1 https://github.com/EMI-Group/evox.git %UserProfile%\evox-demo\evox
xcopy %UserProfile%\evox-demo\evox\docs\source\example\custom_algo_prob.ipynb %UserProfile%\evox-demo /E /I /Y
start code %UserProfile%\evox-demo

echo Reboot is highly recommended to apply the changes.

pause
:: ...existing code...
