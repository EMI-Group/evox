:: filepath: /Users/zaber/gitrepo/ai-win-builder/install.bat
:: ...existing code...
@echo off

REM Check NVIDIA driver version (simple demonstration; replace %LATEST_VERSION% with actual known latest)

where nvidia-smi > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] NVIDIA driver not found! Please install the latest NVIDIA driver first from https://www.nvidia.com/drivers/.
    set /p use_cpu="Do you want install torch cpu version instead? (Y|N, default N)"
    REM 转换用户输入为大写（批处理不区分大小写，可以按需修改）
    @REM set use_cpu=%use_cpu:~0,1%
    @REM set use_cpu=%use_cpu:~0,1%
    if /i "%use_cpu%"=="Y" (
        goto :Continue
    ) else (
        exit /b 1
    )
)

:Continue
for /f "tokens=1* delims=," %%A in ('nvidia-smi --query-gpu=driver_version --format="csv,noheader"') do (
    echo [INFO] Detect installed NVIDIA driver version: %%A. Make sure the latest NVIDIA driver is installed from https://www.nvidia.com/drivers/
)

REM Check if VS Code is installed
where code > nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] VSCode is not founded! Downloading Visual Studio Code installer...
    curl -L -o vscode-installer.exe "https://code.visualstudio.com/sha/download?build=stable&os=win32-x64-user"
    if exist vscode-installer.exe (
        echo [INFO] Installing Visual Studio Code...
        start /wait vscode-installer.exe 
    )
) else (
    echo [INFO] VSCode is already installed. Skip!
)

REM Checking if Git is installed
where git > nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Git is not founded! Downloading Git installer...
    curl -L -o git-installer.exe "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe"
    if exist git-installer.exe (
        start /wait git-installer.exe
        REM Refresh the environment variables
        curl -L -o RefreshEnv.cmd https://raw.githubusercontent.com/chocolatey/choco/refs/heads/master/src/chocolatey.resources/redirects/RefreshEnv.cmd
        call RefreshEnv.cmd
        where git > nul 2>&1
        if %errorlevel% neq 0 (
            echo [ERROR] Git is not installed or detected! Please install Git from https://gitforwindows.org/.
            pause
            exit /b 1
        )
    )
) else (
    echo [INFO] Git is already installed. Skip!
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
    echo [INFO] Miniforge is already installed. Skip!
)


REM Temp activate conda
if exist "%UserProfile%\miniforge3" (
    %UserProfile%\miniforge3\Scripts\activate.bat %UserProfile%\miniforge3
    mamba info
    echo [INFO] Installing EvoX packages...
    conda env list | findstr "evox-zoo" >nul

    IF %ERRORLEVEL%==0 (
        echo Environment "evox-zoo" already exists. Removing...
        conda env remove -n evox-zoo -y
    )
    mamba create -n evox-zoo python=3.10 -y
    mamba activate evox-zoo
    pip install numpy jupyterlab

    if /i "%use_cpu%"=="Y" (
        pip install torch
    ) else (
        pip install torch --index-url https://download.pytorch.org/whl/cu124
    )

    pip install -e git+https://github.com/EMI-Group/evox@evoxtorch-main#egg=evox

    REM Download some demo
    mkdir %UserProfile%\evox-demo
    curl -L -o %UserProfile%\evox-demo\custom_algo_prob.ipynb https://raw.githubusercontent.com/EMI-Group/evox/refs/heads/evoxtorch-main/docs/source/example/custom_algo_prob.ipynb
    code %UserProfile%\evox-demo
) else (
    echo [ERROR] Miniforge is not installed or detected! Please install Miniforge manually from https://github.com/conda-forge/miniforge/releases/latest/download
)



echo Reboot is highly recommended to apply the changes.

echo Print any key to exit the script and open the vscode!

pause
:: ...existing code...