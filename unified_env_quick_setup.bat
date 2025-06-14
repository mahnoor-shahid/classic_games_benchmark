@echo off
REM unified_env_quick_setup.bat - Unified Environment for Separate Games

echo Classic Games - Unified Environment Setup
echo ==========================================
echo Creating SHARED environment for SEPARATE game projects
echo.

REM Check if Python is available
echo [INFO] Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python %PYTHON_VERSION% found

REM Check if PyYAML is available
echo [INFO] Checking PyYAML availability...
python -c "import yaml" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing PyYAML...
    python -m pip install PyYAML
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install PyYAML
        pause
        exit /b 1
    )
    echo [SUCCESS] PyYAML installed
) else (
    echo [SUCCESS] PyYAML already available
)

REM Check if unified_env_setup.py exists
if not exist "unified_env_setup.py" (
    echo [ERROR] unified_env_setup.py not found in current directory
    echo [ERROR] Please ensure you have the unified environment setup script
    pause
    exit /b 1
)

REM Show what this setup does
echo.
echo [INFO] This will create:
echo   * Shared environment: classic_games_env
echo   * Shared MNIST data: ./shared_data/mnist/
echo   * Activation script: activate_classic_games.bat
echo   * Setup guide: SETUP_GUIDE.md
echo.
echo [INFO] Your game projects remain SEPARATE:
echo   * sudoku/     - Independent Sudoku project
echo   * kenken/     - Independent KenKen project  
echo   * your_game/  - Your future game projects
echo.

REM Ask for confirmation
set /p REPLY="Proceed with unified environment setup? [Y/n]: "
if /i not "%REPLY%"=="y" if /i not "%REPLY%"=="yes" if not "%REPLY%"=="" (
    echo Setup cancelled.
    pause
    exit /b 0
)

REM Run the unified environment setup
echo [INFO] Running unified environment setup...
python unified_env_setup.py --skip-confirmation

if %errorlevel% neq 0 (
    echo [ERROR] Unified environment setup failed. Check output above.
    pause
    exit /b 1
)

echo.
echo Unified Environment Setup Complete!
echo ==================================
echo.
echo Environment: classic_games_env
echo Shared data: ./shared_data/
echo Setup guide: SETUP_GUIDE.md
echo.
echo Quick start:
echo   1. activate_classic_games.bat       # Activate environment
echo   2. Read SETUP_GUIDE.md              # Project structure guide
echo   3. Create game directories:
echo      * mkdir sudoku     (your Sudoku project)
echo      * mkdir kenken     (your KenKen project)
echo   4. Each game is independent but uses shared environment
echo.
echo Benefits:
echo   * One environment for all games
echo   * Shared MNIST dataset  
echo   * Independent game projects
echo   * Easy to add new games
echo.

REM Ask if user wants to activate environment
set /p REPLY="Would you like to activate the environment now? [y/N]: "
if /i "%REPLY%"=="y" (
    echo.
    echo [INFO] Activating classic_games_env...
    call activate_classic_games.bat
) else (
    echo.
    echo To activate later: activate_classic_games.bat
    echo.
    echo Press any key to exit...
    pause >nul
)