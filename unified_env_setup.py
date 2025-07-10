#!/usr/bin/env python3
"""
unified_env_setup.py - Classic Games Unified Environment Setup (FIXED)

This script handles the complete setup for multiple puzzle games including:
- Sudoku with MNIST
- KenKen with MNIST  
- Future games (Kakuro, Nonogram, etc.)
"""

import os
import sys
import subprocess
import platform
import shutil
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import venv

class UnifiedEnvironmentSetup:
    def __init__(self):
        self.env_name = "classic_games_env"
        self.setup_log = []
        self.supported_games = ['sudoku', 'kenken', 'kakuro', 'nonogram']
        
    def log_step(self, message: str, level: str = "INFO"):
        """Log setup steps"""
        log_entry = f"[{level}] {message}"
        self.setup_log.append(log_entry)
        print(f"🔧 {message}")
    
    def check_python_compatibility(self) -> bool:
        """Check Python version compatibility"""
        version_info = sys.version_info
        required_version = (3, 8)
        
        if version_info[:2] >= required_version:
            self.log_step(f"Python {version_info.major}.{version_info.minor}.{version_info.micro} - Compatible")
            return True
        else:
            self.log_step(f"Python {version_info.major}.{version_info.minor} - Requires 3.8+", "ERROR")
            return False
    
    def check_system_requirements(self) -> bool:
        """Check system requirements"""
        requirements_met = True
        
        # Check available disk space (minimum 3GB for all games)
        try:
            disk_usage = shutil.disk_usage(os.getcwd())
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb >= 3.0:
                self.log_step(f"Disk space: {free_gb:.1f}GB available")
            else:
                self.log_step(f"Disk space: {free_gb:.1f}GB - Need at least 3GB", "ERROR")
                requirements_met = False
                
        except Exception as e:
            self.log_step(f"Could not check disk space: {e}", "WARNING")
        
        # Check for required system tools
        required_tools = ['pip', 'python']
        for tool in required_tools:
            if shutil.which(tool):
                self.log_step(f"{tool} found")
            else:
                self.log_step(f"{tool} not found", "ERROR")
                requirements_met = False
        
        return requirements_met
    
    def create_unified_environment(self) -> Tuple[bool, Optional[str]]:
        """Create unified virtual environment for all games"""
        try:
            venv_path = os.path.join(os.getcwd(), self.env_name)
            
            if os.path.exists(venv_path):
                self.log_step(f"Unified environment {self.env_name} already exists")
                return True, venv_path
            
            self.log_step(f"Creating unified virtual environment: {self.env_name}")
            venv.create(venv_path, with_pip=True)
            
            # Get paths
            if platform.system() == "Windows":
                activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
                python_exe = os.path.join(venv_path, "Scripts", "python.exe")
            else:
                activate_script = os.path.join(venv_path, "bin", "activate")
                python_exe = os.path.join(venv_path, "bin", "python")
            
            self.log_step(f"Unified environment created: {venv_path}")
            self.log_step(f"Activation script: {activate_script}")
            
            return True, venv_path
            
        except Exception as e:
            self.log_step(f"Error creating unified environment: {e}", "ERROR")
            return False, None
    
    def install_shared_dependencies(self, venv_path: str) -> bool:
        """Install shared dependencies for all games"""
        try:
            # Determine executables
            if platform.system() == "Windows":
                python_exe = os.path.join(venv_path, "Scripts", "python.exe")
                pip_exe = os.path.join(venv_path, "Scripts", "pip.exe")
            else:
                python_exe = os.path.join(venv_path, "bin", "python")
                pip_exe = os.path.join(venv_path, "bin", "pip")
            
            # Upgrade pip first
            self.log_step("Upgrading pip...")
            subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Core dependencies for all games
            shared_requirements = [
                "numpy>=1.21.0",
                "torch>=1.11.0", 
                "torchvision>=0.12.0",
                "pillow>=8.3.0",
                "pyyaml>=6.0",
                "matplotlib>=3.5.0",
                "seaborn>=0.11.0",
                "pandas>=1.3.0",
                "plotly>=5.0.0",
                "networkx>=2.6.0",
                "jupyter>=1.0.0",
                "pytest>=6.0.0"
            ]
            
            self.log_step("Installing shared dependencies for all games...")
            for dep in shared_requirements:
                self.log_step(f"Installing {dep.split('>=')[0]}...")
                subprocess.run([pip_exe, "install", dep], check=True, capture_output=True)
            
            self.log_step("All shared dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log_step(f"Error installing dependencies: {e}", "ERROR")
            return False
        except Exception as e:
            self.log_step(f"Unexpected error during installation: {e}", "ERROR")
            return False
    
    def create_shared_data_directory(self) -> bool:
        """Create shared data directory for MNIST and other datasets"""
        try:
            shared_data_dir = "./shared_data"
            os.makedirs(shared_data_dir, exist_ok=True)
            self.log_step(f"Created shared data directory: {shared_data_dir}")
            
            # Create subdirectories
            subdirs = ['mnist', 'models', 'cache', 'temp']
            for subdir in subdirs:
                os.makedirs(os.path.join(shared_data_dir, subdir), exist_ok=True)
                self.log_step(f"Created subdirectory: {shared_data_dir}/{subdir}")
            
            return True
            
        except Exception as e:
            self.log_step(f"Error creating shared directories: {e}", "ERROR")
            return False
    
    def download_shared_mnist(self, venv_path: str) -> bool:
        """Download MNIST data to shared location"""
        try:
            if platform.system() == "Windows":
                python_exe = os.path.join(venv_path, "Scripts", "python.exe")
            else:
                python_exe = os.path.join(venv_path, "bin", "python")
            
            download_script = """
import torchvision
import torchvision.transforms as transforms
import os

shared_data_dir = './shared_data/mnist'
os.makedirs(shared_data_dir, exist_ok=True)

print("Downloading MNIST dataset to shared location...")
transform = transforms.ToTensor()

# Download training set
train_dataset = torchvision.datasets.MNIST(
    root=shared_data_dir, 
    train=True, 
    download=True, 
    transform=transform
)

# Download test set  
test_dataset = torchvision.datasets.MNIST(
    root=shared_data_dir, 
    train=False, 
    download=True, 
    transform=transform
)

print("MNIST dataset downloaded to shared location!")
print("Training samples: " + str(len(train_dataset)))
print("Test samples: " + str(len(test_dataset)))
print("Location: " + shared_data_dir)
"""
            
            self.log_step("Downloading MNIST to shared location...")
            result = subprocess.run([python_exe, "-c", download_script], 
                                  capture_output=True, text=True, check=True, 
                                  encoding='utf-8', errors='replace')
            
            self.log_step("MNIST dataset downloaded to shared location")
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = str(e.stderr).replace('\u2705', '[OK]').replace('\U0001f3ae', '[GAMES]')
            self.log_step(f"Error downloading MNIST: {error_msg}", "ERROR")
            return False
        except Exception as e:
            error_msg = str(e).replace('\u2705', '[OK]').replace('\U0001f3ae', '[GAMES]')
            self.log_step(f"Unexpected error downloading MNIST: {error_msg}", "ERROR")
            return False
    
    def create_activation_scripts(self, venv_path: str) -> bool:
        """Create activation scripts for the unified environment"""
        try:
            if platform.system() == "Windows":
                venv_activate = os.path.join(venv_path, "Scripts", "activate.bat")
                script_content = f"""@echo off
echo Classic Games Environment - Activating...
call "{venv_activate}"
echo.
echo Environment activated: {self.env_name}
echo Shared data: ./shared_data/
echo.
echo Available games (run in separate directories):
echo   sudoku/     - python main.py --action show_kb
echo   kenken/     - python kenken_main.py --action show_kb 
echo   futoshiki/  - python futoshiki_main.py --action show_kb  
echo   kakuro/     - python main.py --action show_kb 
echo   nonogram/   - Coming soon...
echo.
echo Each game project is independent but shares this environment
echo.
cmd /k
"""
                script_file = "activate_classic_games.bat"
            else:
                venv_activate = os.path.join(venv_path, "bin", "activate")
                script_content = f"""#!/bin/bash
echo "Classic Games Environment - Activating..."
source "{venv_activate}"
echo ""
echo "Environment activated: {self.env_name}"
echo "Shared data: ./shared_data/"
echo ""
echo "Available games (run in separate directories):"
echo "  sudoku/     - python main.py --action show_kb"
echo "  kenken/     - python kenken_main.py --action show_kb"
echo "  kakuro/     - Coming soon..."
echo "  nonogram/   - Coming soon..."
echo ""
echo "Each game project is independent but shares this environment"
echo ""
exec "$SHELL"
"""
                script_file = "activate_classic_games.sh"
            
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            if platform.system() != "Windows":
                os.chmod(script_file, 0o755)
            
            self.log_step(f"Created activation script: {script_file}")
            return True
            
        except Exception as e:
            error_msg = str(e).replace('\u2705', '[OK]').replace('\U0001f3ae', '[GAMES]')
            self.log_step(f"Error creating activation scripts: {error_msg}", "ERROR")
            return False
    
    def test_installation(self, venv_path: str) -> bool:
        """Test the unified installation"""
        try:
            if platform.system() == "Windows":
                python_exe = os.path.join(venv_path, "Scripts", "python.exe")
            else:
                python_exe = os.path.join(venv_path, "bin", "python")
            
            test_script = """
import numpy as np
import torch
import torchvision
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("All core imports successful!")
print("NumPy: " + np.__version__)
print("PyTorch: " + torch.__version__)
print("Torchvision: " + torchvision.__version__)

# Test MNIST access
try:
    import torchvision.transforms as transforms
    transform = transforms.ToTensor()
    
    # Try to load from shared location
    dataset = torchvision.datasets.MNIST(
        root='./shared_data/mnist', 
        train=True, 
        download=False,  # Should not need to download
        transform=transform
    )
    print("Shared MNIST accessible: " + str(len(dataset)) + " samples")
except:
    print("Shared MNIST not yet downloaded (run download step)")

print("Environment ready for all games!")
"""
            
            result = subprocess.run([python_exe, "-c", test_script], 
                                  capture_output=True, text=True, check=True,
                                  encoding='utf-8', errors='replace')
            
            self.log_step("Installation test passed")
            # Print output without problematic characters
            output = result.stdout.replace('\u2705', '[OK]').replace('\U0001f3ae', '[GAMES]')
            print(output)
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = str(e.stderr).replace('\u2705', '[OK]').replace('\U0001f3ae', '[GAMES]')
            self.log_step(f"Installation test failed: {error_msg}", "ERROR")
            return False
        except Exception as e:
            error_msg = str(e).replace('\u2705', '[OK]').replace('\U0001f3ae', '[GAMES]')
            self.log_step(f"Error during installation test: {error_msg}", "ERROR")
            return False
    
    def create_environment_info(self, venv_path: str) -> bool:
        """Create environment information file"""
        try:
            env_info = {
                'environment_name': self.env_name,
                'created_date': str(subprocess.check_output(['date'], text=True).strip()) if platform.system() != "Windows" else "N/A",
                'python_version': sys.version,
                'platform': platform.system(),
                'shared_data_location': './shared_data/',
                'supported_games': self.supported_games,
                'setup_log': self.setup_log,
                'usage': {
                    'activation': {
                        'windows': 'activate_classic_games.bat',
                        'linux_mac': 'source activate_classic_games.sh'
                    },
                    'games': {
                        'sudoku': 'cd sudoku && python main.py --action show_kb',
                        'kenken': 'cd kenken && python kenken_main.py --action show_kb'
                    }
                },
                'shared_dependencies': [
                    'numpy', 'torch', 'torchvision', 'pillow', 'pyyaml',
                    'matplotlib', 'seaborn', 'pandas', 'plotly', 'networkx'
                ]
            }
            
            info_file = './classic_games_env_info.json'
            with open(info_file, 'w') as f:
                json.dump(env_info, f, indent=2, default=str)
            
            self.log_step(f"Environment info saved: {info_file}")
            return True
            
        except Exception as e:
            self.log_step(f"Error creating environment info: {e}", "ERROR")
            return False
    
    def detect_existing_projects(self) -> Dict[str, bool]:
        """Detect existing game projects in the current directory"""
        projects = {
            'sudoku': False,
            'kenken': False,
            'main_py_exists': False,
            'kenken_main_exists': False
        }
        
        # Check for existing project structures
        if os.path.exists('sudoku') and os.path.isdir('sudoku'):
            projects['sudoku'] = True
            
        if os.path.exists('kenken') and os.path.isdir('kenken'):
            projects['kenken'] = True
            
        # Check for main files in current directory (indicates mixed setup)
        if os.path.exists('main.py'):
            projects['main_py_exists'] = True
            
        if os.path.exists('kenken_main.py'):
            projects['kenken_main_exists'] = True
        
        return projects
    
    def create_project_template_guide(self) -> bool:
        """Create guide for setting up individual game projects"""
        try:
            # Detect existing project structure
            projects = self.detect_existing_projects()
            
            # Create appropriate guide based on what exists
            if projects['sudoku'] and projects['kenken']:
                guide_content = self.create_guide_for_existing_structure()
            elif projects['main_py_exists'] or projects['kenken_main_exists']:
                guide_content = self.create_guide_for_mixed_structure(projects)
            else:
                guide_content = self.create_guide_for_new_structure()
            
            with open('SETUP_GUIDE.md', 'w', encoding='utf-8') as f:
                f.write(guide_content)
            
            self.log_step("Setup guide created: SETUP_GUIDE.md")
            return True
            
        except Exception as e:
            error_msg = str(e).replace('\U0001f389', '[SUCCESS]').replace('\U0001f3ae', '[GAMES]')
            self.log_step(f"Error creating setup guide: {error_msg}", "ERROR")
            return False
    
    def create_guide_for_existing_structure(self) -> str:
        """Create guide for users who already have organized project structure"""
        return """# Classic Games Project Structure Guide

## Unified Environment Setup Complete!

Great! You already have your game projects well-organized. Your unified environment is ready to use with your existing structure.

## Your Current Structure (Detected)
```
your_workspace/
├── classic_games_env/          # Shared virtual environment (NEW)
├── shared_data/                # Shared MNIST data (NEW)
├── activate_classic_games.bat  # Environment activation (NEW)
├── sudoku/                     # Your existing Sudoku project
│   ├── main.py
│   ├── config.yaml
│   ├── easy_strategies_kb.py
│   └── ...
├── kenken/                     # Your existing KenKen project
│   ├── kenken_main.py
│   ├── config.yaml
│   ├── kenken_easy_strategies_kb.py
│   └── ...
└── [other files]
```

## Ready to Use!

### 1. Activate Environment
```bash
activate_classic_games.bat
```

### 2. Update MNIST Paths (One-time change)

In your existing game files, update MNIST loading to use the shared location:

**In sudoku/ files:**
```python
# OLD:
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

# NEW (use shared MNIST):
train_dataset = torchvision.datasets.MNIST(
    root='../shared_data/mnist', train=True, download=False, transform=transform
)
```

**In kenken/ files:**
```python
# Same change - point to shared MNIST location:
train_dataset = torchvision.datasets.MNIST(
    root='../shared_data/mnist', train=True, download=False, transform=transform
)
```

### 3. Test Your Projects
```bash
# Test Sudoku (should work immediately)
cd sudoku
python main.py --action show_kb

# Test KenKen (should work immediately)  
cd kenken
python kenken_main.py --action show_kb
```

### 4. Benefits You Now Have
- **Shared Environment**: Both games use `classic_games_env`
- **Shared MNIST**: Both games use `../shared_data/mnist/` (60,000 samples)
- **Independent Projects**: Your sudoku/ and kenken/ remain separate
- **All Dependencies**: PyTorch, NumPy, matplotlib, etc. shared

## Environment Details
- **Name**: classic_games_env
- **Python**: 3.8+
- **Shared MNIST**: ./shared_data/mnist/ (60,000 samples)
- **Your Projects**: sudoku/ and kenken/ (unchanged structure)

## Daily Workflow
```bash
# 1. Activate shared environment (once per session)
activate_classic_games.bat

# 2. Work on Sudoku
cd sudoku
python main.py --action full

# 3. Work on KenKen
cd kenken  
python kenken_main.py --action full --grid-size 5
```

Perfect setup! Your projects stay organized and independent while sharing resources.
"""

    def create_guide_for_mixed_structure(self, projects: Dict[str, bool]) -> str:
        """Create guide for users with mixed file structure"""
        migration_steps = []
        
        if projects['main_py_exists']:
            migration_steps.append("# Move Sudoku files to sudoku/ directory")
            migration_steps.append("move main.py sudoku/")
            migration_steps.append("move easy_strategies_kb.py sudoku/")
            migration_steps.append("move moderate_strategies_kb.py sudoku/")
            migration_steps.append("move hard_strategies_kb.py sudoku/")
            migration_steps.append("move sudoku_generator.py sudoku/")
            migration_steps.append("move puzzle_solver.py sudoku/")
            migration_steps.append("move dataset_analyzer.py sudoku/")
            if not projects['sudoku']:
                migration_steps.insert(0, "mkdir sudoku")
        
        if projects['kenken_main_exists']:
            migration_steps.append("\n# Move KenKen files to kenken/ directory")
            migration_steps.append("move kenken_main.py kenken/")
            migration_steps.append("move kenken_easy_strategies_kb.py kenken/")
            migration_steps.append("move kenken_moderate_strategies_kb.py kenken/")
            migration_steps.append("move kenken_hard_strategies_kb.py kenken/")
            migration_steps.append("move kenken_generator.py kenken/")
            migration_steps.append("move kenken_analyzer.py kenken/")
            if not projects['kenken']:
                migration_steps.insert(-6, "mkdir kenken")
        
        migration_commands = "\n".join(migration_steps)
        
        return f"""# Classic Games Project Structure Guide

## Unified Environment Setup Complete!

Your unified environment is ready! However, we detected you have a mixed file structure that could be better organized.

## Current Situation
You have some game files in the root directory. Let's organize them into separate project directories.

## Quick Migration (Optional but Recommended)

```bash
{migration_commands}
```

## After Migration, Your Structure Will Be:
```
your_workspace/
├── classic_games_env/          # Shared virtual environment
├── shared_data/                # Shared MNIST data  
├── activate_classic_games.bat  # Environment activation
├── sudoku/                     # Organized Sudoku project
│   ├── main.py
│   ├── config.yaml
│   └── ...
├── kenken/                     # Organized KenKen project
│   ├── kenken_main.py
│   ├── config.yaml
│   └── ...
└── [other files]
```

## Update MNIST Paths After Migration
In your game files, change MNIST loading:

```python
# NEW (use shared MNIST):
train_dataset = torchvision.datasets.MNIST(
    root='../shared_data/mnist', train=True, download=False, transform=transform
)
```

## Test After Migration
```bash
activate_classic_games.bat

cd sudoku
python main.py --action show_kb

cd kenken
python kenken_main.py --action show_kb
```

Your environment is ready - organization is optional but recommended!
"""

    def create_guide_for_new_structure(self) -> str:
        """Create guide for users starting fresh"""
        return """# Classic Games Project Structure Guide

## Unified Environment Setup Complete!

Your unified environment is ready for new game projects!

## Create Your Game Projects

```bash
# Create directories for your games
mkdir sudoku
mkdir kenken

# Activate environment
activate_classic_games.bat

# Add your game files to respective directories
# sudoku/ - for Sudoku-related files
# kenken/ - for KenKen-related files
```

## Project Structure
```
your_workspace/
├── classic_games_env/          # Shared virtual environment
├── shared_data/                # Shared MNIST data
├── activate_classic_games.bat  # Environment activation
├── sudoku/                     # Your Sudoku project
└── kenken/                     # Your KenKen project
```

## Using Shared MNIST in Your Games
```python
# In your game code, use shared MNIST:
train_dataset = torchvision.datasets.MNIST(
    root='../shared_data/mnist', train=True, download=False, transform=transform
)
```

Ready to build amazing games!
"""
    
    def run_setup(self, skip_confirmation: bool = False, download_mnist: bool = True) -> bool:
        """Run the complete unified environment setup"""
        print("=" * 80)
        print("CLASSIC GAMES - UNIFIED ENVIRONMENT SETUP")
        print("=" * 80)
        print("This will create a SHARED environment for SEPARATE game projects")
        
        if not skip_confirmation:
            print(f"\nThis setup will:")
            print(f"• Create unified virtual environment: {self.env_name}")
            print(f"• Install shared dependencies for all games")
            print(f"• Create shared data directory for MNIST")
            print(f"• Download MNIST dataset (shared)")
            print(f"• Create activation scripts")
            print(f"• Keep each game project completely separate")
            
            response = input("\nProceed with unified environment setup? [Y/n]: ").strip().lower()
            if response and response != 'y' and response != 'yes':
                print("Setup cancelled.")
                return False
        
        # Step 1: Check compatibility
        if not self.check_python_compatibility():
            return False
        
        if not self.check_system_requirements():
            return False
        
        # Step 2: Create unified environment
        success, venv_path = self.create_unified_environment()
        if not success:
            return False
        
        # Step 3: Install shared dependencies
        if not self.install_shared_dependencies(venv_path):
            return False
        
        # Step 4: Create shared data directory
        if not self.create_shared_data_directory():
            return False
        
        # Step 5: Download shared MNIST (optional)
        if download_mnist:
            if not self.download_shared_mnist(venv_path):
                self.log_step("MNIST download failed, but continuing setup", "WARNING")
        
        # Step 6: Create activation scripts
        if not self.create_activation_scripts(venv_path):
            return False
        
        # Step 7: Test installation
        if not self.test_installation(venv_path):
            return False
        
        # Step 8: Create environment info
        if not self.create_environment_info(venv_path):
            return False
        
        # Step 9: Create project guide
        if not self.create_project_template_guide():
            return False
        
        # Success!
        print("\n" + "=" * 80)
        print("UNIFIED ENVIRONMENT SETUP COMPLETED!")
        print("=" * 80)
        
        print(f"\nShared environment created: {self.env_name}")
        print(f"Shared data location: ./shared_data/")
        print(f"Setup guide: SETUP_GUIDE.md")
        
        if platform.system() == "Windows":
            print(f"\nTo activate environment: activate_classic_games.bat")
        else:
            print(f"\nTo activate environment: source activate_classic_games.sh")
        
        print(f"\nNext steps:")
        print(f"  1. Read SETUP_GUIDE.md for project structure")
        print(f"  2. Activate the environment")
        print(f"  3. Update MNIST paths in your game code")
        print(f"  4. Each game uses the shared environment but remains independent")
        
        print(f"\nYour games can now share:")
        print(f"  • Virtual environment: {self.env_name}")
        print(f"  • MNIST dataset: ./shared_data/mnist/")
        print(f"  • All dependencies: numpy, torch, matplotlib, etc.")
        print(f"  • But keep separate: code, configs, outputs")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Classic Games - Unified Environment Setup")
    parser.add_argument('--skip-confirmation', action='store_true',
                       help='Skip setup confirmation prompt')
    parser.add_argument('--no-mnist', action='store_true',
                       help='Skip MNIST download')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check system compatibility')
    
    args = parser.parse_args()
    
    setup = UnifiedEnvironmentSetup()
    
    if args.check_only:
        print("Checking system compatibility...")
        
        python_ok = setup.check_python_compatibility()
        system_ok = setup.check_system_requirements()
        
        if python_ok and system_ok:
            print("System is compatible for unified environment!")
            return 0
        else:
            print("System compatibility issues found.")
            return 1
    
    # Run unified environment setup
    download_mnist = not args.no_mnist
    if setup.run_setup(args.skip_confirmation, download_mnist):
        return 0
    else:
        print("\nUnified environment setup failed. Check error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())