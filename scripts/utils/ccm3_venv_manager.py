#!/usr/bin/env python3
"""
CCM3 Virtual Environment Manager
Utilities for keeping the CCM3 virtual environment active during MusicHal 9000 operations

This module provides functions to:
- Ensure CCM3 venv is activated
- Run commands within the CCM3 environment
- Validate environment dependencies
- Handle environment switching for CCM4 operations

@author: Jonas Sjøvaag
@date: 2025-10-29
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional, Dict, List, Union


class CCM3EnvironmentManager:
    """
    Manages CCM3 virtual environment activation and dependency management
    """
    
    def __init__(self, ccm4_root: Optional[str] = None):
        """
        Initialize CCM3 environment manager
        
        Args:
            ccm4_root: Root directory of CCM4 project (auto-detected if None)
        """
        if ccm4_root is None:
            ccm4_root = self._find_ccm4_root()
        
        self.ccm4_root = Path(ccm4_root)
        
        # Check for CCM or CCM3 (prefer CCM if available)
        if (self.ccm4_root / "CCM").exists():
            self.ccm3_venv_path = self.ccm4_root / "CCM"
        else:
            self.ccm3_venv_path = self.ccm4_root / "CCM3"
            
        self.is_windows = platform.system() == "Windows"
        
        # Virtual environment paths
        if self.is_windows:
            self.python_executable = self.ccm3_venv_path / "Scripts" / "python.exe"
            self.pip_executable = self.ccm3_venv_path / "Scripts" / "pip.exe"
            self.activate_script = self.ccm3_venv_path / "Scripts" / "activate.bat"
        else:
            self.python_executable = self.ccm3_venv_path / "bin" / "python"
            self.pip_executable = self.ccm3_venv_path / "bin" / "pip"
            self.activate_script = self.ccm3_venv_path / "bin" / "activate"
    
    def _find_ccm4_root(self) -> str:
        """Find CCM4 root directory from current location"""
        current = Path.cwd()
        
        # Check if we're already in CCM4 directory
        if ((current / "CCM3").exists() or (current / "CCM").exists()) and (current / "MusicHal_9000.py").exists():
            return str(current)
        
        # Search upward for CCM4 directory
        while current.parent != current:
            if ((current / "CCM3").exists() or (current / "CCM").exists()) and (current / "MusicHal_9000.py").exists():
                return str(current)
            current = current.parent
        
        # Default fallback
        return os.getcwd()
    
    def is_ccm3_venv_available(self) -> bool:
        """
        Check if CCM3 virtual environment exists and is properly configured
        
        Returns:
            True if CCM3 venv is available, False otherwise
        """
        return (
            self.ccm3_venv_path.exists() and
            self.python_executable.exists() and
            (self.ccm3_venv_path / "pyvenv.cfg").exists()
        )
    
    def is_ccm3_venv_active(self) -> bool:
        """
        Check if we're currently running in the CCM3 virtual environment
        
        Returns:
            True if CCM3 venv is active, False otherwise
        """
        if not self.is_ccm3_venv_available():
            return False
        
        # Check if current Python executable matches CCM3 venv
        current_python = Path(sys.executable)
        ccm3_python = self.python_executable.resolve()
        
        return current_python.resolve() == ccm3_python
    
    def get_activation_command(self) -> str:
        """
        Get the shell command to activate CCM3 virtual environment
        
        Returns:
            Activation command string
        """
        if self.is_windows:
            return f'"{self.activate_script}"'
        else:
            return f"source {self.activate_script}"
    
    def activate_ccm3_venv(self) -> bool:
        """
        Ensure CCM3 virtual environment is active for current Python session
        
        This modifies sys.path and environment variables to use CCM3 dependencies
        without requiring shell activation.
        
        Returns:
            True if activation successful, False otherwise
        """
        if not self.is_ccm3_venv_available():
            print(f"❌ Virtual environment not found at: {self.ccm3_venv_path}")
            return False
        
        if self.is_ccm3_venv_active():
            print(f"✅ Virtual environment {self.ccm3_venv_path.name} already active")
            return True
        
        try:
            # Add CCM3 site-packages to Python path
            if self.is_windows:
                site_packages = self.ccm3_venv_path / "Lib" / "site-packages"
            else:
                # Find the correct Python version directory
                python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
                site_packages = self.ccm3_venv_path / "lib" / python_version / "site-packages"
            
            if site_packages.exists():
                # Insert at beginning of sys.path to prioritize CCM3 packages
                site_packages_str = str(site_packages)
                if site_packages_str not in sys.path:
                    sys.path.insert(0, site_packages_str)
                    print(f"✅ Added {self.ccm3_venv_path.name} site-packages to Python path: {site_packages_str}")
            
            # Update environment variables
            os.environ['VIRTUAL_ENV'] = str(self.ccm3_venv_path)
            if 'PYTHONHOME' in os.environ:
                del os.environ['PYTHONHOME']
            
            # Update PATH to include CCM3 bin directory
            if self.is_windows:
                bin_dir = self.ccm3_venv_path / "Scripts"
            else:
                bin_dir = self.ccm3_venv_path / "bin"
            
            if str(bin_dir) not in os.environ.get('PATH', ''):
                os.environ['PATH'] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            
            print(f"✅ CCM3 virtual environment activated in current session")
            return True
            
        except Exception as e:
            print(f"❌ Failed to activate CCM3 virtual environment: {e}")
            return False
    
    def run_in_ccm3_venv(self, command: Union[str, List[str]], 
                         capture_output: bool = False, 
                         cwd: Optional[str] = None) -> subprocess.CompletedProcess:
        """
        Run a command in the CCM3 virtual environment
        
        Args:
            command: Command to run (string or list of arguments)
            capture_output: Whether to capture stdout/stderr
            cwd: Working directory for command
            
        Returns:
            subprocess.CompletedProcess result
        """
        if not self.is_ccm3_venv_available():
            raise RuntimeError(f"CCM3 virtual environment not available at: {self.ccm3_venv_path}")
        
        # Ensure command is a list
        if isinstance(command, str):
            # Use shell=True for string commands
            shell_command = command
            use_shell = True
        else:
            # Use the CCM3 Python executable for list commands
            cmd_args = [str(self.python_executable)] + command
            use_shell = False
        
        # Set up environment
        env = os.environ.copy()
        env['VIRTUAL_ENV'] = str(self.ccm3_venv_path)
        if 'PYTHONHOME' in env:
            del env['PYTHONHOME']
        
        # Update PATH
        if self.is_windows:
            bin_dir = self.ccm3_venv_path / "Scripts"
        else:
            bin_dir = self.ccm3_venv_path / "bin"
        env['PATH'] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
        
        # Run command
        if use_shell:
            return subprocess.run(
                shell_command,
                shell=True,
                capture_output=capture_output,
                text=True,
                env=env,
                cwd=cwd or self.ccm4_root
            )
        else:
            return subprocess.run(
                cmd_args,
                capture_output=capture_output,
                text=True,
                env=env,
                cwd=cwd or self.ccm4_root
            )
    
    def install_package(self, package: str) -> bool:
        """
        Install a package in the CCM3 virtual environment
        
        Args:
            package: Package name to install
            
        Returns:
            True if installation successful, False otherwise
        """
        try:
            result = self.run_in_ccm3_venv(["-m", "pip", "install", package], capture_output=True)
            if result.returncode == 0:
                print(f"✅ Successfully installed {package} in CCM3 venv")
                return True
            else:
                print(f"❌ Failed to install {package}: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
            return False
    
    def get_environment_info(self) -> Dict[str, str]:
        """
        Get information about the CCM3 virtual environment
        
        Returns:
            Dictionary with environment information
        """
        info = {
            'ccm4_root': str(self.ccm4_root),
            'ccm3_venv_path': str(self.ccm3_venv_path),
            'python_executable': str(self.python_executable),
            'is_available': str(self.is_ccm3_venv_available()),
            'is_active': str(self.is_ccm3_venv_active()),
            'activation_command': self.get_activation_command()
        }
        
        if self.is_ccm3_venv_available():
            try:
                # Get Python version from CCM3 venv
                result = self.run_in_ccm3_venv(["-c", "import sys; print(sys.version)"], capture_output=True)
                if result.returncode == 0:
                    info['python_version'] = result.stdout.strip()
            except:
                info['python_version'] = "Unknown"
        
        return info
    
    def print_status(self):
        """Print detailed status of CCM3 virtual environment"""
        print("🐍 CCM3 Virtual Environment Status")
        print("=" * 40)
        
        info = self.get_environment_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        if self.is_ccm3_venv_available():
            print("\n📦 Checking key packages...")
            key_packages = ['numpy', 'torch', 'librosa', 'scipy']
            for package in key_packages:
                try:
                    # Use string formatting to avoid f-string nesting issues
                    check_cmd = f"import {package}; print('{package}:', getattr({package}, '__version__', 'installed'))"
                    result = self.run_in_ccm3_venv(["-c", check_cmd], capture_output=True)
                    if result.returncode == 0:
                        print(f"✅ {result.stdout.strip()}")
                    else:
                        print(f"❌ {package}: not installed")
                except Exception:
                    print(f"❌ {package}: check failed")


# Global instance for easy access
_ccm3_env_manager = None

def get_ccm3_env_manager() -> CCM3EnvironmentManager:
    """Get global CCM3 environment manager instance"""
    global _ccm3_env_manager
    if _ccm3_env_manager is None:
        _ccm3_env_manager = CCM3EnvironmentManager()
    return _ccm3_env_manager

def ensure_ccm3_venv_active() -> bool:
    """
    Convenience function to ensure CCM3 virtual environment is active
    
    Returns:
        True if CCM3 venv is active, False otherwise
    """
    return get_ccm3_env_manager().activate_ccm3_venv()

def run_with_ccm3_venv(command: Union[str, List[str]], **kwargs) -> subprocess.CompletedProcess:
    """
    Convenience function to run command in CCM3 virtual environment
    
    Args:
        command: Command to run
        **kwargs: Additional arguments for run_in_ccm3_venv
        
    Returns:
        subprocess.CompletedProcess result
    """
    return get_ccm3_env_manager().run_in_ccm3_venv(command, **kwargs)

def main():
    """Test and demonstrate CCM3 environment management"""
    manager = get_ccm3_env_manager()
    manager.print_status()
    
    print(f"\n🔄 Attempting to activate CCM3 virtual environment...")
    success = manager.activate_ccm3_venv()
    
    if success:
        print("✅ CCM3 virtual environment management ready!")
        print(f"\nTo use in your code:")
        print("from ccm3_venv_manager import ensure_ccm3_venv_active")
        print("ensure_ccm3_venv_active()")
    else:
        print("❌ CCM3 virtual environment setup failed")
        print(f"\nTo create CCM3 venv manually:")
        print(f"python -m venv {manager.ccm3_venv_path}")
        print(f"{manager.get_activation_command()}")

if __name__ == "__main__":
    main()