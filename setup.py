#!/usr/bin/env python3
"""
Setup script for Apache-Licensed Summarizers Repository

This script helps users set up the environment and test the installation.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def setup_environment():
    """Set up the Python environment."""
    print("\n🚀 Setting up Apache-Licensed Summarizers Repository")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("\n💡 Try running: pip install --upgrade pip")
        return False
    
    # Run tests
    if not run_command("python tests.py", "Running test suite"):
        print("\n⚠️  Some tests failed, but the basic setup should work")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Try the interactive summarizer: python interactive_summarizer.py")
    print("2. Test with a URL: python demo_summarizer.py 'https://your-url.com'")
    print("3. Run benchmarks: python benchmark_summarizers.py")
    print("4. Check examples in the examples/ directory")
    
    return True


if __name__ == "__main__":
    try:
        success = setup_environment()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {str(e)}")
        sys.exit(1)
