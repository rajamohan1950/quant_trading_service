#!/usr/bin/env python3
"""
Simple Test Runner for ML Pipeline
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print("Error:", e.stderr)
        return False

def main():
    """Main test runner"""
    print("🚀 Starting ML Pipeline Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("ml_service"):
        print("❌ Error: Please run from project root directory")
        sys.exit(1)
    
    # Install test dependencies
    print("\n📦 Installing test dependencies...")
    if not run_command("pip install pytest pytest-cov pytest-mock", "Installing pytest dependencies"):
        sys.exit(1)
    
    # Run tests with coverage
    print("\n🧪 Running tests with coverage...")
    
    # Create test command
    test_cmd = [
        "python", "-m", "pytest", "tests/",
        "-v",
        "--cov=ml_service",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-fail-under=100",
        "--tb=short"
    ]
    
    if not run_command(" ".join(test_cmd), "Running test suite"):
        print("\n❌ Tests failed or coverage below 100%")
        sys.exit(1)
    
    print("\n🎉 All tests passed with 100% coverage!")
    print("📊 Coverage report available at: htmlcov/index.html")
    
    # Try to open coverage report
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", "htmlcov/index.html"])
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", "htmlcov/index.html"])
        elif sys.platform == "win32":
            subprocess.run(["start", "htmlcov/index.html"])
    except:
        print("Could not automatically open coverage report")

if __name__ == "__main__":
    main() 