#!/usr/bin/env python3
"""
Test script to verify all containers can be imported without errors
This runs on the host to check the code before building containers
"""

import sys
import os

def test_container_import(container_name, app_file):
    """Test if a container can be imported without errors"""
    print(f"🧪 Testing {container_name}...")
    
    try:
        # Add container path to Python path
        container_path = f"containers/{container_name}"
        if container_path not in sys.path:
            sys.path.insert(0, container_path)
        
        # Try to import the app
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", f"{container_path}/{app_file}")
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        
        print(f"✅ {container_name}: SUCCESS - No import errors")
        return True
        
    except Exception as e:
        print(f"❌ {container_name}: FAILED - {str(e)}")
        return False

def main():
    """Test all containers"""
    print("🚀 Testing All Container Imports...")
    print("=" * 50)
    
    containers_to_test = [
        ("inference_container", "app.py"),
        ("order_execution_container", "app.py"),
        ("data_synthesizer_container", "app.py"),
        ("training_pipeline_container", "app.py")
    ]
    
    results = []
    for container_name, app_file in containers_to_test:
        success = test_container_import(container_name, app_file)
        results.append((container_name, success))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    failed = 0
    
    for container_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {container_name}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} containers passed")
    
    if failed == 0:
        print("🎉 All containers are ready for Docker build!")
        return True
    else:
        print("⚠️ Some containers have issues that need fixing before Docker build")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
