#!/usr/bin/env python3
"""
Final Container Test Script
Verifies all containers have plotly and other required dependencies
"""

import subprocess
import sys
import time

def test_container(container_name, test_command):
    """Test a specific container"""
    print(f"🧪 Testing {container_name}...")
    
    try:
        # Run the test command
        result = subprocess.run(
            f"docker run --rm --name test-{container_name} {container_name}:latest {test_command}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"✅ {container_name}: SUCCESS")
            print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {container_name}: FAILED")
            print(f"   Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {container_name}: TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {container_name}: ERROR - {str(e)}")
        return False

def main():
    """Test all containers"""
    print("🚀 Final Container Test - All Dependencies")
    print("=" * 60)
    
    # Test commands for each container
    tests = [
        ("b2c-investor-platform", "python -c \"import streamlit as st; import pandas as pd; import numpy as np; import plotly.graph_objects as go; print('✅ All imports successful')\""),
        ("inference-container", "python -c \"import streamlit as st; import pandas as pd; import numpy as np; import plotly.graph_objects as go; print('✅ All imports successful')\""),
        ("order-execution-container", "python -c \"import streamlit as st; import pandas as pd; import requests; import plotly.graph_objects as go; print('✅ All imports successful')\""),
        ("data-synthesizer-container", "python -c \"import streamlit as st; import pandas as pd; import numpy as np; from sklearn.ensemble import ExtraTreesClassifier; import plotly.graph_objects as go; print('✅ All imports successful')\""),
        ("training-pipeline-container", "python -c \"import streamlit as st; import pandas as pd; import numpy as np; from sklearn.ensemble import ExtraTreesClassifier; import lightgbm as lgb; import plotly.graph_objects as go; print('✅ All imports successful')\"")
    ]
    
    results = []
    for container_name, test_command in tests:
        success = test_container(container_name, test_command)
        results.append((container_name, success))
        print()  # Add spacing between tests
    
    # Summary
    print("=" * 60)
    print("📊 Final Test Results Summary:")
    
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
        print("\n🎉 ALL CONTAINERS ARE READY FOR DEPLOYMENT!")
        print("\n🚀 To launch the complete system:")
        print("   docker-compose -f docker-compose.v2.3.yml up -d")
        print("\n🌐 Access URLs:")
        print("   B2C Investor: http://localhost:8501")
        print("   Inference: http://localhost:8502")
        print("   Order Execution: http://localhost:8503")
        print("   Data Synthesizer: http://localhost:8504")
        print("   Training Pipeline: http://localhost:8505")
        print("   Prometheus: http://localhost:9090")
        print("   Grafana: http://localhost:3000 (admin/admin)")
        return True
    else:
        print(f"\n⚠️ {failed} containers have issues that need fixing")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
