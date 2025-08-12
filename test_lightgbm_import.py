#!/usr/bin/env python3
"""
Test script to check LightGBM import and create a basic model
"""

import sys
import os

# Add the ml_service directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_service'))

def test_lightgbm_import():
    """Test if LightGBM can be imported"""
    print("🔍 Testing LightGBM import...")
    
    try:
        # Try to import LightGBM from scikit-learn
        from lightgbm.sklearn import LGBMClassifier
        print("✅ LightGBM scikit-learn interface imported successfully")
        
        # Try to create a simple model
        print("🔧 Creating test LightGBM model...")
        model = LGBMClassifier(
            n_estimators=10,
            random_state=42,
            verbose=-1
        )
        print("✅ LightGBM model created successfully")
        
        # Test basic functionality
        import numpy as np
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)
        
        print("🔧 Training test model...")
        model.fit(X, y)
        print("✅ Model training successful")
        
        # Test prediction
        prediction = model.predict(X[:1])
        print(f"✅ Prediction successful: {prediction}")
        
        return True
        
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        return False
    except OSError as e:
        print(f"❌ OSError: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_lightgbm_adapter():
    """Test the LightGBM adapter"""
    print("\n🔍 Testing LightGBM adapter...")
    
    try:
        from lightgbm_adapter import LightGBMAdapter
        print("✅ LightGBM adapter imported successfully")
        
        # Create adapter
        adapter = LightGBMAdapter("test_model", "test_path")
        print("✅ LightGBM adapter created successfully")
        
        # Test model creation
        if adapter.load_model():
            print("✅ Model loaded/created successfully")
            print(f"📊 Model info: {adapter.get_model_info()}")
        else:
            print("❌ Model loading failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing adapter: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 LightGBM Import and Functionality Test")
    print("=" * 50)
    
    # Test 1: Basic import
    import_success = test_lightgbm_import()
    
    # Test 2: Adapter functionality
    adapter_success = test_lightgbm_adapter()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Import Test: {'✅ PASSED' if import_success else '❌ FAILED'}")
    print(f"   Adapter Test: {'✅ PASSED' if adapter_success else '❌ FAILED'}")
    
    if import_success and adapter_success:
        print("\n🎉 All tests passed! LightGBM is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 