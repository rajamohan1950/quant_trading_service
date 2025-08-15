#!/usr/bin/env python3
"""
Comprehensive UI Component Testing Script
Tests all UI components, buttons, and functionality before launch
"""

import sys
import os
import requests
import time
import json

# Add ml_service to path
sys.path.append('ml_service')

from ml_service.ml_pipeline import MLPipelineService
from ml_service.demo_model import DemoModelAdapter
import pandas as pd

def test_ml_pipeline_service():
    """Test the ML Pipeline Service core functionality"""
    print("🧪 Testing ML Pipeline Service...")
    
    try:
        # Create pipeline service with test database
        pipeline = MLPipelineService(db_file="stock_data.duckdb")
        print("✅ Pipeline service created")
        
        # Setup database
        pipeline.setup_database()
        print("✅ Database setup completed")
        
        # Load models
        loaded_models = pipeline.load_models()
        print(f"✅ Models loaded: {loaded_models}")
        
        # Check active model
        if pipeline.active_model:
            print(f"✅ Active model: {pipeline.active_model.model_name}")
            print(f"✅ Model ready: {pipeline.active_model.is_model_ready()}")
        else:
            print("❌ No active model")
            return False
        
        # Cleanup test database
        pipeline.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline service test failed: {e}")
        return False

def test_demo_model():
    """Test the demo model functionality"""
    print("\n🧪 Testing Demo Model...")
    
    try:
        # Create and load demo model
        demo_model = DemoModelAdapter("test_demo", "test_path")
        print(f"✅ Demo model created: {demo_model.model_name}")
        print(f"✅ Initial state - is_loaded: {demo_model.is_loaded}, model: {demo_model.model}")
        
        success = demo_model.load_model()
        print(f"✅ Load model result: {success}")
        print(f"✅ After load - is_loaded: {demo_model.is_loaded}, model: {demo_model.model}")
        print(f"✅ Model ready: {demo_model.is_model_ready()}")
        
        if success and demo_model.is_model_ready():
            print("✅ Demo model loaded and ready")
            
            # Test prediction
            sample_features = pd.DataFrame([{
                'price_momentum_1': 0.01,
                'volume_momentum_1': 0.05,
                'spread_1': 0.001
            }])
            
            prediction = demo_model.predict(sample_features)
            print(f"✅ Prediction successful: {prediction.prediction}")
            
            # Test trading signal
            signal = demo_model.get_trading_signal(prediction)
            print(f"✅ Trading signal generated: {signal['action']}")
            
            return True
        else:
            print("❌ Demo model not ready")
            return False
            
    except Exception as e:
        print(f"❌ Demo model test failed: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return False

def test_inference_pipeline():
    """Test the complete inference pipeline"""
    print("\n🧪 Testing Inference Pipeline...")
    
    try:
        # Create pipeline with test database
        pipeline = MLPipelineService(db_file="stock_data.duckdb")
        pipeline.setup_database()
        pipeline.load_models()
        
        # Create sample tick data
        sample_ticks = pd.DataFrame([{
            'price': 100.0,
            'volume': 1000,
            'bid': 99.95,
            'ask': 100.05,
            'bid_qty1': 500,
            'ask_qty1': 300,
            'tick_generated_at': '2025-08-13T18:00:00',
            'symbol': 'TEST'
        }])
        
        # Run inference
        result = pipeline.run_inference_pipeline(sample_ticks)
        
        if result['pipeline_status'] == 'success':
            print("✅ Inference pipeline successful")
            print(f"✅ Prediction: {result['prediction'].prediction}")
            print(f"✅ Signal: {result['signal']['action']}")
            
            # Cleanup test database
            pipeline.cleanup()
            return True
        else:
            print(f"❌ Inference pipeline failed: {result.get('error', 'Unknown error')}")
            pipeline.cleanup()
            return False
            
    except Exception as e:
        print(f"❌ Inference pipeline test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\n🧪 Testing Feature Engineering...")
    
    try:
        from ml_service.trading_features import TradingFeatureEngineer
        
        # Create feature engineer
        feature_engineer = TradingFeatureEngineer()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'price': [100.0, 100.1, 100.2],
            'volume': [1000, 1100, 1200],
            'bid': [99.95, 100.05, 100.15],
            'ask': [100.05, 100.15, 100.25],
            'bid_qty1': [500, 550, 600],
            'ask_qty1': [300, 350, 400],
            'tick_generated_at': ['2025-08-13T18:00:00', '2025-08-13T18:01:00', '2025-08-13T18:02:00'],
            'symbol': ['TEST', 'TEST', 'TEST']
        })
        
        # Process features
        features = feature_engineer.process_tick_data(sample_data, create_labels=False)
        
        if not features.empty:
            print(f"✅ Features generated: {len(features)} rows, {features.shape[1]} columns")
            print(f"✅ Feature columns: {list(features.columns[:5])}...")
            return True
        else:
            print("❌ No features generated")
            return False
            
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_streamlit_app():
    """Test if Streamlit app is accessible"""
    print("\n🧪 Testing Streamlit App Accessibility...")
    
    try:
        # Wait for app to be ready
        time.sleep(5)
        
        # Test main page
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("✅ Streamlit app accessible")
            
            # Check for key UI elements
            content = response.text.lower()
            if 'ml pipeline' in content or 'trading signals' in content:
                print("✅ ML Pipeline UI elements found")
                return True
            else:
                print("⚠️ ML Pipeline UI elements not found in main page")
                return False
        else:
            print(f"❌ Streamlit app returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Streamlit app not accessible: {e}")
        return False
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🚀 Starting Comprehensive UI Testing...")
    print("=" * 50)
    
    test_results = {}
    
    # Test 1: ML Pipeline Service
    test_results['ml_pipeline_service'] = test_ml_pipeline_service()
    
    # Test 2: Demo Model
    test_results['demo_model'] = test_demo_model()
    
    # Test 3: Inference Pipeline
    test_results['inference_pipeline'] = test_inference_pipeline()
    
    # Test 4: Feature Engineering
    test_results['feature_engineering'] = test_feature_engineering()
    
    # Test 5: Streamlit App
    test_results['streamlit_app'] = test_streamlit_app()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready for launch!")
        return True
    else:
        print("⚠️ Some tests failed. Please fix issues before launch.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1) 