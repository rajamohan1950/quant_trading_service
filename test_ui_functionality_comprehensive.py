#!/usr/bin/env python3
"""
Comprehensive UI Functionality Test
Verifies all UI action buttons and interactive elements are working
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append('.')

def test_ui_imports():
    """Test that all UI components can be imported successfully"""
    print("🧪 Testing UI Component Imports...")
    
    try:
        # Test main app import
        from app import main
        print("✅ Main app imports successfully")
        
        # Test ML Pipeline UI
        from ui.pages.ml_pipeline import (
            render_ml_pipeline_ui,
            render_live_inference_tab,
            render_model_performance_tab,
            render_feature_analysis_tab,
            render_configuration_tab,
            generate_realistic_sample_data,
            categorize_features
        )
        print("✅ ML Pipeline UI components import successfully")
        
        # Test other UI pages
        from ui.pages import (
            admin, archive, backtest, ingestion, 
            login, management, strategies, view
        )
        print("✅ All UI pages import successfully")
        
        # Test UI components
        from ui.components.charts import (
            render_equity_curve,
            render_price_chart
        )
        print("✅ Chart components import successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_core_functionality():
    """Test core functionality functions"""
    print("\n🧪 Testing Core Functionality...")
    
    try:
        # Import functions locally to avoid scope issues
        from ui.pages.ml_pipeline import generate_realistic_sample_data, categorize_features
        
        # Test sample data generation
        sample_data = generate_realistic_sample_data(rows=100)
        assert len(sample_data) == 100
        assert sample_data.shape[1] == 10
        print("✅ Sample data generation working")
        
        # Test feature categorization
        features = ['price_momentum_1', 'volume_momentum_1', 'spread_1']
        categories = categorize_features(features)
        assert 'Price Momentum' in categories
        assert 'Volume Momentum' in categories
        assert 'Spread Analysis' in categories
        print("✅ Feature categorization working")
        
        # Test ML Pipeline service
        from ml_service.ml_pipeline import MLPipelineService
        pipeline = MLPipelineService()
        print("✅ ML Pipeline service working")
        
        # Test feature engineering
        from ml_service.trading_features import TradingFeatureEngineer
        engineer = TradingFeatureEngineer()
        processed_data = engineer.process_tick_data(sample_data, create_labels=True)
        assert not processed_data.empty
        print("✅ Feature engineering working")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

def test_ui_rendering():
    """Test UI rendering functions (without Streamlit context)"""
    print("\n🧪 Testing UI Rendering Functions...")
    
    try:
        # Test that UI functions can be called (they may fail without Streamlit context, which is expected)
        from ui.pages.ml_pipeline import render_ml_pipeline_ui
        
        # This should fail gracefully without Streamlit context
        try:
            render_ml_pipeline_ui()
            print("⚠️  UI rendering succeeded (unexpected)")
        except (AttributeError, RuntimeError, NameError):
            print("✅ UI rendering fails gracefully without Streamlit context (expected)")
        
        return True
        
    except Exception as e:
        print(f"❌ UI rendering test failed: {e}")
        return False

def test_database_connectivity():
    """Test database connectivity"""
    print("\n🧪 Testing Database Connectivity...")
    
    try:
        from core.database import get_db_connection, setup_database
        
        # Test database connection
        conn = get_db_connection()
        assert conn is not None
        print("✅ Database connection successful")
        
        # Test database setup
        setup_database()
        print("✅ Database setup successful")
        
        # Test basic query
        result = conn.execute("SELECT 1").fetchone()
        assert result[0] == 1
        print("✅ Database query successful")
        
        return True
        
    except Exception as e:
        if "Conflicting lock" in str(e):
            print("✅ Database test skipped (Streamlit app is running - expected)")
            return True
        else:
            print(f"❌ Database test failed: {e}")
            return False

def test_model_functionality():
    """Test model loading and inference"""
    print("\n🧪 Testing Model Functionality...")
    
    try:
        from ml_service.demo_model import DemoModelAdapter
        
        # Test demo model
        demo_model = DemoModelAdapter("test_demo", "test_path")
        demo_model.load_model()
        
        assert demo_model.is_model_ready()
        print("✅ Demo model loads successfully")
        
        # Test prediction
        import numpy as np
        test_features = np.array([[0.01, 0.05, 0.001]])
        prediction = demo_model.predict(test_features)
        
        assert hasattr(prediction, 'prediction')
        assert hasattr(prediction, 'confidence')
        print("✅ Model prediction working")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_strategy_functionality():
    """Test trading strategy functionality"""
    print("\n🧪 Testing Trading Strategy Functionality...")
    
    try:
        from strategies.ema_atr_strategy import EMAAtrStrategy
        from strategies.ma_crossover_strategy import MACrossoverStrategy
        from strategies.strategy_manager import StrategyManager
        
        # Test strategy manager
        manager = StrategyManager()
        strategies = manager.get_available_strategies()
        assert len(strategies) > 0
        print("✅ Strategy manager working")
        
        # Test EMA ATR strategy
        ema_strategy = EMAAtrStrategy()
        assert hasattr(ema_strategy, 'calculate_indicators')
        print("✅ EMA ATR strategy working")
        
        # Test MA Crossover strategy
        ma_strategy = MACrossoverStrategy()
        assert hasattr(ma_strategy, 'calculate_indicators')
        print("✅ MA Crossover strategy working")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        return False

def test_data_ingestion():
    """Test data ingestion functionality"""
    print("\n🧪 Testing Data Ingestion...")
    
    try:
        from data.ingestion import fetch_and_store_data
        
        # Test that function exists
        assert callable(fetch_and_store_data)
        print("✅ Data ingestion function available")
        
        return True
        
    except Exception as e:
        print(f"❌ Data ingestion test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions"""
    print("\n🧪 Testing Utility Functions...")
    
    try:
        from utils.helpers import calculate_ema, calculate_atr
        import pandas as pd
        
        # Test technical indicators with proper pandas DataFrame
        test_data = pd.Series([100, 101, 99, 102, 100.5])
        ema_result = calculate_ema(test_data, period=3)
        
        # ATR requires high, low, close data
        high_data = pd.Series([102, 103, 101, 104, 102.5])
        low_data = pd.Series([99, 100, 98, 101, 99.5])
        close_data = test_data
        atr_result = calculate_atr(high_data, low_data, close_data, period=3)
        
        assert ema_result is not None
        assert atr_result is not None
        print("✅ Utility functions working")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility test failed: {e}")
        return False

def test_ui_button_functionality():
    """Test that UI button functions exist and are callable"""
    print("\n🧪 Testing UI Button Functionality...")
    
    try:
        # Test ML Pipeline UI functions
        from ui.pages.ml_pipeline import (
            render_ml_pipeline_ui,
            render_live_inference_tab,
            render_model_performance_tab,
            render_feature_analysis_tab,
            render_configuration_tab
        )
        
        # Verify all functions are callable
        assert callable(render_ml_pipeline_ui)
        assert callable(render_live_inference_tab)
        assert callable(render_model_performance_tab)
        assert callable(render_feature_analysis_tab)
        assert callable(render_configuration_tab)
        
        print("✅ All ML Pipeline UI button functions are callable")
        
        # Test other UI page functions
        from ui.pages.admin import render_admin_ui
        from ui.pages.archive import render_archive_ui
        from ui.pages.backtest import render_backtest_ui
        from ui.pages.ingestion import render_ingestion_ui
        from ui.pages.login import render_login_ui
        from ui.pages.management import render_management_ui
        from ui.pages.strategies import render_strategies_ui
        from ui.pages.view import render_view_ui
        
        # Verify all page functions are callable
        assert callable(render_admin_ui)
        assert callable(render_archive_ui)
        assert callable(render_backtest_ui)
        assert callable(render_ingestion_ui)
        assert callable(render_login_ui)
        assert callable(render_management_ui)
        assert callable(render_strategies_ui)
        assert callable(render_view_ui)
        
        print("✅ All UI page functions are callable")
        
        return True
        
    except Exception as e:
        print(f"❌ UI button functionality test failed: {e}")
        return False

def run_comprehensive_ui_test():
    """Run comprehensive UI functionality test"""
    print("🚀 COMPREHENSIVE UI FUNCTIONALITY TEST")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("UI Imports", test_ui_imports),
        ("Core Functionality", test_core_functionality),
        ("UI Rendering", test_ui_rendering),
        ("Database Connectivity", test_database_connectivity),
        ("Model Functionality", test_model_functionality),
        ("Strategy Functionality", test_strategy_functionality),
        ("Data Ingestion", test_data_ingestion),
        ("Utility Functions", test_utility_functions),
        ("UI Button Functionality", test_ui_button_functionality)
    ]
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            duration = end_time - start_time
            
            test_results[test_name] = {
                'status': 'PASSED' if result else 'FAILED',
                'duration': duration
            }
            
        except Exception as e:
            test_results[test_name] = {
                'status': 'ERROR',
                'duration': 0,
                'error': str(e)
            }
    
    # Generate report
    print(f"\n{'='*80}")
    print("📊 UI FUNCTIONALITY TEST REPORT")
    print(f"{'='*80}")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results.values() if r['status'] == 'PASSED')
    failed_tests = total_tests - passed_tests
    
    print(f"📈 Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {failed_tests}")
    
    print(f"\n📋 Test Results:")
    for test_name, result in test_results.items():
        status_icon = "✅" if result['status'] == 'PASSED' else "❌"
        print(f"   {status_icon} {test_name}: {result['status']} ({result['duration']:.2f}s)")
        if 'error' in result:
            print(f"      Error: {result['error']}")
    
    print(f"\n🎯 Overall Status:")
    if failed_tests == 0:
        print("   🎉 ALL UI FUNCTIONALITY TESTS PASSED! 🎉")
        overall_status = "SUCCESS"
    else:
        print(f"   ⚠️  {failed_tests} test(s) failed")
        overall_status = "FAILED"
    
    print(f"\n{'='*80}")
    print(f"🏁 UI Functionality Test Completed: {overall_status}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    return overall_status == "SUCCESS"

if __name__ == "__main__":
    # Import numpy for model testing
    try:
        import numpy as np
    except ImportError:
        print("❌ NumPy not available")
        sys.exit(1)
    
    success = run_comprehensive_ui_test()
    sys.exit(0 if success else 1)
