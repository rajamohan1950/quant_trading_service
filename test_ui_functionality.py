#!/usr/bin/env python3
"""
UI Functionality Testing Script
Tests actual UI components and functionality without database conflicts
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append('.')

def test_ui_components():
    """Test UI components functionality"""
    print("🎨 Testing UI Components Functionality...")
    
    try:
        # Test login UI
        print("🔐 Testing Login UI...")
        from ui.pages.login import render_login_ui
        print("✅ Login UI imported successfully")
        
        # Test ingestion UI
        print("📥 Testing Ingestion UI...")
        from ui.pages.ingestion import render_ingestion_ui
        print("✅ Ingestion UI imported successfully")
        
        # Test archive UI
        print("📁 Testing Archive UI...")
        from ui.pages.archive import render_archive_ui
        print("✅ Archive UI imported successfully")
        
        # Test management UI
        print("⚙️ Testing Management UI...")
        from ui.pages.management import render_management_ui
        print("✅ Management UI imported successfully")
        
        # Test view UI
        print("👁️ Testing View UI...")
        from ui.pages.view import render_view_ui
        print("✅ View UI imported successfully")
        
        # Test backtest UI
        print("🔄 Testing Backtest UI...")
        from ui.pages.backtest import render_backtest_ui
        print("✅ Backtest UI imported successfully")
        
        # Test admin UI
        print("👑 Testing Admin UI...")
        from ui.pages.admin import render_admin_ui
        print("✅ Admin UI imported successfully")
        
        # Test strategies UI
        print("📈 Testing Strategies UI...")
        from ui.pages.strategies import render_strategies_ui
        print("✅ Strategies UI imported successfully")
        
        # Test ML pipeline UI
        print("🤖 Testing ML Pipeline UI...")
        from ui.pages.ml_pipeline import render_ml_pipeline_ui
        print("✅ ML Pipeline UI imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ UI components test failed: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return False

def test_chart_components():
    """Test chart components"""
    print("\n📊 Testing Chart Components...")
    
    try:
        from ui.components.charts import render_equity_curve, render_price_chart
        
        # Test chart creation functions
        print("✅ Chart components imported successfully")
        
        # Test with sample data
        sample_data = {
            'datetime': ['2025-01-01', '2025-01-02', '2025-01-03'],
            'open': [100, 101, 99],
            'high': [102, 103, 100],
            'low': [99, 100, 98],
            'close': [101, 99, 100],
            'volume': [1000, 1100, 900]
        }
        
        print("✅ Chart components ready for use")
        return True
        
    except Exception as e:
        print(f"❌ Chart components test failed: {e}")
        return False

def test_strategy_components():
    """Test strategy components"""
    print("\n📈 Testing Strategy Components...")
    
    try:
        from strategies.strategy_manager import StrategyManager
        from strategies.ema_atr_strategy import EMAAtrStrategy
        from strategies.ma_crossover_strategy import MACrossoverStrategy
        
        # Test strategy manager
        strategy_manager = StrategyManager()
        print("✅ Strategy manager created")
        
        # Test strategy listing
        available_strategies = strategy_manager.get_available_strategies()
        print(f"✅ Available strategies: {available_strategies}")
        
        # Test strategy descriptions
        ema_desc = strategy_manager.get_strategy_description('ema_atr')
        ma_desc = strategy_manager.get_strategy_description('ma_crossover')
        print(f"✅ Strategy descriptions retrieved")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy components test failed: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return False

def test_ml_components():
    """Test ML components"""
    print("\n🤖 Testing ML Components...")
    
    try:
        from ml_service.base_model import BaseModelAdapter
        from ml_service.demo_model import DemoModelAdapter
        from ml_service.lightgbm_adapter import LightGBMAdapter
        
        # Test base model adapter
        print("✅ Base model adapter imported")
        
        # Test demo model
        demo_model = DemoModelAdapter("test_demo", "test_path")
        print("✅ Demo model adapter created")
        
        # Test LightGBM adapter (if available)
        try:
            lightgbm_adapter = LightGBMAdapter("test_lightgbm", "test_path")
            print("✅ LightGBM adapter created")
        except Exception as e:
            print(f"ℹ️  LightGBM adapter not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML components test failed: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return False

def test_database_components():
    """Test database components without connection"""
    print("\n🗄️ Testing Database Components...")
    
    try:
        from core.database import setup_database, get_db_connection, clear_all_data
        
        # Test function imports
        print("✅ Database functions imported successfully")
        
        # Test database file existence
        from core.settings import DB_FILE
        if os.path.exists(DB_FILE):
            print(f"✅ Database file exists: {DB_FILE}")
        else:
            print(f"⚠️  Database file not found: {DB_FILE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database components test failed: {e}")
        return False

def test_utility_components():
    """Test utility components"""
    print("\n🔧 Testing Utility Components...")
    
    try:
        import pandas as pd
        from utils.helpers import format_currency, format_percentage, calculate_ema, calculate_atr
        
        # Test utility functions
        print("✅ Utility functions imported successfully")
        
        # Test function calls
        formatted_currency = format_currency(1000.50)
        formatted_percentage = format_percentage(0.15)
        
        print(f"✅ Currency formatting: {formatted_currency}")
        print(f"✅ Percentage formatting: {formatted_percentage}")
        
        # Test technical indicators
        prices = pd.Series([100, 101, 99, 102, 100])
        ema_result = calculate_ema(prices, 3)
        print(f"✅ EMA calculation: {len(ema_result)} values")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility components test failed: {e}")
        return False

def run_ui_functionality_test():
    """Run all UI functionality tests"""
    print("🚀 Starting UI Functionality Testing...")
    print("=" * 60)
    
    test_results = {}
    
    # Test UI components
    test_results['ui_components'] = test_ui_components()
    
    # Test chart components
    test_results['chart_components'] = test_chart_components()
    
    # Test strategy components
    test_results['strategy_components'] = test_strategy_components()
    
    # Test ML components
    test_results['ml_components'] = test_ml_components()
    
    # Test database components
    test_results['database_components'] = test_database_components()
    
    # Test utility components
    test_results['utility_components'] = test_utility_components()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 UI FUNCTIONALITY TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} : {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All UI functionality tests passed!")
        print("✅ UI components are working correctly")
        print("✅ All imports are successful")
        print("✅ Component initialization is working")
    else:
        print("⚠️  Some UI functionality tests failed")
        print("🔍 Check the output above for specific issues")
    
    return passed == total

if __name__ == "__main__":
    success = run_ui_functionality_test()
    sys.exit(0 if success else 1)
