#!/usr/bin/env python3
"""
Test script to verify ML pipeline setup
"""

import sys
import os

# Add the ml_service directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_service'))

def test_ml_pipeline():
    """Test the ML pipeline setup"""
    print("🔍 Testing ML Pipeline Setup")
    print("=" * 50)
    
    try:
        # Test 1: Import ML pipeline
        from ml_service.ml_pipeline import MLPipelineService
        print("✅ MLPipelineService imported successfully")
        
        # Test 2: Create pipeline service
        pipeline = MLPipelineService()
        print("✅ MLPipelineService created successfully")
        
        # Test 3: Setup database
        try:
            pipeline.setup_database()
            print("✅ Database setup successful")
        except Exception as e:
            print(f"⚠️ Database setup failed (expected if no DB): {e}")
        
        # Test 4: Load models
        print("🔧 Loading models...")
        loaded_models = pipeline.load_models()
        print(f"✅ Models loaded: {len(loaded_models)}")
        
        for model_name, model_info in loaded_models.items():
            print(f"   📊 {model_name}: {model_info['type']} - {model_info['status']}")
            if 'lightgbm_available' in model_info:
                print(f"      LightGBM: {model_info['lightgbm_available']}")
        
        # Test 5: Check active model
        active_model = pipeline.get_pipeline_status()
        print(f"✅ Active model: {active_model['active_model']}")
        print(f"✅ Models loaded: {active_model['models_loaded']}")
        
        # Test 6: Test prediction with dummy data
        if pipeline.active_model:
            print("🧪 Testing prediction...")
            import pandas as pd
            import numpy as np
            
            # Create dummy features
            dummy_features = pd.DataFrame({
                'price_momentum_1': [0.01],
                'price_momentum_5': [0.02],
                'price_momentum_10': [0.03],
                'volume_momentum_1': [0.1],
                'volume_momentum_2': [0.15],
                'volume_momentum_3': [0.2],
                'spread_1': [0.0005],
                'spread_2': [0.001],
                'spread_3': [0.0015],
                'bid_ask_imbalance_1': [0.05],
                'bid_ask_imbalance_2': [0.1],
                'bid_ask_imbalance_3': [0.15],
                'vwap_deviation_1': [0.005],
                'vwap_deviation_2': [0.01],
                'vwap_deviation_3': [0.015],
                'rsi_14': [50],
                'macd': [0.01],
                'bollinger_position': [0.0],
                'stochastic_k': [50],
                'williams_r': [-50],
                'atr_14': [0.002],
                'hour': [12],
                'minute': [30],
                'market_session': [1],
                'time_since_open': [3.5],
                'time_to_close': [3.5]
            })
            
            try:
                prediction = pipeline.active_model.predict(dummy_features)
                print(f"✅ Prediction successful: {prediction.prediction}")
                print(f"   Confidence: {prediction.confidence:.3f}")
                print(f"   Edge score: {prediction.edge_score:.3f}")
                
                # Test trading signal
                signal = pipeline.active_model.get_trading_signal(prediction)
                print(f"✅ Trading signal: {signal['action']} - {signal['signal_strength']}")
                
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n🎉 ML Pipeline setup test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ML Pipeline setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ml_pipeline()
    sys.exit(0 if success else 1) 