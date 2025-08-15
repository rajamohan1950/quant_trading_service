#!/usr/bin/env python3
"""
ML Pipeline UI Page
Provides interface for model selection, inference, and signal generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import sys

# Add ml_service to path
sys.path.append('ml_service')

from ml_service.ml_pipeline import MLPipelineService
from ml_service.trading_features import TradingFeatureEngineer

def generate_realistic_sample_data(rows=1000, price_range=(50.0, 200.0), volatility=2.0):
    """Generate realistic sample tick data for testing"""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Input validation
    if rows <= 0:
        raise ValueError("Rows must be positive")
    if price_range[0] >= price_range[1]:
        raise ValueError("Price range min must be less than max")
    if volatility < 0:
        raise ValueError("Volatility must be non-negative")
    
    # Generate base price series with realistic movement
    base_price = np.random.uniform(price_range[0], price_range[1])
    prices = [base_price]
    
    for i in range(1, rows):
        # Add random walk with volatility
        change = np.random.normal(0, volatility / 100)
        new_price = prices[-1] * (1 + change)
        new_price = max(price_range[0] * 0.5, min(price_range[1] * 1.5, new_price))
        prices.append(new_price)
    
    # Generate realistic bid/ask spreads
    spreads = np.random.uniform(0.01, 0.1, rows)
    bids = [p - s/2 for p, s in zip(prices, spreads)]
    asks = [p + s/2 for p, s in zip(prices, spreads)]
    
    # Generate volume data
    volumes = np.random.randint(100, 10000, rows)
    
    # Generate order book data
    bid_qty1 = np.random.randint(100, 1000, rows)
    ask_qty1 = np.random.randint(100, 1000, rows)
    bid_qty2 = np.random.randint(200, 2000, rows)
    ask_qty2 = np.random.randint(200, 2000, rows)
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(hours=rows//60)
    timestamps = [start_time + timedelta(minutes=i) for i in range(rows)]
    
    # Create DataFrame
    data = pd.DataFrame({
        'price': prices,
        'volume': volumes,
        'bid': bids,
        'ask': asks,
        'bid_qty1': bid_qty1,
        'ask_qty1': ask_qty1,
        'bid_qty2': bid_qty2,
        'ask_qty2': ask_qty2,
        'tick_generated_at': timestamps,
        'symbol': 'SAMPLE'
    })
    
    return data

def categorize_features(features):
    """Categorize features into logical groups"""
    categories = {
        'Price Momentum': 0,
        'Volume Momentum': 0,
        'Spread Analysis': 0,
        'Bid-Ask Imbalance': 0,
        'VWAP Deviation': 0,
        'Technical Indicators': 0,
        'Time Features': 0,
        'Other': 0
    }
    
    for feature in features:
        if 'momentum' in feature.lower():
            if 'price' in feature.lower():
                categories['Price Momentum'] += 1
            elif 'volume' in feature.lower():
                categories['Volume Momentum'] += 1
        elif 'spread' in feature.lower():
            categories['Spread Analysis'] += 1
        elif 'imbalance' in feature.lower():
            categories['Bid-Ask Imbalance'] += 1
        elif 'vwap' in feature.lower():
            categories['VWAP Deviation'] += 1
        elif any(indicator in feature.lower() for indicator in ['rsi', 'macd', 'bollinger', 'stochastic', 'williams', 'atr']):
            categories['Technical Indicators'] += 1
        elif any(time_feature in feature.lower() for time_feature in ['hour', 'minute', 'time', 'session']):
            categories['Time Features'] += 1
        else:
            categories['Other'] += 1
    
    # Remove categories with 0 features
    return {k: v for k, v in categories.items() if v > 0}

def render_ml_pipeline_ui():
    """Render the ML Pipeline UI"""
    st.title("🤖 ML Pipeline - Trading Signals")
    st.markdown("---")
    
    # Initialize session state
    if 'ml_pipeline' not in st.session_state:
        st.session_state.ml_pipeline = None
        st.session_state.pipeline_initialized = False
    
    # Initialize ML Pipeline
    if not st.session_state.pipeline_initialized:
        try:
            with st.spinner("🚀 Initializing ML Pipeline..."):
                st.write("🔧 Creating ML Pipeline Service...")
                ml_pipeline = MLPipelineService()
                
                st.write("🗄️ Setting up database...")
                ml_pipeline.setup_database()
                
                st.write("🤖 Loading models...")
                loaded_models = ml_pipeline.load_models()
                st.write(f"📊 Models loaded: {loaded_models}")
                
                st.session_state.ml_pipeline = ml_pipeline
                st.session_state.pipeline_initialized = True
            st.success("✅ ML Pipeline initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize ML Pipeline: {e}")
            st.error(f"❌ Error details: {str(e)}")
            import traceback
            st.error(f"❌ Traceback: {traceback.format_exc()}")
            return
    
    ml_pipeline = st.session_state.ml_pipeline
    
    # Sidebar for model selection and controls
    with st.sidebar:
        st.header("🎛️ Pipeline Controls")
        
        # Model selection
        if ml_pipeline.models:
            available_models = list(ml_pipeline.models.keys())
            selected_model = st.selectbox(
                "Select Active Model:",
                available_models,
                index=available_models.index(ml_pipeline.active_model.model_name) if ml_pipeline.active_model else 0
            )
            
            if st.button("🔄 Switch Model", key="switch_model_btn"):
                if ml_pipeline.set_active_model(selected_model):
                    st.success(f"✅ Switched to {selected_model}")
                    st.experimental_rerun()
                else:
                    st.error("❌ Failed to switch model")
        else:
            st.warning("⚠️ No models loaded")
        
        # Pipeline status
        st.subheader("📊 Pipeline Status")
        status = ml_pipeline.get_pipeline_status()
        
        st.metric("Models Loaded", status['models_loaded'])
        st.metric("Inference Count", status['inference_count'])
        st.metric("Avg Inference Time", f"{status['avg_inference_time']:.4f}s")
        
        if status['active_model']:
            st.success(f"🎯 Active: {status['active_model']}")
        else:
            st.error("❌ No active model")
        
        # Database status
        if status['database_connected']:
            st.success("🗄️ Database Connected")
        else:
            st.error("❌ Database Disconnected")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Inference", "📊 Model Performance", "🔍 Feature Analysis", "⚙️ Configuration"])
    
    with tab1:
        render_live_inference_tab(ml_pipeline)
    
    with tab2:
        render_model_performance_tab(ml_pipeline)
    
    with tab3:
        render_feature_analysis_tab(ml_pipeline)
    
    with tab4:
        render_configuration_tab(ml_pipeline)

def render_live_inference_tab(ml_pipeline):
    """Render the live inference tab"""
    st.header("🎯 Live Trading Signal Inference")
    
    # Check if pipeline is ready
    if not ml_pipeline.active_model:
        st.error("❌ No active model selected. Please load a model first.")
        return
    
    # Input section
    st.subheader("📥 Input Data")
    
    # Option 1: Manual input
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Manual Input**")
        price = st.number_input("Price", value=100.0, step=0.01, key="manual_price")
        volume = st.number_input("Volume", value=1000, step=100, key="manual_volume")
        bid = st.number_input("Bid", value=99.95, step=0.01, key="manual_bid")
        ask = st.number_input("Ask", value=100.05, step=0.01, key="manual_ask")
    
    with col2:
        st.write("**Order Book Data**")
        bid_qty1 = st.number_input("Bid Qty L1", value=500, step=100, key="manual_bid_qty1")
        ask_qty1 = st.number_input("Ask Qty L1", value=300, step=100, key="manual_ask_qty1")
        bid_qty2 = st.number_input("Bid Qty L2", value=800, step=100, key="manual_bid_qty2")
        ask_qty2 = st.number_input("Ask Qty L2", value=600, step=100, key="manual_ask_qty2")
    
    # Create sample tick data
    sample_data = pd.DataFrame([{
        'price': price,
        'volume': volume,
        'bid': bid,
        'ask': ask,
        'bid_qty1': bid_qty1,
        'ask_qty1': ask_qty1,
        'bid_qty2': bid_qty2,
        'ask_qty2': ask_qty2,
        'tick_generated_at': datetime.now().isoformat(),
        'symbol': 'SAMPLE'
    }])
    
    # Run inference button
    if st.button("🚀 Run Inference", key="run_inference_btn"):
        with st.spinner("🔄 Processing..."):
            try:
                # Run inference pipeline
                result = ml_pipeline.run_inference_pipeline(sample_data)
                
                if result['pipeline_status'] == 'success':
                    st.success("✅ Inference completed successfully!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        prediction = result['prediction']
                        st.metric("Prediction", prediction.prediction)
                        st.metric("Confidence", f"{prediction.confidence:.1%}")
                    
                    with col2:
                        signal = result['signal']
                        st.metric("Signal Strength", signal.get('signal_strength', 'Unknown'))
                        st.metric("Risk Level", signal.get('risk_level', 'Unknown'))
                    
                    with col3:
                        st.metric("Edge Score", f"{prediction.edge_score:.3f}")
                        st.metric("Position Size", f"{signal.get('position_size', 0):.1%}")
                    
                    # Display inference timing with microsecond precision
                    if 'inference_time' in result:
                        inference_time = result['inference_time']
                        if isinstance(inference_time, (int, float)):
                            # Convert to microseconds if in seconds
                            if inference_time < 1:
                                micro_time = inference_time * 1000000
                                st.metric("Inference Time", f"{micro_time:.2f} μs")
                            else:
                                st.metric("Inference Time", f"{inference_time:.6f}s")
                    
                    # Display detailed signal information
                    st.subheader("📋 Trading Signal Details")
                    
                    # Signal information
                    st.write("**Signal Information:**")
                    signal_info = {
                        'Action': signal.get('action', 'Unknown'),
                        'Confidence': f"{signal.get('confidence', 0):.1%}",
                        'Edge Score': f"{signal.get('edge_score', 0):.3f}",
                        'Signal Strength': signal.get('signal_strength', 'Unknown'),
                        'Risk Level': signal.get('risk_level', 'Unknown'),
                        'Position Size': f"{signal.get('position_size', 0):.1%}",
                        'Model Type': signal.get('model_type', 'Unknown')
                    }
                    
                    signal_df = pd.DataFrame(list(signal_info.items()), columns=['Property', 'Value'])
                    st.dataframe(signal_df)
                    
                    # Feature values
                    if 'feature_values' in signal:
                        st.write("**Key Feature Values:**")
                        feature_df = pd.DataFrame(list(signal['feature_values'].items()), 
                                                columns=['Feature', 'Value'])
                        st.dataframe(feature_df)
                    
                    # Probabilities
                    st.write("**Prediction Probabilities:**")
                    prob_df = pd.DataFrame(list(prediction.probabilities.items()), 
                                         columns=['Class', 'Probability'])
                    prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(prob_df)
                    
                    # Store result in session state for other tabs
                    st.session_state.last_inference_result = result
                    
                else:
                    st.error(f"❌ Inference failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error during inference: {e}")
    
    # Option 2: Load from database
    st.subheader("📊 Load Data from Database")
    
    if st.button("📥 Load Recent Ticks", key="load_ticks_btn"):
        try:
            # Check if database is connected
            if not ml_pipeline.db_conn:
                st.error("❌ Database not connected. Please check the configuration.")
                return
                
            # Load recent tick data from database
            query = """
                SELECT * FROM tick_data
                WHERE tick_generated_at >= now() - INTERVAL '1 hour'
                ORDER BY tick_generated_at DESC
                LIMIT 100
            """
            recent_ticks = ml_pipeline.db_conn.execute(query).fetchdf()
            
            if not recent_ticks.empty:
                st.success(f"✅ Loaded {len(recent_ticks)} recent ticks")
                
                # Display sample
                st.write("**Recent Tick Data Sample:**")
                st.dataframe(recent_ticks.head(10))
                
                # Store for inference
                st.session_state.recent_ticks = recent_ticks
                
                if st.button("🚀 Run Inference on Recent Data", key="run_recent_inference_btn"):
                    with st.spinner("🔄 Processing recent data..."):
                        result = ml_pipeline.run_inference_pipeline(recent_ticks)
                        
                        if result['pipeline_status'] == 'success':
                            st.success("✅ Inference on recent data completed!")
                            st.session_state.last_inference_result = result
                        else:
                            st.error(f"❌ Inference failed: {result.get('error', 'Unknown error')}")
            else:
                st.warning("⚠️ No recent tick data found")
                
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")

def render_model_performance_tab(ml_pipeline):
    """Render the model performance tab"""
    st.header("📊 Model Performance & Evaluation")
    
    if not ml_pipeline.models:
        st.warning("⚠️ No models loaded for performance analysis")
        return
    
    # Model selection for evaluation
    model_to_evaluate = st.selectbox(
        "Select Model for Evaluation:",
        list(ml_pipeline.models.keys()),
        key="eval_model_select"
    )
    
    # Data source selection
    st.subheader("📊 Test Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Database (if available)", "Generate Sample Data"],
        key="data_source_radio"
    )
    
    if data_source == "Generate Sample Data":
        # Sample data generation controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_rows = st.number_input(
                "Number of Rows:",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key="sample_rows_input"
            )
        
        with col2:
            price_range = st.slider(
                "Price Range ($):",
                min_value=10.0,
                max_value=1000.0,
                value=(50.0, 200.0),
                step=10.0,
                key="price_range_slider"
            )
        
        with col3:
            volatility = st.slider(
                "Volatility (%):",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                key="volatility_slider"
            )
        
        # Generate sample data button
        if st.button("🔧 Generate Sample Data", key="generate_sample_data_btn"):
            with st.spinner("🔄 Generating sample data..."):
                try:
                    # Generate realistic sample data
                    sample_data = generate_realistic_sample_data(
                        rows=sample_rows,
                        price_range=price_range,
                        volatility=volatility
                    )
                    
                    st.success(f"✅ Generated {len(sample_data)} sample records")
                    st.session_state.sample_test_data = sample_data
                    
                    # Show sample preview
                    st.write("**Sample Data Preview:**")
                    st.dataframe(sample_data.head(10))
                    
                except Exception as e:
                    st.error(f"❌ Error generating sample data: {e}")
    
    # Evaluation button
    if st.button("📈 Evaluate Model Performance", key="eval_model_btn"):
        with st.spinner("🔄 Evaluating model performance..."):
            try:
                test_data = None
                
                if data_source == "Database (if available)" and ml_pipeline.db_conn:
                    # Try to load from database
                    query = """
                        SELECT * FROM tick_data 
                        WHERE tick_generated_at >= now() - INTERVAL '24 hours'
                        ORDER BY tick_generated_at DESC 
                        LIMIT 1000
                    """
                    test_data = ml_pipeline.db_conn.execute(query).fetchdf()
                    
                    if test_data.empty:
                        st.warning("⚠️ No database data available. Please generate sample data instead.")
                        return
                elif data_source == "Generate Sample Data":
                    # Use generated sample data
                    if 'sample_test_data' in st.session_state:
                        test_data = st.session_state.sample_test_data
                    else:
                        st.error("❌ Please generate sample data first")
                        return
                
                if test_data is not None and not test_data.empty:
                    # Process test data with features and labels
                    feature_engineer = TradingFeatureEngineer()
                    processed_data = feature_engineer.process_tick_data(test_data, create_labels=True)
                    
                    if not processed_data.empty and 'trading_label_encoded' in processed_data.columns:
                        # For model evaluation, we need only the features (not labels)
                        # The model expects exactly 26 features
                        feature_columns = [col for col in processed_data.columns 
                                         if col not in ['trading_label', 'trading_label_encoded']]
                        
                        # Ensure we have exactly 26 features for the model
                        if len(feature_columns) == 26:
                            # Prepare data for evaluation (features only)
                            features_for_evaluation = processed_data[feature_columns].copy()
                            
                            # Evaluate model
                            evaluation_result = ml_pipeline.evaluate_model_performance(
                                model_to_evaluate, processed_data
                            )
                        else:
                            st.error(f"❌ Feature mismatch: Expected 26 features, got {len(feature_columns)}")
                            st.info("ℹ️ This usually happens when the feature engineering process changes")
                            return
                        
                        if 'error' not in evaluation_result:
                            st.success("✅ Model evaluation completed!")
                            
                            # Display metrics with microsecond precision
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                metrics = evaluation_result.get('metrics', {})
                                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                                st.metric("Macro-F1", f"{metrics.get('macro_f1', 0):.3f}")
                                st.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
                            
                            with col2:
                                st.metric("Training Samples", metrics.get('training_samples', 0))
                                st.metric("Validation Samples", metrics.get('validation_samples', 0))
                                st.metric("Model Type", evaluation_result.get('model_name', 'Unknown'))
                            
                            with col3:
                                # Per-class metrics summary
                                st.write("**Per-Class F1 Scores:**")
                                f1_scores = metrics.get('f1_score', {})
                                for class_name in ['HOLD', 'BUY', 'SELL']:
                                    if class_name in f1_scores:
                                        f1_val = f1_scores[class_name]
                                        color = "🟢" if f1_val >= 0.7 else "🟡" if f1_val >= 0.5 else "🔴"
                                        st.write(f"{color} {class_name}: {f1_val:.3f}")
                                    else:
                                        st.write(f"⚪ {class_name}: N/A")
                            
                            # Feature importance chart
                            st.subheader("🔍 Feature Importance")
                            feature_importance = evaluation_result.get('feature_importance', {})
                            
                            if feature_importance:
                                # Top 15 features
                                top_features = dict(sorted(feature_importance.items(), 
                                                          key=lambda x: x[1], reverse=True)[:15])
                                
                                fig = px.bar(
                                    x=list(top_features.values()),
                                    y=list(top_features.keys()),
                                    orientation='h',
                                    title="Top 15 Most Important Features"
                                )
                                fig.update_layout(height=600)
                                st.plotly_chart(fig)
                            else:
                                st.warning("⚠️ No feature importance data available")
                            
                            # Detailed per-class metrics
                            st.subheader("📊 Detailed Per-Class Metrics")
                            detailed_metrics = pd.DataFrame({
                                'Class': ['HOLD', 'BUY', 'SELL'],
                                'Precision': [metrics.get('precision', {}).get('HOLD', 0), 
                                            metrics.get('precision', {}).get('BUY', 0), 
                                            metrics.get('precision', {}).get('SELL', 0)],
                                'Recall': [metrics.get('recall', {}).get('HOLD', 0), 
                                         metrics.get('recall', {}).get('BUY', 0), 
                                         metrics.get('recall', {}).get('SELL', 0)],
                                'F1-Score': [metrics.get('f1_score', {}).get('HOLD', 0), 
                                           metrics.get('f1_score', {}).get('BUY', 0), 
                                           metrics.get('f1_score', {}).get('SELL', 0)]
                            })
                            
                            # Add color coding to the dataframe
                            def color_f1_score(val):
                                if val >= 0.7:
                                    return 'background-color: #d4edda'  # Green
                                elif val >= 0.5:
                                    return 'background-color: #fff3cd'  # Yellow
                                else:
                                    return 'background-color: #f8d7da'  # Red
                            
                            styled_metrics = detailed_metrics.style.map(
                                color_f1_score, subset=['F1-Score']
                            ).format({
                                'Precision': '{:.3f}',
                                'Recall': '{:.3f}',
                                'F1-Score': '{:.3f}'
                            })
                            
                            st.dataframe(styled_metrics)
                            
                            # Metrics interpretation
                            st.subheader("📈 Metrics Interpretation")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Macro-F1 Score:**")
                                macro_f1_val = metrics['macro_f1']
                                if macro_f1_val >= 0.7:
                                    st.success(f"Excellent performance: {macro_f1_val:.3f}")
                                elif macro_f1_val >= 0.5:
                                    st.warning(f"Good performance: {macro_f1_val:.3f}")
                                else:
                                    st.error(f"Needs improvement: {macro_f1_val:.3f}")
                                
                                st.write("**PR-AUC Score:**")
                                pr_auc_val = metrics['pr_auc']
                                if pr_auc_val >= 0.7:
                                    st.success(f"Excellent precision-recall: {pr_auc_val:.3f}")
                                elif pr_auc_val >= 0.5:
                                    st.warning(f"Good precision-recall: {pr_auc_val:.3f}")
                                else:
                                    st.error(f"Needs improvement: {pr_auc_val:.3f}")
                            
                            with col2:
                                st.write("**Overall Assessment:**")
                                if metrics['accuracy'] >= 0.7 and macro_f1_val >= 0.6:
                                    st.success("🎯 Model performs well across all classes")
                                elif metrics['accuracy'] >= 0.6 and macro_f1_val >= 0.5:
                                    st.warning("⚠️ Model shows moderate performance")
                                else:
                                    st.error("❌ Model needs significant improvement")
                                
                                st.write("**Class Balance:**")
                                f1_scores = [metrics['f1_score'].get(cls, 0) for cls in ['HOLD', 'BUY', 'SELL']]
                                f1_std = np.std(f1_scores)
                                if f1_std <= 0.1:
                                    st.success("✅ Well-balanced across classes")
                                elif f1_std <= 0.2:
                                    st.warning("⚠️ Moderate class imbalance")
                                else:
                                    st.error("❌ Significant class imbalance")
                            
                            # Confusion matrix
                            st.subheader("📊 Confusion Matrix")
                            conf_matrix = np.array(evaluation_result['confusion_matrix'])
                            
                            fig = px.imshow(
                                conf_matrix,
                                text_auto=True,
                                aspect="auto",
                                labels=dict(x="Predicted", y="Actual"),
                                x=['HOLD', 'BUY', 'SELL'],
                                y=['HOLD', 'BUY', 'SELL'],
                                title="Confusion Matrix"
                            )
                            st.plotly_chart(fig)
                            
                            # Store evaluation result
                            st.session_state.last_evaluation = evaluation_result
                            
                        else:
                            st.error(f"❌ Evaluation failed: {evaluation_result['error']}")
                    else:
                        st.error("❌ Test data must include trading labels")
                else:
                    st.warning("⚠️ No test data available")
                    
            except Exception as e:
                st.error(f"❌ Error during evaluation: {e}")
    
    # Display last evaluation if available
    if 'last_evaluation' in st.session_state:
        st.subheader("📋 Last Evaluation Results")
        evaluation = st.session_state.last_evaluation
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", evaluation['model_name'])
        with col2:
            st.metric("Accuracy", f"{evaluation['metrics']['accuracy']:.3f}")
        with col3:
            st.metric("Macro-F1", f"{evaluation['metrics']['macro_f1']:.3f}")
        with col4:
            st.metric("PR-AUC", f"{evaluation['metrics']['pr_auc']:.3f}")
        
        # Evaluation timestamp with microsecond precision
        evaluation_time = evaluation.get('evaluation_timestamp', 'Unknown')
        if evaluation_time != 'Unknown':
            try:
                # Parse and format with microsecond precision
                dt = datetime.fromisoformat(evaluation_time.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds
                st.caption(f"Evaluation performed at: {formatted_time}")
            except:
                st.caption(f"Evaluation performed at: {evaluation_time}")
        else:
            st.caption(f"Evaluation performed at: {evaluation_time}")

def render_feature_analysis_tab(ml_pipeline):
    """Render the feature analysis tab"""
    st.header("🔍 Feature Engineering & Analysis")
    
    # Model features overview
    st.subheader("🤖 Model Features Overview")
    
    if ml_pipeline.active_model:
        active_model = ml_pipeline.active_model
        supported_features = active_model.get_supported_features()
        
        if supported_features:
            st.success(f"✅ Active model supports {len(supported_features)} features")
            
            # Display all features the model is using
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**All Model Features:**")
                features_df = pd.DataFrame({
                    'Feature Name': supported_features,
                    'Feature Index': range(len(supported_features))
                })
                st.dataframe(features_df)
            
            with col2:
                st.write("**Feature Categories:**")
                feature_categories = categorize_features(supported_features)
                for category, count in feature_categories.items():
                    st.metric(category, count)
            
            # Feature importance visualization (if available)
            if hasattr(active_model, 'get_feature_importance') and callable(getattr(active_model, 'get_feature_importance')):
                try:
                    feature_importance = active_model.get_feature_importance()
                    if feature_importance:
                        st.subheader("📊 Feature Importance (Model)")
                        
                        # Top features
                        top_features = dict(sorted(feature_importance.items(), 
                                                  key=lambda x: x[1], reverse=True)[:20])
                        
                        fig = px.bar(
                            x=list(top_features.values()),
                            y=list(top_features.keys()),
                            orientation='h',
                            title="Top 20 Most Important Features"
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig)
                except Exception as e:
                    st.info(f"ℹ️ Feature importance not available: {e}")
        else:
            st.warning("⚠️ Active model has no feature information")
    else:
        st.warning("⚠️ No active model selected")
    
    # Feature engineering parameters
    st.subheader("⚙️ Feature Engineering Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lookback_periods = st.multiselect(
            "Lookback Periods:",
            [5, 10, 20, 50, 100],
            default=[5, 10, 20, 50],
            key="feature_lookback"
        )
        
        horizon_ticks = st.slider(
            "Prediction Horizon (ticks):",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            key="feature_horizon"
        )
    
    with col2:
        threshold_ticks = st.slider(
            "Signal Threshold (ticks):",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            key="feature_threshold"
        )
        
        create_labels = st.checkbox(
            "Create Trading Labels",
            value=True,
            key="feature_labels"
        )
    
    # Sample data generation for feature analysis
    st.subheader("📊 Sample Data Generation for Feature Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_rows = st.number_input(
            "Number of Rows:",
            min_value=100,
            max_value=5000,
            value=500,
            step=100,
            key="analysis_rows_input"
        )
    
    with col2:
        analysis_price_range = st.slider(
            "Price Range ($):",
            min_value=10.0,
            max_value=1000.0,
            value=(50.0, 200.0),
            step=10.0,
            key="analysis_price_range_slider"
        )
    
    with col3:
        analysis_volatility = st.slider(
            "Volatility (%):",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            key="analysis_volatility_slider"
        )
    
    # Feature engineering demo
    if st.button("🔧 Generate Sample Features", key="generate_features_btn"):
        try:
            # Create sample data using the parameters
            sample_data = generate_realistic_sample_data(
                rows=analysis_rows,
                price_range=analysis_price_range,
                volatility=analysis_volatility
            )
            
            # Initialize feature engineer with selected parameters
            feature_engineer = TradingFeatureEngineer()
            
            with st.spinner("🔄 Generating features..."):
                # Process data
                if create_labels:
                    processed_data = feature_engineer.process_tick_data(
                        sample_data, create_labels=True
                    )
                else:
                    processed_data = feature_engineer.process_tick_data(
                        sample_data, create_labels=False
                    )
                
                if not processed_data.empty:
                    st.success(f"✅ Generated {len(processed_data)} feature records with {processed_data.shape[1]} columns")
                    
                    # Display feature categories
                    feature_categories = feature_engineer.get_feature_columns(processed_data)
                    
                    st.subheader("📊 Feature Categories")
                    for category, features in feature_categories.items():
                        if features:
                            with st.expander(f"{category} ({len(features)} features)"):
                                st.write(features)
                    
                    # Feature statistics
                    st.subheader("📈 Feature Statistics")
                    
                    # Select numeric features for visualization
                    numeric_features = processed_data.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_features) > 0:
                        selected_features = st.multiselect(
                            "Select features to visualize:",
                            numeric_features,
                            default=numeric_features[:5],
                            key="feature_viz_select"
                        )
                        
                        if selected_features:
                            # Feature distributions
                            fig = make_subplots(
                                rows=len(selected_features), cols=1,
                                subplot_titles=selected_features,
                                vertical_spacing=0.1
                            )
                            
                            for i, feature in enumerate(selected_features):
                                fig.add_trace(
                                    go.Histogram(x=processed_data[feature], name=feature),
                                    row=i+1, col=1
                                )
                            
                            fig.update_layout(height=200*len(selected_features), showlegend=False)
                            st.plotly_chart(fig)
                    
                    # Store processed data
                    st.session_state.sample_features = processed_data
                    
                else:
                    st.error("❌ Failed to generate features")
                    
        except Exception as e:
            st.error(f"❌ Error generating features: {e}")
    
    # Display sample features if available
    if 'sample_features' in st.session_state:
        st.subheader("📋 Sample Generated Features")
        sample_features = st.session_state.sample_features
        
        # Show first few rows
        st.write("**First 10 rows:**")
        st.dataframe(sample_features.head(10))
        
        # Feature correlation matrix
        st.subheader("🔗 Feature Correlations")
        
        # Select numeric features for correlation
        numeric_features = sample_features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) > 1:
            correlation_matrix = sample_features[numeric_features].corr()
            
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig)

def render_configuration_tab(ml_pipeline):
    """Render the configuration tab"""
    st.header("⚙️ Pipeline Configuration")
    
    # Model information
    st.subheader("🤖 Model Information")
    
    model_info = ml_pipeline.get_model_info()
    
    if 'error' not in model_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Models", model_info['total_models'])
            st.metric("Active Model", model_info['active_model_name'] or "None")
        
        with col2:
            if model_info['active_model']:
                active_info = model_info['active_model']
                st.metric("Model Type", active_info.get('model_type', 'Unknown'))
                st.metric("Features", active_info.get('supported_features', 0))
    
        # Available models
        st.write("**Available Models:**")
        for model_name in model_info['available_models']:
            if model_name == model_info['active_model_name']:
                st.success(f"✅ {model_name} (Active)")
            else:
                st.info(f"📁 {model_name}")
        
        # Active model details
        if model_info['active_model']:
            st.subheader("🔍 Active Model Details")
            active_info = model_info['active_model']
            
            # Model parameters
            if 'model_parameters' in active_info:
                st.write("**Model Parameters:**")
                params_df = pd.DataFrame(list(active_info['model_parameters'].items()), 
                                       columns=['Parameter', 'Value'])
                st.dataframe(params_df)
            
            # Supported features
            if 'supported_features' in active_info:
                st.write("**Supported Features:**")
                if isinstance(active_info['supported_features'], list):
                    st.write(f"Total: {len(active_info['supported_features'])} features")
                    st.write(", ".join(active_info['supported_features'][:10]))  # Show first 10
                    if len(active_info['supported_features']) > 10:
                        st.write(f"... and {len(active_info['supported_features']) - 10} more")
                else:
                    st.write(active_info['supported_features'])
    
    # Pipeline configuration
    st.subheader("🔧 Pipeline Settings")
    
    # Database settings
    st.write("**Database Configuration:**")
    st.code(f"Database File: {ml_pipeline.db_file}")
    st.code(f"Connected: {ml_pipeline.db_conn is not None}")
    
    # Kafka settings
    st.write("**Kafka Configuration:**")
    st.code(f"Bootstrap Servers: {ml_pipeline.kafka_bootstrap_servers}")
    st.code(f"Input Topic: {ml_pipeline.input_topic}")
    st.code(f"Output Topic: {ml_pipeline.output_topic}")
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    
    performance = ml_pipeline.get_pipeline_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Inference Count", performance['inference_count'])
        st.metric("Models Loaded", performance['models_loaded'])
    
    with col2:
        st.metric("Avg Inference Time", f"{performance['avg_inference_time']:.6f}s")
        st.metric("Database Connected", "✅" if performance['database_connected'] else "❌")
    
    with col3:
        st.metric("Feature Engineer", "✅" if performance['feature_engineer_ready'] else "❌")
        if performance['last_inference']:
            # Format with microsecond precision
            try:
                last_inference_time = performance['last_inference']
                if isinstance(last_inference_time, str):
                    # Try to parse and format with microsecond precision
                    dt = datetime.fromisoformat(last_inference_time.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%H:%M:%S.%f')[:-3]  # Include milliseconds
                    st.metric("Last Inference", formatted_time)
                else:
                    st.metric("Last Inference", str(last_inference_time)[:19])
            except:
                st.metric("Last Inference", str(performance['last_inference'])[:19])
    
    # Pipeline actions
    st.subheader("🔄 Pipeline Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reload Models", key="reload_models_btn"):
            with st.spinner("🔄 Reloading models..."):
                try:
                    ml_pipeline.load_models()
                    st.success("✅ Models reloaded successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Failed to reload models: {e}")
    
    with col2:
        if st.button("🧹 Clear Performance Metrics", key="clear_metrics_btn"):
            ml_pipeline.inference_count = 0
            ml_pipeline.avg_inference_time = 0.0
            ml_pipeline.last_inference_time = None
            st.success("✅ Performance metrics cleared!")
            st.experimental_rerun()
    
    # Export configuration
    st.subheader("📤 Export Configuration")
    
    if st.button("💾 Export Config", key="export_config_btn"):
        config = {
            'model_dir': ml_pipeline.model_dir,
            'db_file': ml_pipeline.db_file,
            'kafka_bootstrap_servers': ml_pipeline.kafka_bootstrap_servers,
            'input_topic': ml_pipeline.input_topic,
            'output_topic': ml_pipeline.output_topic,
            'active_model': ml_pipeline.active_model.model_name if ml_pipeline.active_model else None,
            'models_loaded': list(ml_pipeline.models.keys()),
            'export_timestamp': datetime.now().isoformat()
        }
        
        st.download_button(
            label="📥 Download Configuration JSON",
            data=json.dumps(config, indent=2),
            file_name=f"ml_pipeline_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        ) 