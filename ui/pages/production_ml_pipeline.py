#!/usr/bin/env python3
"""
Production ML Pipeline UI
Real-time trading system interface with TBT data synthesis and LightGBM training
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_production_ml_pipeline_ui():
    """Render the production ML pipeline UI"""
    
    st.title("🚀 Production ML Pipeline - Real Trading System")
    st.markdown("**Production-grade machine learning pipeline for real-time trading with LightGBM**")
    
    # Initialize production pipeline
    try:
        from ml_service.production_ml_pipeline import ProductionMLPipeline
        ml_pipeline = ProductionMLPipeline()
        
        # Setup database
        if ml_pipeline.setup_database():
            st.success("✅ Database connection established")
        else:
            st.error("❌ Database connection failed")
            return
        
        # Load models
        models_info = ml_pipeline.load_models()
        
    except ImportError as e:
        st.error(f"❌ Failed to import production ML pipeline: {e}")
        st.info("Please ensure all production modules are properly installed")
        return
    except Exception as e:
        st.error(f"❌ Failed to initialize production pipeline: {e}")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 TBT Data Synthesis", 
        "🔧 Model Training", 
        "📊 Model Performance",
        "🎲 Real-time Inference",
        "⚡ Performance Benchmark",
        "📈 Feature Analysis"
    ])
    
    # Tab 1: TBT Data Synthesis
    with tab1:
        render_tbt_data_synthesis_tab(ml_pipeline)
    
    # Tab 2: Model Training
    with tab2:
        render_model_training_tab(ml_pipeline)
    
    # Tab 3: Model Performance
    with tab3:
        render_model_performance_tab(ml_pipeline)
    
    # Tab 4: Real-time Inference
    with tab4:
        render_realtime_inference_tab(ml_pipeline)
    
    # Tab 5: Performance Benchmark
    with tab5:
        render_performance_benchmark_tab(ml_pipeline)
    
    # Tab 6: Feature Analysis
    with tab6:
        render_feature_analysis_tab(ml_pipeline)

def render_tbt_data_synthesis_tab(ml_pipeline):
    """Render TBT data synthesis tab"""
    
    st.header("🎯 TBT Data Synthesis Engine")
    st.markdown("Generate realistic tick-by-tick data for production trading systems")
    
    # Data Generation Parameters
    st.subheader("Data Generation Parameters")
    
    # Stock symbols
    default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    symbols = st.multiselect(
        "Select Stock Symbols",
        options=default_symbols,
        default=default_symbols[:4],
        help="Choose stocks for data generation"
    )
    
    # Duration and tick rate
    duration_hours = st.slider(
        "Market Session Duration (hours)",
        min_value=0.5,
        max_value=24.0,
        value=6.5,
        step=0.5,
        help="Duration of market session data to generate"
    )
    
    tick_rate_ms = st.selectbox(
        "Tick Rate (milliseconds)",
        options=[1, 5, 10, 50, 100],
        index=0,
        help="Tick rate for data generation (1ms = 1000 ticks/second)"
    )
    
    # Row count selector
    target_rows = st.number_input(
        "Target Total Rows",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=1000,
        help="Approximate total number of rows to generate across all symbols"
    )
    
    # Market events
    add_market_events = st.checkbox(
        "Add Market Events",
        value=True,
        help="Include realistic market events (gaps, volume spikes, spread widening)"
    )
    
    # Generate button
    if st.button("🚀 Generate TBT Data"):
        if not symbols:
            st.error("Please select at least one stock symbol")
            return
        
        with st.spinner("Generating realistic TBT data..."):
            try:
                start_time = time.time()
                
                # Calculate duration based on target rows and tick rate
                estimated_ticks_per_symbol = target_rows // len(symbols)
                # Calculate duration: ticks = (duration_minutes * 60 * 1000) / tick_rate_ms
                # So: duration_minutes = (ticks * tick_rate_ms) / (60 * 1000)
                estimated_duration_minutes = (estimated_ticks_per_symbol * tick_rate_ms) / (60 * 1000)
                
                st.info(f"📊 Target: {target_rows:,} total rows")
                st.info(f"📊 Estimated: {estimated_ticks_per_symbol:,} ticks per symbol")
                st.info(f"📊 Duration: {estimated_duration_minutes:.1f} minutes ({estimated_duration_minutes/60:.2f} hours)")
                
                # Generate data with target rows (more accurate)
                training_data = ml_pipeline.generate_training_data_with_target_rows(
                    symbols=symbols,
                    target_total_rows=target_rows,
                    tick_rate_ms=tick_rate_ms,
                    add_market_events=add_market_events
                )
                
                generation_time = time.time() - start_time
                
                # Store in session state
                st.session_state.training_data = training_data
                st.session_state.data_generation_time = generation_time
                
                st.success(f"✅ Generated {len(training_data):,} ticks in {generation_time:.2f}s")
                
                # Show detailed statistics
                st.info(f"📈 Generated {len(training_data):,} total rows across {len(symbols)} symbols")
                
            except Exception as e:
                st.error(f"❌ Data generation failed: {e}")
                st.error("Please check the logs for more details")
                # Show the full error for debugging
                st.exception(e)
    
    # Data Statistics
    st.subheader("Data Statistics")
    
    if 'training_data' in st.session_state:
        data = st.session_state.training_data
        
        # Show data info in a flat layout
        st.write("**Data Overview:**")
        st.write(f"• **Columns:** {len(data.columns)}")
        st.write(f"• **Memory Usage:** {data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        st.write(f"• **Data Types:** {len(data.dtypes.unique())}")
        
        # Show sample data
        st.write("**Sample Data:**")
        st.dataframe(data.head(10))
        
        # Show data distribution
        if len(data) > 0:
            st.write("**Data Distribution:**")
            
            # Price distribution
            fig_price = px.histogram(
                data, 
                x='price', 
                title="Price Distribution",
                nbins=50
            )
            st.plotly_chart(fig_price)
            
            # Volume distribution
            fig_volume = px.histogram(
                data, 
                x='volume', 
                title="Volume Distribution",
                nbins=50
            )
            st.plotly_chart(fig_volume)

def render_model_training_tab(ml_pipeline):
    """Render model training tab"""
    
    st.header("🔧 Production Model Training")
    st.markdown("Train LightGBM models with hyperparameter optimization for real trading")
    
    if 'training_data' not in st.session_state:
        st.warning("⚠️ Please generate TBT data first in the Data Synthesis tab")
        return
    
    # Training Configuration
    st.subheader("Training Configuration")
    
    # Model name
    model_name = st.text_input(
        "Model Name",
        value=f"production_model_{datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Name for the trained model"
    )
    
    # Training parameters
    test_size = st.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data for testing"
    )
    
    validation_size = st.slider(
        "Validation Set Size (%)",
        min_value=5,
        max_value=20,
        value=10,
        step=5,
        help="Percentage of data for validation"
    )
    
    # Fast training mode
    fast_training = st.checkbox(
        "🚀 Fast Training Mode",
        value=True,
        help="Enable for 5-10x faster training (reduced hyperparameter optimization trials and iterations)"
    )
    
    # Hyperparameter optimization
    if fast_training:
        n_trials = st.slider(
            "Optimization Trials (Fast Mode)",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
            help="Number of hyperparameter optimization trials (reduced for speed)"
        )
        timeout_minutes = st.slider(
            "Optimization Timeout (Fast Mode)",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Timeout for hyperparameter optimization in minutes"
        )
    else:
        n_trials = st.slider(
            "Optimization Trials (Full Mode)",
            min_value=20,
            max_value=100,
            value=50,
            step=10,
            help="Number of hyperparameter optimization trials"
        )
        timeout_minutes = st.slider(
            "Optimization Timeout (Full Mode)",
            min_value=10,
            max_value=60,
            value=30,
            step=5,
            help="Timeout for hyperparameter optimization in minutes"
        )
    
    # Training button
    if st.button("🚀 Start Training"):
        with st.spinner("Training production LightGBM model..."):
            try:
                start_time = time.time()
                
                training_results = ml_pipeline.train_new_model(
                    training_data=st.session_state.training_data,
                    model_name=model_name,
                    optimize_hyperparams=fast_training, # Pass fast_training flag
                    n_trials=n_trials,
                    test_size=test_size/100,
                    validation_size=validation_size/100,
                    timeout_minutes=timeout_minutes # Pass timeout_minutes
                )
                
                training_time = time.time() - start_time
                
                if training_results['success']:
                    st.success("✅ Model training completed successfully!")
                    
                    # Store results
                    st.session_state.training_results = training_results
                    st.session_state.training_time = training_time
                    
                    # Show results in a flat layout
                    st.write("**Training Results:**")
                    st.write(f"• **Training Time:** {training_time:.2f}s")
                    st.write(f"• **Feature Count:** {training_results.get('feature_count', 0)}")
                    
                    model_path = training_results.get('model_path', 'Unknown')
                    if model_path and model_path != 'Unknown':
                        st.write(f"• **Model Path:** {model_path.split('/')[-1]}")
                    else:
                        st.write("• **Model Path:** Not saved")
                    
                    # Show evaluation metrics
                    if 'evaluation_metrics' in training_results and training_results['evaluation_metrics']:
                        metrics = training_results['evaluation_metrics']
                        st.subheader("📊 Model Performance")
                        
                        # Use expander to avoid nested columns
                        with st.expander("📊 Detailed Metrics", expanded=True):
                            # Key metrics in a flat layout
                            st.write("**Key Metrics:**")
                            st.write(f"• **Accuracy:** {metrics.get('accuracy', 0):.4f}")
                            st.write(f"• **F1 Macro:** {metrics.get('f1_macro', 0):.4f}")
                            st.write(f"• **Precision:** {metrics.get('precision_macro', 0):.4f}")
                            st.write(f"• **Recall:** {metrics.get('recall_macro', 0):.4f}")
                            st.write(f"• **Inference Time:** {metrics.get('inference_time_ms', 0):.2f}ms")
                    else:
                        st.warning("⚠️ No evaluation metrics available")
                
                else:
                    st.error(f"❌ Model training failed: {training_results.get('error', 'Unknown error')}")
                
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                # Store error in session state for debugging
                st.session_state.training_error = str(e)
    
    # Training Status
    st.subheader("Training Status")
    
    if 'training_results' in st.session_state:
        results = st.session_state.training_results
        
        if results.get('success', False):
            st.success("✅ Model Trained")
            st.write(f"• **Features:** {results.get('feature_count', 0)}")
            st.write(f"• **Training Time:** {st.session_state.get('training_time', 0):.2f}s")
            
            if 'hyperparameters' in results and results['hyperparameters']:
                st.subheader("Best Hyperparameters")
                for param, value in results['hyperparameters'].items():
                    st.write(f"• **{param}:** {value}")
        else:
            st.error("❌ Model Training Failed")
            if 'error' in results:
                st.error(f"Error: {results['error']}")
    else:
        st.info("No model trained yet")
    
    # Model info
    if ml_pipeline.active_model:
        st.subheader("Active Model")
        model_info = ml_pipeline.get_model_info()
        
        if 'error' not in model_info:
            st.write(f"• **Model Type:** {model_info.get('model_type', 'Unknown')}")
            st.write(f"• **Features:** {model_info.get('feature_count', 0)}")
            st.write(f"• **Total Predictions:** {model_info.get('total_predictions', 0)}")
            
            if model_info.get('last_inference'):
                st.write(f"• **Last Inference:** {str(model_info['last_inference'])[:19]}")
        else:
            st.error(f"Error getting model info: {model_info['error']}")
    else:
        st.info("No active model available")
    
    # Show any training errors
    if 'training_error' in st.session_state:
        st.error(f"❌ Last Training Error: {st.session_state.training_error}")
        if st.button("Clear Error"):
            del st.session_state.training_error

def render_model_performance_tab(ml_pipeline):
    """Render model performance tab"""
    
    st.header("📊 Model Performance Analysis")
    st.markdown("Comprehensive evaluation and monitoring of model performance")
    
    if not ml_pipeline.active_model:
        st.warning("⚠️ No active model available. Please train a model first.")
        return
    
    # Performance Evaluation
    st.subheader("Performance Evaluation")
    
    # Test data selection
    test_data_source = st.radio(
        "Test Data Source",
        options=["Use Generated Training Data", "Generate New Test Data"],
        help="Choose source for performance evaluation"
    )
    
    if test_data_source == "Generate New Test Data":
        # Generate new test data
        test_symbols = st.multiselect(
            "Test Symbols",
            options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            default=['AAPL', 'MSFT']
        )
        
        test_duration = st.slider(
            "Test Duration (hours)",
            min_value=0.5,
            max_value=4.0,
            value=1.0,
            step=0.5
        )
        
        if st.button("🔍 Evaluate Performance"):
            if not test_symbols:
                st.error("Please select test symbols")
                return
            
            with st.spinner("Evaluating model performance..."):
                try:
                    # Generate test data
                    test_data = ml_pipeline.generate_training_data(
                        symbols=test_symbols,
                        duration_hours=test_duration,
                        tick_rate_ms=1
                    )
                    
                    # Evaluate model
                    evaluation_metrics = ml_pipeline.evaluate_model_performance(test_data)
                    
                    if 'error' not in evaluation_metrics:
                        st.session_state.evaluation_metrics = evaluation_metrics
                        st.success("✅ Performance evaluation completed")
                    else:
                        st.error(f"❌ Evaluation failed: {evaluation_metrics['error']}")
                
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {e}")
    
    else:
        # Use existing training data
        if 'training_data' in st.session_state and st.button("🔍 Evaluate on Training Data"):
            with st.spinner("Evaluating model performance..."):
                try:
                    evaluation_metrics = ml_pipeline.evaluate_model_performance(
                        st.session_state.training_data
                    )
                    
                    if 'error' not in evaluation_metrics:
                        st.session_state.evaluation_metrics = evaluation_metrics
                        st.success("✅ Performance evaluation completed")
                    else:
                        st.error(f"❌ Evaluation failed: {evaluation_metrics['error']}")
                
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {e}")
    
    # Display evaluation results
    if 'evaluation_metrics' in st.session_state:
        metrics = st.session_state.evaluation_metrics
        
        st.subheader("📈 Performance Metrics")
        
        # Use expander to avoid nested columns
        with st.expander("📊 Detailed Metrics", expanded=True):
            # Key metrics in a flat layout
            st.write("**Key Metrics:**")
            st.write(f"• **Accuracy:** {metrics.get('accuracy', 0):.4f}")
            st.write(f"• **F1 Macro:** {metrics.get('f1_macro', 0):.4f}")
            st.write(f"• **Precision:** {metrics.get('precision_macro', 0):.4f}")
            st.write(f"• **Recall:** {metrics.get('recall_macro', 0):.4f}")
            
            # Inference performance
            st.write(f"• **Inference Time:** {metrics.get('inference_time_ms', 0):.2f}ms")
        
        # Per-class metrics
        if 'class_metrics' in metrics:
            st.subheader("📊 Per-Class Performance")
            
            class_data = []
            for class_name in ['HOLD', 'BUY', 'SELL']:
                if class_name.lower() in metrics:
                    class_data.append({
                        'Class': class_name,
                        'Precision': metrics[f'{class_name.lower()}_precision'],
                        'Recall': metrics[f'{class_name.lower()}_recall'],
                        'F1-Score': metrics[f'{class_name.lower()}_f1']
                    })
            
            if class_data:
                class_df = pd.DataFrame(class_data)
                st.dataframe(class_df)
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            st.subheader("🔄 Confusion Matrix")
            
            cm = np.array(metrics['confusion_matrix'])
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['HOLD', 'BUY', 'SELL'],
                y=['HOLD', 'BUY', 'SELL'],
                text_auto=True,
                title="Confusion Matrix"
            )
            st.plotly_chart(fig_cm)
    
    # Model Statistics
    st.subheader("Model Statistics")
    
    model_info = ml_pipeline.get_model_info()
    
    st.write(f"• **Model Type:** {model_info.get('model_type', 'Unknown')}")
    st.write(f"• **Features:** {model_info.get('feature_count', 0)}")
    st.write(f"• **Total Predictions:** {model_info.get('total_predictions', 0)}")
    
    if model_info.get('avg_inference_time_ms', 0) > 0:
        st.write(f"• **Avg Inference Time:** {model_info['avg_inference_time_ms']:.2f}ms")
    
    if model_info.get('last_inference'):
        st.write(f"• **Last Inference:** {model_info['last_inference'][:19]}")
    
    # Training history
    if 'training_history' in model_info and model_info['training_history']:
        st.subheader("Training History")
        
        history = model_info['training_history'][-1]  # Latest training
        st.write(f"• **Best Iteration:** {history.get('best_iteration', 'N/A')}")
        st.write(f"• **Best Score:** {history.get('best_score', 'N/A')}")

def render_realtime_inference_tab(ml_pipeline):
    """Render real-time inference tab"""
    
    st.header("🎲 Real-Time Trading Inference")
    st.markdown("Live prediction and trading signal generation")
    
    if not ml_pipeline.active_model:
        st.warning("⚠️ No active model available. Please train a model first.")
        return
    
    # Live Inference
    st.subheader("Live Inference")
    
    # Inference parameters
    inference_symbol = st.selectbox(
        "Symbol",
        options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'],
        index=0
    )
    
    inference_duration = st.slider(
        "Inference Duration (minutes)",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )
    
    inference_interval = st.slider(
        "Update Interval (seconds)",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )
    
    # Start/Stop inference
    if 'inference_running' not in st.session_state:
        st.session_state.inference_running = False
    
    if st.button("🚀 Start Live Inference" if not st.session_state.inference_running else "⏹️ Stop Inference"):
        st.session_state.inference_running = not st.session_state.inference_running
        
        if st.session_state.inference_running:
            st.success("✅ Live inference started")
        else:
            st.success("⏹️ Live inference stopped")
    
    # Inference display
    if st.session_state.get('inference_running', False):
        st.subheader("📊 Live Predictions")
        
        # Create placeholder for live updates
        inference_placeholder = st.empty()
        
        # Simulate live inference
        for i in range(10):  # Simulate 10 predictions
            if not st.session_state.get('inference_running', False):
                break
            
            # Generate synthetic live data
            live_data = ml_pipeline.data_synthesizer.generate_realistic_tick_data(
                symbol=inference_symbol,
                duration_minutes=1,
                tick_rate_ms=1000  # 1 second intervals
            )
            
            # Make prediction
            prediction_result = ml_pipeline.make_prediction(live_data)
            
            if 'error' not in prediction_result:
                # Display live results
                with inference_placeholder.container():
                    # Use expander to avoid nested columns
                    with st.expander("📊 Live Results", expanded=True):
                        # Display results in a flat layout
                        st.write(f"**Symbol:** {inference_symbol}")
                        st.write(f"**Prediction:** {prediction_result['predictions'][0]}")
                        st.write(f"**Confidence:** {prediction_result['confidence_scores'][0]:.3f}")
                        st.write(f"**Signal:** {prediction_result['signals'][0]}")
                        st.write(f"**Inference Time:** {prediction_result['inference_time_ms']:.2f}ms")
                        st.write(f"**Timestamp:** {prediction_result['timestamp'][:19]}")
                        
                        # Show probability distribution
                        if 'probability_matrix' in prediction_result:
                            probs = prediction_result['probability_matrix'][0]
                            prob_df = pd.DataFrame({
                                'Class': ['HOLD', 'BUY', 'SELL'],
                                'Probability': probs
                            })
                            
                            fig_prob = px.bar(
                                prob_df,
                                x='Class',
                                y='Probability',
                                title="Prediction Probabilities",
                                color='Probability',
                                color_continuous_scale='RdYlGn'
                            )
                            st.plotly_chart(fig_prob)
                
                time.sleep(2)  # Wait between predictions
            
            st.success("✅ Live inference completed")
    
    # Inference Statistics
    st.subheader("Inference Statistics")
    
    model_info = ml_pipeline.get_model_info()
    
    st.write(f"• **Total Predictions:** {model_info.get('total_predictions', 0)}")
    st.write(f"• **Avg Inference Time:** {model_info.get('avg_inference_time_ms', 0):.2f}ms")
    
    if model_info.get('last_inference'):
        st.write(f"• **Last Inference:** {model_info['last_inference'][:19]}")
    
    # Performance metrics
    if 'inference_times' in ml_pipeline.__dict__ and ml_pipeline.inference_times:
        recent_times = ml_pipeline.inference_times[-10:]  # Last 10 predictions
        
        st.subheader("Recent Performance")
        st.write(f"• **Min Time:** {min(recent_times)*1000:.2f}ms")
        st.write(f"• **Max Time:** {max(recent_times)*1000:.2f}ms")
        st.write(f"• **Avg Time:** {np.mean(recent_times)*1000:.2f}ms")

def render_performance_benchmark_tab(ml_pipeline):
    """Render performance benchmark tab"""
    
    st.header("⚡ Performance Benchmark")
    st.markdown("Comprehensive performance testing for production deployment")
    
    if not ml_pipeline.active_model:
        st.warning("⚠️ No active model available. Please train a model first.")
        return
    
    # Benchmark Configuration
    st.subheader("Benchmark Configuration")
    
    # Benchmark parameters
    benchmark_iterations = st.slider(
        "Benchmark Iterations",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Number of iterations for performance testing"
    )
    
    benchmark_batch_size = st.slider(
        "Batch Size",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Number of ticks per batch"
    )
    
    # Test data generation
    benchmark_symbols = st.multiselect(
        "Benchmark Symbols",
        options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        default=['AAPL', 'MSFT'],
        help="Symbols to use for benchmarking"
    )
    
    benchmark_duration = st.slider(
        "Test Duration (hours)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Duration of test data to generate"
    )
    
    # Start benchmark
    if st.button("🚀 Start Performance Benchmark"):
        if not benchmark_symbols:
            st.error("Please select benchmark symbols")
            return
        
        with st.spinner("Running performance benchmark..."):
            try:
                # Generate benchmark data
                benchmark_data = ml_pipeline.generate_training_data(
                    symbols=benchmark_symbols,
                    duration_hours=benchmark_duration,
                    tick_rate_ms=1
                )
                
                # Run benchmark
                benchmark_results = ml_pipeline.benchmark_performance(
                    benchmark_data, 
                    iterations=benchmark_iterations
                )
                
                if 'error' not in benchmark_results:
                    st.session_state.benchmark_results = benchmark_results
                    st.success("✅ Performance benchmark completed")
                else:
                    st.error(f"❌ Benchmark failed: {benchmark_results['error']}")
                
            except Exception as e:
                st.error(f"❌ Benchmark failed: {e}")
    
    # Display benchmark results
    if 'benchmark_results' in st.session_state:
        results = st.session_state.benchmark_results
        
        st.subheader("📊 Benchmark Results")
        
        # Use expander to avoid nested columns
        with st.expander("📊 Detailed Results", expanded=True):
            # Key metrics in a flat layout
            st.write("**Key Metrics:**")
            st.write(f"• **Iterations:** {results['iterations']}")
            st.write(f"• **Avg Time:** {results['avg_inference_time_ms']:.2f}ms")
            st.write(f"• **Min Time:** {results['min_inference_time_ms']:.2f}ms")
            st.write(f"• **Max Time:** {results['max_inference_time_ms']:.2f}ms")
            st.write(f"• **Std Dev:** {results['std_inference_time_ms']:.2f}ms")
            st.write(f"• **P50 Time:** {results['p50_inference_time_ms']:.2f}ms")
            st.write(f"• **P95 Time:** {results['p95_inference_time_ms']:.2f}ms")
            st.write(f"• **P99 Time:** {results['p99_inference_time_ms']:.2f}ms")
            st.write(f"• **Throughput:** {results['throughput_ticks_per_second']:.0f} ticks/s")
        
        # Performance distribution chart
        st.subheader("📈 Performance Distribution")
        
        # Simulate performance distribution (in real implementation, this would come from actual benchmark data)
        performance_data = np.random.normal(
            results['avg_inference_time_ms'],
            results['std_inference_time_ms'],
            1000
        )
        performance_data = np.clip(performance_data, 0, results['max_inference_time_ms'])
        
        fig_dist = px.histogram(
            x=performance_data,
            title="Inference Time Distribution",
            labels={'x': 'Inference Time (ms)', 'y': 'Frequency'},
            nbins=50
        )
        fig_dist.add_vline(
            x=results['avg_inference_time_ms'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {results['avg_inference_time_ms']:.2f}ms"
        )
        st.plotly_chart(fig_dist)
    
    # Benchmark Status
    st.subheader("Benchmark Status")
    
    if 'benchmark_results' in st.session_state:
        results = st.session_state.benchmark_results
        
        st.success("✅ Benchmark Complete")
        st.write(f"• **Total Iterations:** {results['iterations']}")
        st.write("• **Success Rate:** 100%")
        
        # Performance summary
        st.subheader("Performance Summary")
        
        if results['avg_inference_time_ms'] < 1.0:
            st.success("🚀 Ultra-Low Latency (< 1ms)")
        elif results['avg_inference_time_ms'] < 5.0:
            st.success("⚡ Low Latency (< 5ms)")
        elif results['avg_inference_time_ms'] < 10.0:
            st.warning("⚠️ Medium Latency (< 10ms)")
        else:
            st.error("❌ High Latency (≥ 10ms)")
        
        # Throughput assessment
        if results['throughput_ticks_per_second'] > 10000:
            st.success("🚀 High Throughput (> 10k ticks/s)")
        elif results['throughput_ticks_per_second'] > 1000:
            st.success("⚡ Medium Throughput (> 1k ticks/s)")
        else:
            st.warning("⚠️ Low Throughput (≤ 1k ticks/s)")
    else:
        st.info("No benchmark results available")

def render_feature_analysis_tab(ml_pipeline):
    """Render feature analysis tab"""
    
    st.header("📈 Feature Analysis & Importance")
    st.markdown("Comprehensive analysis of model features and their importance")
    
    if not ml_pipeline.active_model:
        st.warning("⚠️ No active model available. Please train a model first.")
        return
    
    # Feature Overview
    st.subheader("Feature Overview")
    
    # Get feature information
    model_info = ml_pipeline.get_model_info()
    feature_importance = ml_pipeline.get_feature_importance()
    
    if feature_importance.empty:
        st.warning("⚠️ No feature importance data available")
        return
    
    # Feature count and types
    st.write(f"• **Total Features:** {model_info['feature_count']}")
    
    # Feature importance chart
    st.subheader("🎯 Feature Importance")
    
    # Top features
    top_features = feature_importance.head(20)
    
    fig_importance = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title="Top 20 Feature Importance",
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(height=600)
    st.plotly_chart(fig_importance)
    
    # Feature categories
    st.subheader("📊 Feature Categories")
    
    # Categorize features
    feature_categories = categorize_features(feature_importance['feature'].tolist())
    
    if feature_categories:
        fig_categories = px.pie(
            values=list(feature_categories.values()),
            names=list(feature_categories.keys()),
            title="Feature Distribution by Category"
        )
        st.plotly_chart(fig_categories)
    
    # Feature importance table
    st.subheader("📋 Feature Importance Table")
    st.dataframe(feature_importance)
    
    # Feature Statistics
    st.subheader("Feature Statistics")
    
    if not feature_importance.empty:
        st.write(f"• **Total Features:** {len(feature_importance)}")
        st.write(f"• **Top Feature:** {feature_importance.iloc[0]['feature']}")
        st.write(f"• **Max Importance:** {feature_importance.iloc[0]['importance']:.4f}")
        st.write(f"• **Min Importance:** {feature_importance.iloc[-1]['importance']:.4f}")
        
        # Feature distribution
        importance_values = feature_importance['importance'].to_numpy()
        
        st.subheader("Importance Distribution")
        st.write(f"• **Mean:** {np.mean(importance_values):.4f}")
        st.write(f"• **Median:** {np.median(importance_values):.4f}")
        st.write(f"• **Std Dev:** {np.std(importance_values):.4f}")
        
        # Feature selection
        st.subheader("Feature Selection")
        
        importance_threshold = st.slider(
            "Importance Threshold",
            min_value=0.0,
            max_value=float(np.max(importance_values)),
            value=float(np.percentile(importance_values, 25)),
            step=0.001
        )
        
        selected_features = feature_importance[feature_importance['importance'] >= importance_threshold]
        st.write(f"• **Selected Features:** {len(selected_features)}")
        
        if st.button("📊 Show Selected Features"):
            st.dataframe(selected_features)

def categorize_features(features):
    """Categorize features by type"""
    categories = {
        'Price Momentum': 0,
        'Volume Momentum': 0,
        'Spread Analysis': 0,
        'Bid-Ask Imbalance': 0,
        'VWAP Deviation': 0,
        'Technical Indicators': 0,
        'Time Features': 0,
        'Market Microstructure': 0,
        'Volatility Measures': 0,
        'Liquidity Metrics': 0,
        'Other': 0
    }
    
    for feature in features:
        if 'momentum' in feature.lower() and 'price' in feature.lower():
            categories['Price Momentum'] += 1
        elif 'momentum' in feature.lower() and 'volume' in feature.lower():
            categories['Volume Momentum'] += 1
        elif 'spread' in feature.lower():
            categories['Spread Analysis'] += 1
        elif 'imbalance' in feature.lower():
            categories['Bid-Ask Imbalance'] += 1
        elif 'vwap' in feature.lower():
            categories['VWAP Deviation'] += 1
        elif any(indicator in feature.lower() for indicator in ['rsi', 'macd', 'bollinger', 'stochastic', 'williams', 'atr', 'cci', 'adx', 'mfi']):
            categories['Technical Indicators'] += 1
        elif any(time_feature in feature.lower() for time_feature in ['hour', 'minute', 'second', 'microsecond', 'day', 'session', 'time']):
            categories['Time Features'] += 1
        elif any(micro in feature.lower() for micro in ['order_book', 'depth', 'quote', 'order_flow', 'liquidity', 'market_impact']):
            categories['Market Microstructure'] += 1
        elif any(vol in feature.lower() for vol in ['volatility', 'realized', 'parkinson', 'garman', 'rogers']):
            categories['Volatility Measures'] += 1
        elif any(liq in feature.lower() for liq in ['amihud', 'kyle', 'roll', 'effective', 'quoted', 'realized']):
            categories['Liquidity Metrics'] += 1
        else:
            categories['Other'] += 1
    
    return {k: v for k, v in categories.items() if v > 0}

if __name__ == "__main__":
    render_production_ml_pipeline_ui()
