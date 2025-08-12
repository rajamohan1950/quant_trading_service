#!/usr/bin/env python3
"""
Comprehensive ML Training Script for Stock Trading Patterns
Trains LightGBM and XGBoost models from tick data
"""

import logging
import sys
import os
import glob
from datetime import datetime
import argparse
import pandas as pd

# Add ml_service to path
sys.path.append('ml_service')

from feature_engineering import TickFeatureEngineer
from model_trainer import StockMLTrainer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'ml_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train ML models for stock prediction')
    parser.add_argument('--data-dir', default='data/parquet/', help='Directory containing parquet files')
    parser.add_argument('--output-dir', default='data/processed/', help='Directory for processed features')
    parser.add_argument('--model-dir', default='ml_models/', help='Directory to save trained models')
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter optimization')
    parser.add_argument('--models', nargs='+', default=['lightgbm', 'xgboost'], 
                       choices=['lightgbm', 'xgboost'], help='Models to train')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting ML Training Pipeline for Stock Trading Patterns")
    logger.info(f"📁 Data Directory: {args.data_dir}")
    logger.info(f"💾 Output Directory: {args.output_dir}")
    logger.info(f"🤖 Model Directory: {args.model_dir}")
    logger.info(f"⚙️ Hyperparameter Optimization: {args.optimize}")
    logger.info(f"🔧 Models to Train: {args.models}")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    try:
        # 1. Feature Engineering
        logger.info("📊 Step 1: Feature Engineering")
        feature_engineer = TickFeatureEngineer(lookback_periods=[5, 10, 20, 50])
        
        # Find all parquet files
        parquet_files = glob.glob(os.path.join(args.data_dir, "*.parquet"))
        
        if not parquet_files:
            logger.error(f"❌ No parquet files found in {args.data_dir}")
            return
        
        logger.info(f"📋 Found {len(parquet_files)} parquet files")
        
        # Process each file and combine
        all_features = []
        
        for parquet_file in parquet_files:
            logger.info(f"🔄 Processing {parquet_file}")
            
            # Process features
            processed_file = os.path.join(
                args.output_dir, 
                f"features_{os.path.basename(parquet_file)}"
            )
            
            features_df = feature_engineer.process_tick_data(parquet_file, processed_file)
            
            if not features_df.empty:
                all_features.append(features_df)
                logger.info(f"✅ Processed {len(features_df)} records from {parquet_file}")
            else:
                logger.warning(f"⚠️ No features generated from {parquet_file}")
        
        if not all_features:
            logger.error("❌ No features were generated from any files")
            return
        
        # Combine all features
        logger.info("🔗 Combining all feature datasets")
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Save combined features
        combined_file = os.path.join(args.output_dir, "combined_features.parquet")
        combined_df.to_parquet(combined_file, index=False)
        
        logger.info(f"💾 Combined features saved: {len(combined_df)} records, {combined_df.shape[1]} columns")
        logger.info(f"📁 File: {combined_file}")
        
        # 2. Model Training
        logger.info("🤖 Step 2: Model Training")
        trainer = StockMLTrainer(model_dir=args.model_dir)
        
        # Define target configurations
        target_configs = [
            # Direction prediction (classification)
            {'target': 'direction_1', 'type': 'classification'},
            {'target': 'direction_5', 'type': 'classification'},
            {'target': 'direction_10', 'type': 'classification'},
            
            # Return prediction (regression)
            {'target': 'future_return_1', 'type': 'regression'},
            {'target': 'future_return_5', 'type': 'regression'},
            {'target': 'future_return_10', 'type': 'regression'},
            
            # Return bucket prediction (classification)
            {'target': 'return_bucket_1', 'type': 'classification'},
            {'target': 'return_bucket_5', 'type': 'classification'},
            {'target': 'return_bucket_10', 'type': 'classification'},
        ]
        
        # Filter targets that exist in the data
        available_targets = []
        for config in target_configs:
            if config['target'] in combined_df.columns:
                available_targets.append(config)
            else:
                logger.warning(f"⚠️ Target {config['target']} not found in data")
        
        logger.info(f"🎯 Training models for {len(available_targets)} targets")
        
        # Train models
        results = trainer.train_multiple_models(
            combined_df, 
            available_targets, 
            model_types=args.models
        )
        
        # 3. Results Summary
        logger.info("📈 Step 3: Training Results Summary")
        
        for target, target_results in results.items():
            logger.info(f"\n🎯 Target: {target}")
            
            for model_type, model_result in target_results.items():
                metrics = model_result['metrics']
                logger.info(f"  🔧 {model_type}:")
                
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        logger.info(f"    📊 {metric}: {value:.4f}")
                    else:
                        logger.info(f"    📊 {metric}: {value}")
        
        # Get model summary
        summary_df = trainer.get_model_summary()
        if not summary_df.empty:
            summary_file = os.path.join(args.model_dir, "model_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"📋 Model summary saved: {summary_file}")
        
        # Feature importance analysis
        logger.info("🔍 Feature Importance Analysis")
        for target, target_results in results.items():
            for model_type, model_result in target_results.items():
                feature_importance = model_result.get('feature_importance', {})
                if feature_importance:
                    logger.info(f"\n🔍 Top 10 features for {model_type}_{target}:")
                    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                        logger.info(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
        logger.info("🎉 ML Training Pipeline completed successfully!")
        logger.info(f"📊 Total models trained: {sum(len(tr) for tr in results.values())}")
        logger.info(f"🎯 Targets covered: {len(results)}")
        logger.info(f"🤖 Model types: {args.models}")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 