#!/usr/bin/env python3
"""
Start ML Deployment Service
Launches the FastAPI server for real-time stock predictions
"""

import logging
import sys
import os
import asyncio
import argparse
from datetime import datetime

# Add ml_service to path
sys.path.append('ml_service')

from model_deployment import MLDeploymentService, MLPredictionStream
import uvicorn

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'ml_service_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

async def main():
    """Main function to start ML deployment service"""
    parser = argparse.ArgumentParser(description='Start ML Deployment Service')
    parser.add_argument('--model-dir', default='ml_models/', help='Directory containing trained models')
    parser.add_argument('--db-file', default='tick_data.db', help='DuckDB database file')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the service on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the service on')
    parser.add_argument('--kafka-servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--input-topic', default='tick-data', help='Kafka input topic')
    parser.add_argument('--output-topic', default='predictions', help='Kafka output topic')
    parser.add_argument('--stream-only', action='store_true', help='Run only prediction stream (no API)')
    parser.add_argument('--api-only', action='store_true', help='Run only API server (no stream)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting ML Deployment Service")
    logger.info(f"🤖 Model Directory: {args.model_dir}")
    logger.info(f"🗄️ Database File: {args.db_file}")
    logger.info(f"🌐 API Server: {args.host}:{args.port}")
    logger.info(f"📡 Kafka Servers: {args.kafka_servers}")
    logger.info(f"📥 Input Topic: {args.input_topic}")
    logger.info(f"📤 Output Topic: {args.output_topic}")
    
    try:
        # Create deployment service
        deployment_service = MLDeploymentService(
            model_dir=args.model_dir,
            db_file=args.db_file
        )
        
        # Load models
        models = deployment_service.load_models()
        
        if not models:
            logger.error("❌ No models loaded! Please train models first.")
            sys.exit(1)
        
        logger.info(f"✅ Loaded {len(models)} models")
        
        # Prepare services based on arguments
        services = []
        
        if not args.stream_only:
            # FastAPI server
            config = uvicorn.Config(
                app=deployment_service.app,
                host=args.host,
                port=args.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            services.append(server.serve())
            logger.info(f"🌐 API Server will start on http://{args.host}:{args.port}")
        
        if not args.api_only:
            # Prediction stream
            prediction_stream = MLPredictionStream(
                deployment_service=deployment_service,
                input_topic=args.input_topic,
                output_topic=args.output_topic,
                kafka_bootstrap_servers=args.kafka_servers
            )
            services.append(prediction_stream.start_prediction_stream())
            logger.info(f"📡 Prediction stream will consume from {args.input_topic}")
        
        if not services:
            logger.error("❌ No services configured to run!")
            sys.exit(1)
        
        # Start services
        logger.info("🎯 Starting all services...")
        await asyncio.gather(*services)
        
    except KeyboardInterrupt:
        logger.info("🛑 ML Deployment Service stopped by user")
    except Exception as e:
        logger.error(f"❌ Service startup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 