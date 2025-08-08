#!/usr/bin/env python3
"""
WebSocket Server for Tick Data
Receives tick data from clients and forwards to Kafka
"""

import asyncio
import websockets
import json
import logging
import os
from datetime import datetime
from kafka import KafkaProducer
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'tick-data')
WS_HOST = os.getenv('WS_HOST', '0.0.0.0')
WS_PORT = int(os.getenv('WS_PORT', 8080))

# Initialize Kafka producer
producer = None

def init_kafka_producer():
    """Initialize Kafka producer"""
    global producer
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        logger.info(f"✅ Kafka producer initialized with bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Kafka producer: {e}")
        raise

async def handle_tick_data(websocket):
    """Handle incoming tick data from WebSocket clients"""
    client_id = f"client_{id(websocket)}"
    logger.info(f"🔄 New WebSocket connection: {client_id}")
    
    try:
        async for message in websocket:
            try:
                # Parse the tick data
                tick_data = json.loads(message)
                
                # Add timestamps for latency tracking
                tick_data['ws_received_at'] = datetime.now().isoformat()
                tick_data['ws_processed_at'] = datetime.now().isoformat()
                
                # Forward to Kafka
                if producer:
                    future = producer.send(
                        KAFKA_TOPIC,
                        value=tick_data,
                        key=tick_data.get('symbol', 'unknown')
                    )
                    
                    # Wait for the send to complete
                    record_metadata = future.get(timeout=10)
                    tick_data['kafka_sent_at'] = datetime.now().isoformat()
                    
                    logger.info(f"📤 Sent tick to Kafka: {tick_data['symbol']} @ {tick_data['price']}")
                    
                    # Send acknowledgment back to client
                    ack = {
                        'status': 'success',
                        'message': 'Tick data sent to Kafka',
                        'kafka_topic': KAFKA_TOPIC,
                        'kafka_partition': record_metadata.partition,
                        'kafka_offset': record_metadata.offset,
                        'timestamp': datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(ack))
                    
                else:
                    logger.error("❌ Kafka producer not initialized")
                    error_ack = {
                        'status': 'error',
                        'message': 'Kafka producer not available',
                        'timestamp': datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(error_ack))
                    
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON received: {e}")
                error_msg = {
                    'status': 'error',
                    'message': 'Invalid JSON format',
                    'timestamp': datetime.now().isoformat()
                }
                await websocket.send(json.dumps(error_msg))
                
            except Exception as e:
                logger.error(f"❌ Error processing tick data: {e}")
                error_msg = {
                    'status': 'error',
                    'message': f'Processing error: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
                await websocket.send(json.dumps(error_msg))
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"🔌 WebSocket connection closed: {client_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")

async def main():
    """Main function to start the WebSocket server"""
    logger.info("🚀 Starting WebSocket Server...")
    
    # Initialize Kafka producer
    init_kafka_producer()
    
    # Start WebSocket server
    server = await websockets.serve(
        handle_tick_data,
        WS_HOST,
        WS_PORT
    )
    
    logger.info(f"✅ WebSocket server started on {WS_HOST}:{WS_PORT}")
    logger.info(f"📊 Forwarding to Kafka topic: {KAFKA_TOPIC}")
    
    # Keep the server running
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 WebSocket server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
    finally:
        if producer:
            producer.close()
            logger.info("🔌 Kafka producer closed") 