#!/usr/bin/env python3
"""
Kafka Consumer for Tick Data
Consumes tick data from Kafka and calculates latency metrics
"""

import json
import logging
import os
from datetime import datetime
from kafka import KafkaConsumer
import time
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'tick-data')
CONSUMER_GROUP_ID = os.getenv('CONSUMER_GROUP_ID', 'latency-monitor-group')

# Latency tracking
latency_data = []
total_messages = 0

def calculate_latency(tick_data):
    """Calculate latency metrics from tick data"""
    try:
        # Extract timestamps
        tick_generated_at = datetime.fromisoformat(tick_data.get('timestamp', ''))
        ws_received_at = datetime.fromisoformat(tick_data.get('ws_received_at', ''))
        ws_processed_at = datetime.fromisoformat(tick_data.get('ws_processed_at', ''))
        kafka_sent_at = datetime.fromisoformat(tick_data.get('kafka_sent_at', ''))
        kafka_consumed_at = datetime.now()
        
        # Calculate latencies
        t1 = (ws_received_at - tick_generated_at).total_seconds() * 1000  # Tick → WS
        t2 = (kafka_sent_at - ws_processed_at).total_seconds() * 1000    # WS → Kafka
        t3 = (kafka_consumed_at - kafka_sent_at).total_seconds() * 1000  # Kafka → Consumer
        total = t1 + t2 + t3
        
        latency_info = {
            'symbol': tick_data.get('symbol', 'unknown'),
            'price': tick_data.get('price', 0),
            't1': round(t1, 2),
            't2': round(t2, 2),
            't3': round(t3, 2),
            'total': round(total, 2),
            'timestamp': kafka_consumed_at.isoformat(),
            'message_id': tick_data.get('id', 'unknown')
        }
        
        return latency_info
        
    except Exception as e:
        logger.error(f"❌ Error calculating latency: {e}")
        return None

def print_statistics():
    """Print current latency statistics"""
    if not latency_data:
        return
    
    t1_values = [d['t1'] for d in latency_data]
    t2_values = [d['t2'] for d in latency_data]
    t3_values = [d['t3'] for d in latency_data]
    total_values = [d['total'] for d in latency_data]
    
    stats = {
        'T1 (Tick→WS)': {
            'mean': statistics.mean(t1_values),
            'std': statistics.stdev(t1_values) if len(t1_values) > 1 else 0,
            'min': min(t1_values),
            'max': max(t1_values)
        },
        'T2 (WS→Kafka)': {
            'mean': statistics.mean(t2_values),
            'std': statistics.stdev(t2_values) if len(t2_values) > 1 else 0,
            'min': min(t2_values),
            'max': max(t2_values)
        },
        'T3 (Kafka→Consumer)': {
            'mean': statistics.mean(t3_values),
            'std': statistics.stdev(t3_values) if len(t3_values) > 1 else 0,
            'min': min(t3_values),
            'max': max(t3_values)
        },
        'Total (End-to-End)': {
            'mean': statistics.mean(total_values),
            'std': statistics.stdev(total_values) if len(total_values) > 1 else 0,
            'min': min(total_values),
            'max': max(total_values)
        }
    }
    
    logger.info("📊 Latency Statistics:")
    for metric, values in stats.items():
        logger.info(f"  {metric}:")
        logger.info(f"    Mean: {values['mean']:.2f}ms")
        logger.info(f"    Std:  {values['std']:.2f}ms")
        logger.info(f"    Min:  {values['min']:.2f}ms")
        logger.info(f"    Max:  {values['max']:.2f}ms")

def main():
    """Main function to start the Kafka consumer"""
    global total_messages, latency_data
    
    logger.info("🚀 Starting Kafka Consumer...")
    
    try:
        # Initialize Kafka consumer
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=CONSUMER_GROUP_ID,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        logger.info(f"✅ Kafka consumer initialized")
        logger.info(f"📊 Consuming from topic: {KAFKA_TOPIC}")
        logger.info(f"👥 Consumer group: {CONSUMER_GROUP_ID}")
        
        # Consume messages
        for message in consumer:
            try:
                tick_data = message.value
                total_messages += 1
                
                # Calculate latency
                latency_info = calculate_latency(tick_data)
                if latency_info:
                    latency_data.append(latency_info)
                    
                    # Keep only last 1000 records
                    if len(latency_data) > 1000:
                        latency_data = latency_data[-1000:]
                    
                    # Log the latency
                    logger.info(
                        f"📈 Tick {total_messages}: {latency_info['symbol']} @ {latency_info['price']} | "
                        f"T1: {latency_info['t1']}ms, T2: {latency_info['t2']}ms, "
                        f"T3: {latency_info['t3']}ms, Total: {latency_info['total']}ms"
                    )
                    
                    # Print statistics every 100 messages
                    if total_messages % 100 == 0:
                        print_statistics()
                
            except Exception as e:
                logger.error(f"❌ Error processing message: {e}")
                
    except KeyboardInterrupt:
        logger.info("🛑 Consumer stopped by user")
    except Exception as e:
        logger.error(f"❌ Consumer error: {e}")
    finally:
        if 'consumer' in locals():
            consumer.close()
            logger.info("🔌 Kafka consumer closed")
        
        # Final statistics
        if latency_data:
            print_statistics()

if __name__ == "__main__":
    main() 