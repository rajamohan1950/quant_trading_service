#!/usr/bin/env python3
"""
Test script to verify the complete tick generation pipeline
"""

import asyncio
import websockets
import json
import time
from datetime import datetime

async def test_pipeline():
    """Test the complete pipeline by sending test ticks"""
    
    print("🧪 Testing Tick Generation Pipeline...")
    
    # Test data
    test_ticks = [
        {
            "id": "test-1",
            "symbol": "TESTSTOCK",
            "price": 1000.50,
            "volume": 1000,
            "timestamp": datetime.now().isoformat(),
            "source": "pipeline_test"
        },
        {
            "id": "test-2", 
            "symbol": "TESTSTOCK",
            "price": 1001.25,
            "volume": 1500,
            "timestamp": datetime.now().isoformat(),
            "source": "pipeline_test"
        }
    ]
    
    try:
        # Connect to WebSocket server
        uri = "ws://localhost:8080"
        print(f"🔌 Connecting to WebSocket: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket server")
            
            for i, tick in enumerate(test_ticks, 1):
                print(f"📤 Sending test tick {i}: {tick['symbol']} @ {tick['price']}")
                
                # Send tick data
                await websocket.send(json.dumps(tick))
                
                # Wait for acknowledgment
                response = await websocket.recv()
                ack = json.loads(response)
                
                if ack.get('status') == 'success':
                    print(f"✅ Tick {i} successfully processed!")
                    print(f"   Kafka partition: {ack.get('kafka_partition')}")
                    print(f"   Kafka offset: {ack.get('kafka_offset')}")
                else:
                    print(f"❌ Tick {i} failed: {ack.get('message')}")
                
                # Small delay between ticks
                await asyncio.sleep(1)
                
        print("\n🎉 Pipeline test completed successfully!")
        print("📊 Check the Kafka consumer logs to see latency metrics")
        print("📱 Check the Streamlit app at http://localhost:8501")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_pipeline()) 