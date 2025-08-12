#!/usr/bin/env python3
"""
Manual test to send one tick and see the data format
"""

import asyncio
import websockets
import json
import time
from datetime import datetime

async def send_test_tick():
    uri = "ws://localhost:8080"
    
    # Generate test tick data
    tick_data = {
        'id': 'test-123',
        'symbol': 'TEST',
        'price': 100.50,
        'volume': 1000,
        'tick_generated_at': datetime.now().isoformat(),
        'bid': 100.49,
        'ask': 100.51,
        'spread': 0.02,
        'source': 'manual_test'
    }
    
    print(f"🚀 Sending test tick: {tick_data}")
    
    try:
        async with websockets.connect(uri) as websocket:
            # Send tick data
            await websocket.send(json.dumps(tick_data))
            
            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            
            print(f"✅ Response: {response_data}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(send_test_tick()) 