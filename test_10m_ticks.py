#!/usr/bin/env python3
"""
Test script for tick generation (max 1000 ticks)
"""

import asyncio
import websockets
import json
import time
from datetime import datetime
import random
import sys

async def generate_test_ticks(num_ticks=1000, rate=100):
    """Generate specified number of ticks at given rate (max 1000)"""
    
    # Enforce maximum limit
    if num_ticks > 1000:
        print(f"⚠️  Limiting test to 1000 ticks (requested: {num_ticks:,})")
        num_ticks = 1000
    
    print(f"🚀 Starting {num_ticks:,} tick generation test at {rate} ticks/sec")
    
    symbols = ["NIFTY", "TCS", "RELIANCE", "HDFC"]
    symbol_prices = {"NIFTY": 20800, "TCS": 4000, "RELIANCE": 2400, "HDFC": 1600}
    
    start_time = time.time()
    successful_ticks = 0
    failed_ticks = 0
    
    try:
        uri = "ws://localhost:8080"
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to WebSocket server")
            
            for i in range(num_ticks):
                symbol = random.choice(symbols)
                
                # Generate realistic tick
                base_price = symbol_prices[symbol]
                price_change = random.gauss(0, base_price * 0.001)
                new_price = base_price + price_change
                symbol_prices[symbol] = new_price
                
                tick_data = {
                    'id': f"test-{i+1}",
                    'symbol': symbol,
                    'price': round(new_price, 2),
                    'volume': random.randint(100, 10000),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'performance_test'
                }
                
                try:
                    # Send tick
                    await websocket.send(json.dumps(tick_data))
                    
                    # Wait for ack
                    response = await websocket.recv()
                    ack = json.loads(response)
                    
                    if ack.get('status') == 'success':
                        successful_ticks += 1
                    else:
                        failed_ticks += 1
                        print(f"❌ Tick {i+1} failed: {ack.get('message')}")
                    
                    # Progress reporting
                    if (i + 1) % 1000 == 0:
                        elapsed = time.time() - start_time
                        current_rate = (i + 1) / elapsed
                        print(f"📈 Progress: {i+1:,}/{num_ticks:,} ticks ({current_rate:.1f} ticks/sec)")
                    
                    # Rate limiting
                    if rate > 0:
                        await asyncio.sleep(1.0 / rate)
                        
                except Exception as e:
                    failed_ticks += 1
                    print(f"❌ Error on tick {i+1}: {e}")
                    
        # Final statistics
        total_time = time.time() - start_time
        actual_rate = successful_ticks / total_time
        
        print(f"\n🎉 Test Completed!")
        print(f"📊 Results:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Successful ticks: {successful_ticks:,}")
        print(f"  Failed ticks: {failed_ticks:,}")
        print(f"  Actual rate: {actual_rate:.1f} ticks/sec")
        print(f"  Success rate: {successful_ticks/num_ticks*100:.1f}%")
        
        if successful_ticks > 0:
            print(f"\n✅ Pipeline is working! Check Kafka consumer for latency metrics.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Command line arguments
    requested_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rate = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Enforce 1000 tick limit
    num_ticks = min(requested_ticks, 1000)
    
    print(f"🧪 Tick Generation Test (Max 1000)")
    if requested_ticks > 1000:
        print(f"⚠️  Requested {requested_ticks:,} ticks, limiting to {num_ticks:,}")
    print(f"📋 Parameters: {num_ticks:,} ticks at {rate} ticks/sec")
    
    asyncio.run(generate_test_ticks(num_ticks, rate)) 