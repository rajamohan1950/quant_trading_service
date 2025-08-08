#!/usr/bin/env python3
"""
Tick Generator
Generates market tick data and sends to WebSocket server
"""

import asyncio
import websockets
import json
import logging
import os
import random
import time
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
WS_HOST = os.getenv('WS_HOST', 'localhost')
WS_PORT = int(os.getenv('WS_PORT', 8080))
TICK_RATE = int(os.getenv('TICK_RATE', 1000))  # ticks per second
SYMBOLS = os.getenv('SYMBOLS', 'NIFTY,BANKNIFTY,RELIANCE,TCS').split(',')

# Market data simulation
symbol_prices = {
    'NIFTY': 19500,
    'BANKNIFTY': 44500,
    'RELIANCE': 2450,
    'TCS': 3850,
    'INFY': 1450,
    'HDFC': 1650,
    'ICICIBANK': 950,
    'WIPRO': 450,
    'HCLTECH': 1150,
    'TATAMOTORS': 650
}

def generate_tick_data(symbol):
    """Generate realistic tick data for a symbol"""
    base_price = symbol_prices.get(symbol, 1000)
    
    # Simulate price movement (random walk with mean reversion)
    price_change = random.gauss(0, base_price * 0.001)  # 0.1% volatility
    new_price = base_price + price_change
    
    # Update the base price for next tick
    symbol_prices[symbol] = new_price
    
    # Generate tick data
    tick_data = {
        'id': str(uuid.uuid4()),
        'symbol': symbol,
        'price': round(new_price, 2),
        'volume': random.randint(100, 10000),
        'timestamp': datetime.now().isoformat(),
        'bid': round(new_price * 0.999, 2),
        'ask': round(new_price * 1.001, 2),
        'spread': round(new_price * 0.002, 2),
        'source': 'tick_generator'
    }
    
    return tick_data

async def send_tick_data(websocket, symbol):
    """Send tick data to WebSocket server"""
    try:
        tick_data = generate_tick_data(symbol)
        
        # Send to WebSocket
        await websocket.send(json.dumps(tick_data))
        
        # Wait for acknowledgment
        response = await websocket.recv()
        ack = json.loads(response)
        
        if ack.get('status') == 'success':
            logger.info(f"✅ Tick sent: {symbol} @ {tick_data['price']}")
        else:
            logger.error(f"❌ Tick failed: {ack.get('message', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ Error sending tick for {symbol}: {e}")

async def tick_generator():
    """Main tick generator function"""
    logger.info("🚀 Starting Tick Generator...")
    logger.info(f"📊 Target rate: {TICK_RATE} ticks/sec")
    logger.info(f"📈 Symbols: {', '.join(SYMBOLS)}")
    
    # Calculate delay between ticks
    delay = 1.0 / TICK_RATE if TICK_RATE > 0 else 1.0
    
    try:
        # Connect to WebSocket server
        uri = f"ws://{WS_HOST}:{WS_PORT}"
        logger.info(f"🔌 Connecting to WebSocket: {uri}")
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to WebSocket server")
            
            tick_count = 0
            start_time = time.time()
            
            while True:
                try:
                    # Send tick for each symbol
                    for symbol in SYMBOLS:
                        await send_tick_data(websocket, symbol)
                        tick_count += 1
                        
                        # Log progress every 1000 ticks
                        if tick_count % 1000 == 0:
                            elapsed = time.time() - start_time
                            rate = tick_count / elapsed
                            logger.info(f"📈 Generated {tick_count} ticks (rate: {rate:.1f} ticks/sec)")
                    
                    # Wait before next batch
                    await asyncio.sleep(delay)
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.error("🔌 WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in tick generation: {e}")
                    await asyncio.sleep(1)  # Wait before retrying
                    
    except Exception as e:
        logger.error(f"❌ Failed to connect to WebSocket: {e}")

async def main():
    """Main function"""
    try:
        await tick_generator()
    except KeyboardInterrupt:
        logger.info("🛑 Tick generator stopped by user")
    except Exception as e:
        logger.error(f"❌ Tick generator error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 