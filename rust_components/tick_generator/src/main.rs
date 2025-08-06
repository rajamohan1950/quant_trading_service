use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::time::interval;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use url::Url;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// WebSocket server URL
    #[arg(short, long, default_value = "ws://localhost:8080")]
    websocket_url: String,

    /// Tick rate per second
    #[arg(short, long, default_value_t = 100)]
    tick_rate: u64,

    /// Symbols to generate ticks for
    #[arg(short, long, default_value = "NIFTY,BANKNIFTY,RELIANCE,TCS")]
    symbols: String,

    /// Test duration in seconds
    #[arg(short, long, default_value_t = 60)]
    duration: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct TickData {
    timestamp: String,
    symbol: String,
    price: f64,
    volume: u64,
    bid: f64,
    ask: f64,
    bid_size: u64,
    ask_size: u64,
    latency_t1: f64, // Tick Generator → WebSocket
    latency_t2: f64, // WebSocket → Kafka Producer
    latency_t3: f64, // Kafka Producer → Consumer
    total_latency: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct LatencyMetrics {
    tick_generator_to_ws: f64,
    ws_to_kafka_producer: f64,
    kafka_producer_to_consumer: f64,
    end_to_end: f64,
}

struct TickGenerator {
    symbols: Vec<String>,
    base_prices: std::collections::HashMap<String, f64>,
    rng: rand::rngs::ThreadRng,
}

impl TickGenerator {
    fn new(symbols: Vec<String>) -> Self {
        let mut base_prices = std::collections::HashMap::new();
        let mut rng = rand::thread_rng();

        // Initialize base prices for each symbol
        for symbol in &symbols {
            let base_price = match symbol.as_str() {
                "NIFTY" => 19000.0,
                "BANKNIFTY" => 45000.0,
                "RELIANCE" => 2500.0,
                "TCS" => 3500.0,
                "INFY" => 1500.0,
                "HDFC" => 1600.0,
                "ICICIBANK" => 900.0,
                _ => 1000.0,
            };
            base_prices.insert(symbol.clone(), base_price);
        }

        Self {
            symbols,
            base_prices,
            rng,
        }
    }

    fn generate_tick(&mut self, symbol: &str) -> TickData {
        let base_price = self.base_prices.get(symbol).unwrap_or(&1000.0);
        
        // Generate realistic price movement
        let price_change = self.rng.gen_range(-0.5..0.5) * base_price * 0.01;
        let new_price = base_price + price_change;
        
        // Update base price
        self.base_prices.insert(symbol.to_string(), new_price);
        
        let spread = new_price * 0.001; // 0.1% spread
        let bid = new_price - spread / 2.0;
        let ask = new_price + spread / 2.0;
        
        let volume = self.rng.gen_range(100..10000);
        let bid_size = self.rng.gen_range(100..5000);
        let ask_size = self.rng.gen_range(100..5000);

        TickData {
            timestamp: chrono::Utc::now().to_rfc3339(),
            symbol: symbol.to_string(),
            price: new_price,
            volume,
            bid,
            ask,
            bid_size,
            ask_size,
            latency_t1: 0.0,
            latency_t2: 0.0,
            latency_t3: 0.0,
            total_latency: 0.0,
        }
    }

    fn get_random_symbol(&mut self) -> String {
        self.symbols[self.rng.gen_range(0..self.symbols.len())].clone()
    }
}

async fn send_tick_data(websocket_url: &str, tick_data: TickData) -> Result<f64, Box<dyn std::error::Error>> {
    let url = Url::parse(websocket_url)?;
    let (ws_stream, _) = connect_async(url).await?;
    
    let (mut write, _read) = ws_stream.split();
    
    let start_time = Instant::now();
    
    let message = serde_json::to_string(&tick_data)?;
    write.send(Message::Text(message)).await?;
    
    let latency = start_time.elapsed().as_micros() as f64 / 1000.0; // Convert to milliseconds
    Ok(latency)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    
    println!("🚀 Starting Tick Generator");
    println!("📊 WebSocket URL: {}", args.websocket_url);
    println!("⚡ Tick Rate: {} ticks/sec", args.tick_rate);
    println!("📈 Symbols: {}", args.symbols);
    println!("⏱️ Duration: {} seconds", args.duration);

    let symbols: Vec<String> = args.symbols.split(',').map(|s| s.trim().to_string()).collect();
    let mut tick_generator = TickGenerator::new(symbols);

    let interval_duration = Duration::from_millis(1000 / args.tick_rate);
    let mut interval = interval(interval_duration);
    
    let start_time = Instant::now();
    let mut tick_count = 0;
    let mut total_latency = 0.0;
    let mut min_latency = f64::MAX;
    let mut max_latency = 0.0;

    println!("🎯 Starting tick generation...");

    while start_time.elapsed().as_secs() < args.duration {
        interval.tick().await;

        let symbol = tick_generator.get_random_symbol();
        let mut tick_data = tick_generator.generate_tick(&symbol);

        // Measure latency T1 (Tick Generator → WebSocket)
        let t1_start = Instant::now();
        match send_tick_data(&args.websocket_url, tick_data.clone()).await {
            Ok(latency) => {
                tick_data.latency_t1 = latency;
                
                // Simulate T2 and T3 latencies (in real implementation, these would be measured)
                tick_data.latency_t2 = rand::thread_rng().gen_range(0.5..5.0);
                tick_data.latency_t3 = rand::thread_rng().gen_range(1.0..10.0);
                tick_data.total_latency = tick_data.latency_t1 + tick_data.latency_t2 + tick_data.latency_t3;

                total_latency += tick_data.total_latency;
                min_latency = min_latency.min(tick_data.total_latency);
                max_latency = max_latency.max(tick_data.total_latency);

                tick_count += 1;

                if tick_count % 100 == 0 {
                    println!("📊 Sent {} ticks, Avg Latency: {:.2f}ms", 
                            tick_count, total_latency / tick_count as f64);
                }
            }
            Err(e) => {
                eprintln!("❌ Error sending tick data: {}", e);
            }
        }
    }

    println!("\n📈 Tick Generation Complete!");
    println!("📊 Total Ticks Sent: {}", tick_count);
    println!("⚡ Average Latency: {:.2f}ms", total_latency / tick_count as f64);
    println!("📉 Min Latency: {:.2f}ms", min_latency);
    println!("📈 Max Latency: {:.2f}ms", max_latency);
    println!("🎯 Actual Tick Rate: {:.2f} ticks/sec", 
             tick_count as f64 / start_time.elapsed().as_secs() as f64);

    Ok(())
} 