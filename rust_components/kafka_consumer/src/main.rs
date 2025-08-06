use clap::Parser;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::config::ClientConfig;
use rdkafka::message::Message as KafkaMessage;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::time::Duration;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Kafka bootstrap servers
    #[arg(short, long, default_value = "localhost:9092")]
    kafka_bootstrap: String,

    /// Kafka topic
    #[arg(short, long, default_value = "tick-data")]
    kafka_topic: String,

    /// Consumer group ID
    #[arg(short, long, default_value = "latency-monitor")]
    group_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct TickData {
    timestamp: String,
    symbol: String,
    price: f64,
    volume: u64,
    bid: f64,
    ask: f64,
    bid_size: u64,
    ask_size: u64,
    latency_t1: f64,
    latency_t2: f64,
    latency_t3: f64,
    total_latency: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct LatencyMetrics {
    tick_generator_to_ws: f64,
    ws_to_kafka_producer: f64,
    kafka_producer_to_consumer: f64,
    end_to_end: f64,
}

struct KafkaConsumer {
    consumer: StreamConsumer,
    topic: String,
    latency_stats: LatencyStats,
}

struct LatencyStats {
    total_messages: u64,
    total_latency: f64,
    min_latency: f64,
    max_latency: f64,
    avg_latency: f64,
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            total_messages: 0,
            total_latency: 0.0,
            min_latency: f64::MAX,
            max_latency: 0.0,
            avg_latency: 0.0,
        }
    }

    fn update(&mut self, latency: f64) {
        self.total_messages += 1;
        self.total_latency += latency;
        self.min_latency = self.min_latency.min(latency);
        self.max_latency = self.max_latency.max(latency);
        self.avg_latency = self.total_latency / self.total_messages as f64;
    }

    fn print_stats(&self) {
        println!("📊 Latency Statistics:");
        println!("   📈 Total Messages: {}", self.total_messages);
        println!("   ⚡ Average Latency: {:.2f}ms", self.avg_latency);
        println!("   📉 Min Latency: {:.2f}ms", self.min_latency);
        println!("   📈 Max Latency: {:.2f}ms", self.max_latency);
    }
}

impl KafkaConsumer {
    async fn new(bootstrap_servers: String, topic: String, group_id: String) -> Result<Self, Box<dyn std::error::Error>> {
        let consumer: StreamConsumer = ClientConfig::new()
            .set("group.id", &group_id)
            .set("bootstrap.servers", &bootstrap_servers)
            .set("enable.partition.eof", "false")
            .set("session.timeout.ms", "6000")
            .set("enable.auto.commit", "true")
            .set("auto.offset.reset", "earliest")
            .create()?;

        Ok(Self {
            consumer,
            topic,
            latency_stats: LatencyStats::new(),
        })
    }

    async fn start_consuming(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.consumer.subscribe(&[&self.topic])?;
        
        println!("🚀 Starting Kafka Consumer");
        println!("📊 Topic: {}", self.topic);
        println!("📈 Consumer Group: latency-monitor");
        println!("🎯 Waiting for messages...");

        let start_time = Instant::now();
        let mut message_count = 0;

        loop {
            match self.consumer.recv().await {
                Ok(msg) => {
                    let payload = match msg.payload() {
                        Some(p) => p,
                        None => {
                            eprintln!("❌ Empty message received");
                            continue;
                        }
                    };

                    let message_str = match std::str::from_utf8(payload) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("❌ Invalid UTF-8 in message: {}", e);
                            continue;
                        }
                    };

                    if let Ok(tick_data) = serde_json::from_str::<TickData>(message_str) {
                        message_count += 1;
                        
                        // Calculate end-to-end latency
                        let end_to_end_latency = tick_data.total_latency;
                        self.latency_stats.update(end_to_end_latency);

                        println!("📊 Message #{} - {}: ₹{:.2f} (E2E: {:.2f}ms)", 
                                message_count, tick_data.symbol, tick_data.price, end_to_end_latency);

                        // Print detailed latency breakdown every 10 messages
                        if message_count % 10 == 0 {
                            println!("📈 Latency Breakdown:");
                            println!("   T1 (Tick→WS): {:.2f}ms", tick_data.latency_t1);
                            println!("   T2 (WS→Kafka): {:.2f}ms", tick_data.latency_t2);
                            println!("   T3 (Kafka→Consumer): {:.2f}ms", tick_data.latency_t3);
                            println!("   Total (E2E): {:.2f}ms", tick_data.total_latency);
                            println!("   Running Avg: {:.2f}ms", self.latency_stats.avg_latency);
                            println!("---");
                        }

                        // Print stats every 100 messages
                        if message_count % 100 == 0 {
                            self.latency_stats.print_stats();
                            println!("⏱️  Runtime: {:.1f}s", start_time.elapsed().as_secs_f64());
                            println!("---");
                        }
                    } else {
                        eprintln!("❌ Failed to parse tick data: {}", message_str);
                    }
                }
                Err(e) => {
                    eprintln!("❌ Error receiving message: {}", e);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    
    println!("🚀 Starting Kafka Consumer");
    println!("📊 Bootstrap Servers: {}", args.kafka_bootstrap);
    println!("📈 Topic: {}", args.kafka_topic);
    println!("📋 Group ID: {}", args.group_id);

    let mut consumer = KafkaConsumer::new(
        args.kafka_bootstrap,
        args.kafka_topic,
        args.group_id,
    ).await?;

    consumer.start_consuming().await?;

    Ok(())
} 