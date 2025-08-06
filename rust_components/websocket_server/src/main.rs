use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio_tungstenite::{accept_async, tungstenite::protocol::Message};
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// WebSocket server port
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Kafka bootstrap servers
    #[arg(short, long, default_value = "localhost:9092")]
    kafka_bootstrap: String,

    /// Kafka topic
    #[arg(short, long, default_value = "tick-data")]
    kafka_topic: String,
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
    ws_to_kafka_producer: f64,
    kafka_producer_to_consumer: f64,
    end_to_end: f64,
}

struct WebSocketServer {
    clients: Arc<RwLock<HashMap<Uuid, tokio::sync::mpsc::UnboundedSender<Message>>>>,
    kafka_producer: Option<KafkaProducer>,
}

struct KafkaProducer {
    bootstrap_servers: String,
    topic: String,
}

impl KafkaProducer {
    fn new(bootstrap_servers: String, topic: String) -> Self {
        Self {
            bootstrap_servers,
            topic,
        }
    }

    async fn send_message(&self, message: &str) -> Result<f64, Box<dyn std::error::Error>> {
        // Simulate Kafka producer latency
        let start_time = Instant::now();
        
        // In a real implementation, this would send to actual Kafka
        // For now, we'll simulate the latency
        tokio::time::sleep(tokio::time::Duration::from_millis(
            rand::thread_rng().gen_range(1..10)
        )).await;
        
        let latency = start_time.elapsed().as_micros() as f64 / 1000.0;
        Ok(latency)
    }
}

impl WebSocketServer {
    fn new(kafka_bootstrap: String, kafka_topic: String) -> Self {
        let kafka_producer = Some(KafkaProducer::new(kafka_bootstrap, kafka_topic));
        
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            kafka_producer,
        }
    }

    async fn handle_connection(
        &self,
        socket: TcpStream,
        addr: std::net::SocketAddr,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ws_stream = accept_async(socket).await?;
        let (mut ws_sender, mut ws_receiver) = ws_stream.split();
        
        let client_id = Uuid::new_v4();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Message>();
        
        {
            let mut clients = self.clients.write().await;
            clients.insert(client_id, tx);
        }
        
        println!("🔌 New WebSocket connection from {} (ID: {})", addr, client_id);

        // Spawn task to handle incoming messages
        let kafka_producer = self.kafka_producer.clone();
        let clients = self.clients.clone();
        let client_id_clone = client_id;
        
        tokio::spawn(async move {
            while let Some(msg) = ws_receiver.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        if let Ok(tick_data) = serde_json::from_str::<TickData>(&text) {
                            println!("📊 Received tick for {}: ₹{:.2f}", tick_data.symbol, tick_data.price);
                            
                            // Forward to Kafka with latency tracking
                            if let Some(producer) = &kafka_producer {
                                let start_time = Instant::now();
                                
                                match producer.send_message(&text).await {
                                    Ok(latency) => {
                                        let total_latency = start_time.elapsed().as_micros() as f64 / 1000.0;
                                        println!("📈 Forwarded to Kafka - Latency: {:.2f}ms", total_latency);
                                        
                                        // Broadcast to all connected clients
                                        let mut clients = clients.write().await;
                                        for (id, sender) in clients.iter() {
                                            if *id != client_id_clone {
                                                if let Err(e) = sender.send(Message::Text(text.clone())) {
                                                    eprintln!("❌ Error sending to client {}: {}", id, e);
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("❌ Error forwarding to Kafka: {}", e);
                                    }
                                }
                            }
                        }
                    }
                    Ok(Message::Close(_)) => {
                        println!("🔌 Client {} disconnected", client_id);
                        break;
                    }
                    Err(e) => {
                        eprintln!("❌ WebSocket error for client {}: {}", client_id, e);
                        break;
                    }
                    _ => {}
                }
            }
            
            // Remove client from active connections
            let mut clients = clients.write().await;
            clients.remove(&client_id_clone);
        });

        // Handle outgoing messages
        while let Some(msg) = rx.recv().await {
            if let Err(e) = ws_sender.send(msg).await {
                eprintln!("❌ Error sending message to client {}: {}", client_id, e);
                break;
            }
        }

        Ok(())
    }

    async fn start(&self, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        let addr = format!("0.0.0.0:{}", port);
        let listener = TcpListener::bind(&addr).await?;
        
        println!("🚀 WebSocket server listening on {}", addr);
        println!("📊 Kafka Producer: {}", self.kafka_producer.as_ref().unwrap().bootstrap_servers);
        println!("📈 Kafka Topic: {}", self.kafka_producer.as_ref().unwrap().topic);

        while let Ok((stream, addr)) = listener.accept().await {
            let server = self.clone();
            tokio::spawn(async move {
                if let Err(e) = server.handle_connection(stream, addr).await {
                    eprintln!("❌ Error handling connection: {}", e);
                }
            });
        }

        Ok(())
    }
}

impl Clone for WebSocketServer {
    fn clone(&self) -> Self {
        Self {
            clients: self.clients.clone(),
            kafka_producer: self.kafka_producer.clone(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    
    println!("🚀 Starting WebSocket Server");
    println!("📊 Port: {}", args.port);
    println!("📈 Kafka Bootstrap: {}", args.kafka_bootstrap);
    println!("📋 Kafka Topic: {}", args.kafka_topic);

    let server = WebSocketServer::new(args.kafka_bootstrap, args.kafka_topic);
    server.start(args.port).await?;

    Ok(())
} 