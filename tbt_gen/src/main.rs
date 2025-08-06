mod tick;
mod generator;
mod websocket_server;
mod websocket_client;
mod kafka_consumer;

use tokio::signal;

#[tokio::main]
async fn main() {
    println!("✅ main.rs started");
    tokio::select! {
        _ = websocket_server::start_server() => {},
        _ = websocket_client::start_websocket_client("localhost:9092") => {},
        _ = kafka_consumer::start_kafka_consumer() => {},
        _ = signal::ctrl_c() => {
            println!("Shutting down.");
        }
    }
}
