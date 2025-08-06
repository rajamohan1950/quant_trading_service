use rdkafka::config::ClientConfig;
use rdkafka::consumer::{StreamConsumer, Consumer};
use rdkafka::message::Message;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::StreamExt;
use std::fs::OpenOptions;
use std::io::Write;

pub async fn start_kafka_consumer() {
    println!("🧪 Starting Kafka consumer on topic tbt_raw_ticks...");

    let consumer: StreamConsumer = ClientConfig::new()
        .set("group.id", "tbt_consumer_group")
        .set("bootstrap.servers", "localhost:9092")
        .set("enable.partition.eof", "false")
        .create()
        .expect("Failed to create Kafka consumer");

    consumer.subscribe(&["tbt_raw_ticks"]).expect("Failed to subscribe");

    let mut stream = consumer.stream();
    let mut log_file = OpenOptions::new().create(true).append(true).open("latency.log").unwrap();

    while let Some(result) = stream.next().await {
        match result {
            Ok(msg) => {
                if let Some(payload) = msg.payload() {
                    if payload.len() == 32 {
                        // Extract t0 (first 8 bytes, u64 big-endian)
                        let t0 = u64::from_be_bytes(payload[0..8].try_into().unwrap());
                        let t1 = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_nanos();

                        let latency_ns = t1 as u64 - t0;

                        writeln!(log_file, "{}", latency_ns).unwrap();
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Kafka error: {:?}", e);
            }
        }
    }
}
