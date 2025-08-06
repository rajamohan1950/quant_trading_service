
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use rdkafka::config::ClientConfig;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::message::Message;
use rdkafka::util::get_rdkafka_version;

fn main() {
    let (version_n, version_s) = get_rdkafka_version();
    println!("rd_kafka_version: 0x{:08x}, {}", version_n, version_s);

    let consumer: StreamConsumer = ClientConfig::new()
        .set("group.id", "tick-receiver-group")
        .set("bootstrap.servers", "localhost:9092")
        .set("auto.offset.reset", "earliest")
        .create()
        .expect("Consumer creation failed");

    consumer
        .subscribe(&["tbt_raw_ticks"])
        .expect("Can't subscribe to topic");

    let mut log_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("latency.log")
        .expect("Failed to open latency.log");

    println!("Tick Receiver is running. Waiting for messages...");

    for message in consumer.iter() {
        match message {
            Ok(m) => {
                if let Some(payload) = m.payload() {
                    if payload.len() >= 8 {
                        let t0_ns = u64::from_be_bytes(payload[0..8].try_into().unwrap());
                        let t1 = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64;
                        let latency = t1.saturating_sub(t0_ns);

                        let log_entry = format!("latency: {} ns\n", latency);
                        print!("{}", log_entry);
                        log_file.write_all(log_entry.as_bytes()).unwrap();
                    }
                }
            }
            Err(e) => eprintln!("Kafka error: {}", e),
        }
    }
}
