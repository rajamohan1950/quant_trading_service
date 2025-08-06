use crate::generator::tick_stream;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use tokio::sync::Notify;

pub async fn start_server() {
    println!("✅ start_server() called");

    let tick_counter = Arc::new(AtomicU64::new(0));
    let notify = Arc::new(Notify::new());

    tick_stream(
        move |_tick| {
            // No-op tick handler
        },
        notify,
        tick_counter.clone(),
    )
    .await;
}
