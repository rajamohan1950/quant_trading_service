use crate::tick::Tick;
use rand::{thread_rng, Rng};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

pub fn generate_tick(symbol_id: u32, sequence: u32) -> Tick {
    let mut rng = thread_rng();
    Tick::new(
        symbol_id,
        rng.gen_range(100_000..200_000),
        rng.gen_range(1..1000),
        sequence,
        1,
        0,
    )
}

pub async fn tick_stream<F>(mut send: F, _notify: Arc<Notify>, tick_counter: Arc<AtomicU64>)
where
    F: FnMut([u8; 32]) + Send + 'static,
{
    let mut sequence = 0;
    let start_time = Instant::now();

    // Stats logger
    tokio::spawn({
        let tick_counter = tick_counter.clone();
        async move {
            loop {
                let ticks = tick_counter.load(Ordering::Relaxed);
                let elapsed = start_time.elapsed().as_secs_f64();
                let tps = (ticks as f64 / elapsed).round();
                let mb = (ticks as f64 * 32.0) / (1024.0 * 1024.0);
                println!(
                    "[TickGen] Uptime: {:>6.2}s | Ticks: {:>10} | Rate: {:>8} tps | Size: {:>7.2} MB",
                    elapsed, ticks, tps, mb
                );
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    });

    println!("🚀 tick_stream() started");
    loop {
        for symbol_id in 1..=1000 {
            println!("✔ generating tick for symbol {}", symbol_id);
            let tick = generate_tick(symbol_id, sequence);
            send(tick.to_bytes());
            tick_counter.fetch_add(1, Ordering::Relaxed);
            sequence += 1;
        }
        tokio::time::sleep(Duration::from_micros(100)).await;
    }
}
