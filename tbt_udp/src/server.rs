use std::net::UdpSocket;
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::VecDeque;

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct TickMessage {
    pub timestamp_ns: u64,
    pub token: u32,
    pub price: f32,
    pub quantity: u32,
    pub side: u8,
    pub event_type: u8,
    pub order_id: u64,
    pub _padding: u16,
}

fn percentile(sorted: &[u128], p: f64) -> u128 {
    let rank = (p * sorted.len() as f64).ceil() as usize - 1;
    sorted.get(rank).copied().unwrap_or(0)
}

pub fn run() -> std::io::Result<()> {
    let socket = UdpSocket::bind("0.0.0.0:9000")?;
    let mut buf = [0u8; std::mem::size_of::<TickMessage>()];

    let mut latencies: VecDeque<u128> = VecDeque::with_capacity(5000);
    let mut last_report = std::time::Instant::now();

    loop {
        let (amt, _src) = socket.recv_from(&mut buf)?;
        if amt == std::mem::size_of::<TickMessage>() {
            let recv_ns = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;

            let tick: TickMessage = unsafe { std::ptr::read(buf.as_ptr() as *const _) };

            let latency = (recv_ns - tick.timestamp_ns) as u128;
            latencies.push_back(latency);

            // Every 1 second, compute and print percentiles
            if last_report.elapsed().as_secs_f64() >= 1.0 {
                let mut sorted: Vec<u128> = latencies.drain(..).collect();
                sorted.sort_unstable();

                let p50 = percentile(&sorted, 0.50);
                let p95 = percentile(&sorted, 0.95);
                let p99 = percentile(&sorted, 0.99);

                println!(
                    "Latency (ms): P50 = {:.3}, P95 = {:.3}, P99 = {:.3}  [count: {}]",
                    p50 as f64 / 1_000_000.0,
                    p95 as f64 / 1_000_000.0,
                    p99 as f64 / 1_000_000.0,
                    sorted.len()
                );

                last_report = std::time::Instant::now();
            }
        }
    }
}

