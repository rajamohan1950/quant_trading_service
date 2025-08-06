
use std::net::UdpSocket;
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use std::thread::sleep;

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

pub fn run() -> std::io::Result<()> {
    let socket = UdpSocket::bind("0.0.0.0:0")?;
    socket.connect("127.0.0.1:9000")?;

    let mut order_id = 1;
    let sleep_duration = Duration::from_micros(666); // ~1500 ticks/sec

    loop {
        let sys_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let tick = TickMessage {
            timestamp_ns: sys_ts,
            token: 12345,
            price: 19575.25,
            quantity: 50,
            side: 0,
            event_type: 1,
            order_id,
            _padding: 0,
        };

        let bytes = unsafe {
            std::slice::from_raw_parts(
                (&tick as *const TickMessage) as *const u8,
                std::mem::size_of::<TickMessage>(),
            )
        };

        socket.send(bytes)?;
        order_id += 1;

        sleep(sleep_duration);
    }
}

