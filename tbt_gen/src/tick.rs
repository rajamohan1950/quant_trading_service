use bytes::{BufMut, BytesMut};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Tick {
    pub timestamp: u64,
    pub symbol_id: u32,
    pub price: u64,
    pub size: u32,
    pub flags: u16,
    pub sequence: u32,
    pub exchange_id: u16,
}

impl Tick {
    pub fn new(symbol_id: u32, price: u64, size: u32, sequence: u32, exchange_id: u16, flags: u16) -> Self {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
        Tick {
            timestamp,
            symbol_id,
            price,
            size,
            flags,
            sequence,
            exchange_id,
        }
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        let mut buf = BytesMut::with_capacity(32);
        buf.put_u64(self.timestamp);
        buf.put_u32(self.symbol_id);
        buf.put_u64(self.price);
        buf.put_u32(self.size);
        buf.put_u16(self.flags);
        buf.put_u32(self.sequence);
        buf.put_u16(self.exchange_id);
        let mut out = [0u8; 32];
        out.copy_from_slice(&buf[..]);
        out
    }
}
