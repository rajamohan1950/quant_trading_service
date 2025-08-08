# Use a newer Rust version
FROM rust:1.82 as builder

WORKDIR /usr/src/app
COPY . .

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/app/target/release/kafka_consumer /usr/local/bin/

CMD ["kafka_consumer"] 