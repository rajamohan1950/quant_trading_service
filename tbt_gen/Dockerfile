
FROM rust:1.77-bullseye as builder

# Install native dependencies
RUN apt-get update && apt-get install -y \
    cmake pkg-config libssl-dev librdkafka-dev

WORKDIR /app

COPY . .

# Build the release binary
RUN cargo build --release

# Slim final image
FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y \
    librdkafka1 libssl1.1 zstd && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/tick_generator /tick_generator

ENV KAFKA_BROKER=kafka:9092
CMD ["/tick_generator"]

