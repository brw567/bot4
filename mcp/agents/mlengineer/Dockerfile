# Dockerfile for MLEngineer MCP Agent
FROM rust:1.75-alpine AS builder

RUN apk add --no-cache musl-dev openssl-dev pkgconfig clang llvm

WORKDIR /build

# Copy and build dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    cargo build --release && rm -rf src

# Copy source and build
COPY src ./src
RUN cargo build --release && \
    strip /build/target/release/bot4-agent-mlengineer

# Runtime stage
FROM alpine:3.19

RUN apk add --no-cache ca-certificates openssl libgcc && \
    adduser -D -u 1000 agent

COPY --from=builder /build/target/release/bot4-agent-mlengineer /usr/local/bin/mlengineer

ENV RUST_LOG=info \
    REDIS_URL=redis://redis:6379 \
    MODEL_CACHE_SIZE=100 \
    MAX_INFERENCE_BATCH=32

USER agent

EXPOSE 8082

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD nc -z localhost 8082 || exit 1

CMD ["/usr/local/bin/mlengineer"]