# Dockerfile for RiskQuant MCP Agent
FROM rust:1.75-alpine AS builder

RUN apk add --no-cache musl-dev openssl-dev pkgconfig

WORKDIR /build

# Copy and build dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    cargo build --release && rm -rf src

# Copy source and build
COPY src ./src
RUN cargo build --release && \
    strip /build/target/release/bot4-agent-riskquant

# Runtime stage
FROM alpine:3.19

RUN apk add --no-cache ca-certificates openssl libgcc && \
    adduser -D -u 1000 agent

COPY --from=builder /build/target/release/bot4-agent-riskquant /usr/local/bin/riskquant

ENV RUST_LOG=info \
    REDIS_URL=redis://redis:6379 \
    MAX_POSITION_SIZE=0.02 \
    MAX_PORTFOLIO_RISK=0.15 \
    MAX_CORRELATION=0.7

USER agent

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD nc -z localhost 8081 || exit 1

CMD ["/usr/local/bin/riskquant"]