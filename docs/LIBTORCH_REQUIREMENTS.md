# LibTorch Requirements and Configuration
## ML Infrastructure Documentation
## Date: August 24, 2025

---

## Overview

LibTorch is the C++ distribution of PyTorch, required for our ML components including:
- rust-bert for sentiment analysis
- candle for neural network operations
- Deep learning models for market prediction
- Reinforcement learning for strategy optimization

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **Architecture**: x86_64
- **RAM**: 8GB minimum (16GB+ recommended for training)
- **Disk**: 2GB for LibTorch installation
- **Compiler**: GCC 7.3+ or Clang 5.0+

### Supported Versions
- **LibTorch**: 1.8.0 - 1.13.1
- **CUDA**: Optional (11.7 or 11.8 for GPU acceleration)
- **cuDNN**: Optional (8.5+ for GPU)

## Installation Methods

### Method 1: System Packages (Recommended)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libtorch1.8 libtorch-dev libtorch-test

# The libraries will be installed to:
# /usr/lib/x86_64-linux-gnu/libtorch.so
# /usr/lib/x86_64-linux-gnu/libc10.so
# /usr/lib/x86_64-linux-gnu/libtorch_cpu.so
```

### Method 2: Using PyTorch Installation
If you have PyTorch installed via pip:
```bash
# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Use PyTorch's LibTorch
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

### Method 3: Manual Download
```bash
# Download LibTorch (CPU version)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip

# Extract
unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip -d /opt/

# Set environment
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

### Method 4: Using Setup Script
```bash
# Run our automated setup script
sudo ./scripts/setup_libtorch.sh

# This will:
# 1. Detect existing installations
# 2. Install if needed
# 3. Configure environment variables
# 4. Verify installation
```

## Environment Configuration

### Required Environment Variables
```bash
# Primary LibTorch path
export LIBTORCH=/usr/lib/x86_64-linux-gnu  # or your installation path

# Include directories (optional, auto-detected)
export LIBTORCH_INCLUDE=$LIBTORCH/include

# Library path
export LD_LIBRARY_PATH=$LIBTORCH:$LD_LIBRARY_PATH

# For PyTorch users
export LIBTORCH_USE_PYTORCH=1  # Use PyTorch's LibTorch
export LIBTORCH_BYPASS_VERSION_CHECK=1  # Skip version checks
```

### Persistent Configuration
Add to `~/.bashrc` or `~/.zshrc`:
```bash
# Bot4 LibTorch Configuration
export LIBTORCH=/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$LIBTORCH:$LD_LIBRARY_PATH
```

Or use our environment file:
```bash
source ~/.bot4_env
```

## Rust Integration

### Cargo Configuration
The following crates depend on LibTorch:

#### Direct Dependencies
- `tch` - Rust bindings for LibTorch
- `rust-bert` - BERT models for NLP
- `candle-core` - Neural network framework

#### In Cargo.toml
```toml
[dependencies]
tch = "0.13"
rust-bert = "0.21"
candle-core = "0.3"
```

### Build Commands
```bash
# Build with LibTorch
LIBTORCH=/usr/lib/x86_64-linux-gnu cargo build --release

# Build specific crate
cargo build -p ml --release

# Run tests
cargo test --all
```

## Troubleshooting

### Common Issues

#### 1. LibTorch Not Found
```
Error: Cannot find a libtorch install
```
**Solution**: Set LIBTORCH environment variable:
```bash
export LIBTORCH=/usr/lib/x86_64-linux-gnu
```

#### 2. Version Mismatch
```
Error: LibTorch version mismatch
```
**Solution**: Use bypass flag:
```bash
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

#### 3. Missing Symbols
```
Error: undefined symbol: _ZN3c10...
```
**Solution**: Update LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

#### 4. CUDA Issues
If you don't have CUDA but LibTorch expects it:
```bash
# Force CPU-only version
export TORCH_CUDA_ARCH_LIST=""
export CUDA_VISIBLE_DEVICES=""
```

### Verification Commands

#### Check Installation
```bash
# Check if LibTorch is installed
ls -la /usr/lib/x86_64-linux-gnu/libtorch*

# Check with pkg-config
pkg-config --libs torch

# Test with our script
./scripts/setup_libtorch.sh --verify-only
```

#### Test Rust Build
```bash
# Test ML crate
cd rust_core
LIBTORCH=/usr/lib/x86_64-linux-gnu cargo check -p ml

# Test data_intelligence crate
cargo check -p data_intelligence
```

## Performance Optimization

### CPU Optimization
```bash
# Enable CPU optimizations
export OMP_NUM_THREADS=8  # Set to number of cores
export MKL_NUM_THREADS=8
export TORCH_NUM_THREADS=8

# Use Intel MKL if available
export MKL_SERVICE_FORCE_INTEL=1
```

### Memory Management
```bash
# Limit memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable memory efficient mode
export TOKENIZERS_PARALLELISM=false  # For rust-bert
```

## Docker Configuration

### Dockerfile Example
```dockerfile
FROM ubuntu:22.04

# Install LibTorch
RUN apt-get update && \
    apt-get install -y \
    libtorch1.8 \
    libtorch-dev \
    libtorch-test

# Set environment
ENV LIBTORCH=/usr/lib/x86_64-linux-gnu
ENV LD_LIBRARY_PATH=$LIBTORCH:$LD_LIBRARY_PATH

# Copy and build
COPY . /app
WORKDIR /app/rust_core
RUN cargo build --release
```

### Docker Compose
```yaml
services:
  bot4:
    build: .
    environment:
      - LIBTORCH=/usr/lib/x86_64-linux-gnu
      - LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    volumes:
      - ./models:/app/models
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Setup LibTorch
  run: |
    sudo apt-get update
    sudo apt-get install -y libtorch1.8 libtorch-dev
    echo "LIBTORCH=/usr/lib/x86_64-linux-gnu" >> $GITHUB_ENV
    
- name: Build with LibTorch
  env:
    LIBTORCH: /usr/lib/x86_64-linux-gnu
  run: |
    cd rust_core
    cargo build --release
```

## Model Management

### Model Storage
```
/home/hamster/bot4/
├── models/
│   ├── sentiment/      # BERT models for sentiment
│   ├── prediction/     # Price prediction models
│   ├── rl/            # Reinforcement learning
│   └── checkpoints/   # Training checkpoints
```

### Model Loading
```rust
// Example: Loading a model with tch
use tch::{nn, Device, Tensor, vision};

let device = Device::cuda_if_available();
let vs = nn::VarStore::new(device);
let model = model::create_model(&vs.root());
vs.load("models/prediction/model.pt")?;
```

## Monitoring and Debugging

### Environment Check Script
```bash
#!/bin/bash
echo "LibTorch Environment Check"
echo "=========================="
echo "LIBTORCH: $LIBTORCH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""
echo "Files found:"
ls -la $LIBTORCH/libtorch* 2>/dev/null || echo "Not found"
echo ""
echo "Rust check:"
cd rust_core && cargo check -p ml 2>&1 | grep -E "error|warning" | head -5
```

### Debug Build
```bash
# Enable debug output
export RUST_LOG=debug
export RUST_BACKTRACE=full

# Build with debug symbols
cargo build -p ml

# Run with debugging
gdb target/debug/bot4-trading
```

## Best Practices

1. **Version Consistency**: Keep LibTorch version consistent across development and production
2. **Environment Management**: Use `.env` files or scripts to manage environment variables
3. **CI Testing**: Always test ML components in CI pipeline
4. **Model Versioning**: Version control trained models separately
5. **Resource Limits**: Set appropriate thread and memory limits
6. **Error Handling**: Gracefully handle LibTorch initialization failures

## Support

### Resources
- [PyTorch C++ API](https://pytorch.org/cppdocs/)
- [tch-rs Documentation](https://github.com/LaurentMazare/tch-rs)
- [rust-bert Documentation](https://github.com/guillaume-be/rust-bert)
- [candle Documentation](https://github.com/huggingface/candle)

### Common Questions

**Q: Can I use LibTorch without CUDA?**
A: Yes, use the CPU-only version. Our setup scripts default to CPU.

**Q: How much disk space does LibTorch require?**
A: CPU version: ~1.5GB, CUDA version: ~3.5GB

**Q: Can I use different LibTorch versions for different crates?**
A: No, all crates must use the same LibTorch version to avoid conflicts.

**Q: Is LibTorch required for all Bot4 components?**
A: No, only ML and data_intelligence crates require it. Other components work without it.

---

*Document maintained by: ML Team (Morgan, Jordan, Sam)*
*Last updated: August 24, 2025*