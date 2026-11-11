# Multi-stage Dockerfile for high performance Apache TVM compilation & deployment
# Optimized for GPU/CUDA and CPU acceleration backends

# =====================================================================
# Stage 1: Build environment (Compilation and dependencies)
# =====================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools, LLVM, Python, and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    llvm-15 \
    llvm-15-dev \
    clang-15 \
    libgoogle-glog-dev \
    libgtest-dev \
    libssl-dev \
    zlib1g-dev \
    libtinfo-dev \
    libedit-dev \
    libxml2-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    & \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for compilation tools
ENV CC=clang-15
ENV CXX=clang++-15

# Install PyTorch, ONNX, and necessary python packages for importing & verification
RUN pip3 install --no-cache-dir \
    numpy<2.0.0 \
    scipy \
    onnx \
    onnxruntime \
    torch --index-url https://download.pytorch.org/whl/cu121 \
    decorator \
    attrs \
    tornado \
    psutil \
    xgboost \
    cloudpickle

# Clone Apache TVM repository (using TVM Unity/Relax stable release/v0.15+)
WORKDIR /workspace
RUN git clone --recursive https://github.com/apache/tvm.git tvm-source && \
    cd tvm-source && \
    git checkout v0.15.0

# Set up build configurations for TVM
WORKDIR /workspace/tvm-source/build
RUN cp ../cmake/config.cmake . && \
    # Enable LLVM back-end (crucial for CPU optimization/codegen)
    sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/g' config.cmake && \
    # Enable CUDA support (crucial for NVIDIA GPU optimization/codegen)
    sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/g' config.cmake && \
    # Enable C++ RPC server/client
    sed -i 's/set(USE_RPC OFF)/set(USE_RPC ON)/g' config.cmake && \
    # Enable modern C++ Runtime support
    sed -i 's/set(USE_MICRO OFF)/set(USE_MICRO ON)/g' config.cmake

# Build the Apache TVM core compiler library and runtime
RUN cmake .. && make -j$(nproc)

# Install Python bindings for the compiled TVM source
WORKDIR /workspace/tvm-source/python
RUN python3 setup.py install

# =====================================================================
# Stage 2: Clean, ultra-lean Runtime Environment (No source files)
# =====================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime execution dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgomp1 \
    libtinfo6 \
    libedit2 \
    ca-certificates \
    & \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy compiled libraries and python environment from the builder stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/
COPY --from=builder /workspace/tvm-source/build/libtvm.so /usr/local/lib/
COPY --from=builder /workspace/tvm-source/build/libtvm_runtime.so /usr/local/lib/

# Set up system library paths
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV TVM_HOME=/workspace/tvm-source

WORKDIR /app
COPY . /app

CMD ["python3", "tvm_compiler.py"]
