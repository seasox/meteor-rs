# syntax=docker/dockerfile:1.2

# Start with a rust alpine image
FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04 as builder

# Update default packages
RUN apt-get update

# Get Ubuntu packages
RUN apt-get install -y \
    build-essential \
    curl xz-utils pkg-config libssl-dev zlib1g-dev libtinfo-dev libxml2-dev \
    git

# Update new packages
RUN apt-get update

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

# get prebuilt llvm
RUN curl -O https://releases.llvm.org/7.0.1/clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz &&\
    xz -d /clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz &&\
    tar xf /clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04.tar &&\
    rm /clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04.tar &&\
    mv /clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04 /root/llvm

# set env
ENV LLVM_CONFIG=/root/llvm/bin/llvm-config
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_PATH=$CUDA_ROOT
ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info
ENV PATH=$CUDA_ROOT/nvvm/lib64:/root/.cargo/bin:$PATH

# make ld aware of necessary *.so libraries
RUN echo $CUDA_ROOT/lib64 >> /etc/ld.so.conf &&\
    echo $CUDA_ROOT/compat >> /etc/ld.so.conf &&\
    echo $CUDA_ROOT/nvvm/lib64 >> /etc/ld.so.conf &&\
    ldconfig

# This is important, see https://github.com/rust-lang/docker-rust/issues/85
ENV RUSTFLAGS="-C target-feature=-crt-static"
# set the workdir and copy the source into it
WORKDIR /app
# DEVEL: copy PWD to /app
# COPY ./ /app
# AI-LAB: pull repo
RUN git clone --recurse-submodules https://github.com/seasox/meteor-rs /app
# do a release build
RUN cargo build --release --bin meteor-rs
RUN strip target/release/meteor-rs

# use a plain alpine image, the alpine version needs to match the builder
FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04
# if needed, install additional dependencies here
# copy the binary into the final image
COPY --from=builder /app/target/release/meteor-rs .