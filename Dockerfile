FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    zip \
    unzip \
    pkg-config \
    wget \
    ninja-build \
    python3 \
    python3-pip

# Install vcpkg
WORKDIR /opt
RUN git clone https://github.com/Microsoft/vcpkg.git && \
    cd vcpkg && \
    ./bootstrap-vcpkg.sh && \
    ./vcpkg integrate install

# Install dependencies via vcpkg
RUN /opt/vcpkg/vcpkg install \
    torch \
    libtorch \
    grpc \
    protobuf \
    prometheus-cpp \
    eigen3 \
    yaml-cpp \
    gtest \
    spdlog \
    cli11

# Set up working directory
WORKDIR /app
COPY . .

# Build
RUN cmake -B build -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake && \
    cmake --build build --config Release -j$(nproc)

# Add to path
ENV PATH="/app/build:${PATH}"

# Default command
ENTRYPOINT ["rldronesim"]
CMD ["--help"]
